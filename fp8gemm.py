#!POPCORN leaderboard amd-fp8-mm

import os
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")  # target MI300

# -----------------------------------------------------------------------------
# 1. C++ wrapper – declares the symbol that Python will call
# -----------------------------------------------------------------------------
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

# -----------------------------------------------------------------------------
# 2. Inline HIP source – contains the full kernel
# -----------------------------------------------------------------------------
CUDA_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_cooperative_groups.h>
#include <rocwmma/rocwmma.hpp>
#include <torch/extension.h>

namespace cg = cooperative_groups;

// Tuning Parameters
#define TILE_K_JUMP 128
#define FRAG_M 32
#define FRAG_N 32
#define FRAG_K 16
#define NUM_WARPS_X 4
#define NUM_WARPS_Y 4
#define THREADS_PER_WARP 64
#define UNUSED_PLACEHOLDER 4

// Main HIP kernel – one block computes a 128×128 tile of C
__global__ void __launch_bounds__(1024, 8) custom_kernel(const __hip_fp8_e4m3_fnuz* A_ptr, const __hip_fp8_e4m3_fnuz* B_ptr, const float* As_ptr, const float* Bs_ptr, rocwmma::bfloat16_t* C_ptr, int M, int N, int K) {    

    // Shared‑memory tiles
    __shared__ __hip_fp8_e4m3_fnuz smem_tileA[128][64]; // row-major
    __shared__ __hip_fp8_e4m3_fnuz smem_tileB[128][64]; // col-major, = row-major transposed
    int lda = 64;
    int ldb = 64;
    __shared__ float smem_tileAs[128];

    // Accumulator fragments
    auto frag_acc_master = rocwmma::fragment<rocwmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float>();
    rocwmma::fill_fragment(frag_acc_master, 0.f);

    // Warp id helpers
    cg::thread_block this_tb = cg::this_thread_block();
    auto warp_tile = cg::tiled_partition<THREADS_PER_WARP>(this_tb); // 1024 threads -> 64 x 16
    int lane_id = warp_tile.thread_rank();
    int warp_id = warp_tile.meta_group_rank(); 

    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += TILE_K_JUMP) {
        // FIRST 64 columns of this 128‑slice
        #pragma unroll
        for (int kmini_load = 0; kmini_load < 64; kmini_load += 16) {
            #pragma unroll
            for (int idx = 0; idx < 2; idx++) {
                int offset = idx * 64;
                smem_tileA[lane_id + offset][kmini_load + warp_id] =
                    A_ptr[blockIdx.y * FRAG_M * NUM_WARPS_Y + lane_id + offset + M * (k_tile_base + kmini_load + warp_id)];
                smem_tileB[lane_id + offset][kmini_load + warp_id] =
                    B_ptr[(k_tile_base + kmini_load + warp_id) * N + blockIdx.x * FRAG_N * NUM_WARPS_X + lane_id + offset];
            }
        }

        // Pre‑compute row‑wise scale once per 128×128 C‑tile (warp row 0 does it)
        if (warp_id == 0) {
            int sn = (N + 128 - 1) / 128;    
            float tileBs = Bs_ptr[(k_tile_base / 128) * sn + ((blockIdx.x * NUM_WARPS_X * FRAG_N) / 128)];
            float2 val = reinterpret_cast<const float2*>(As_ptr + blockIdx.y * FRAG_M * NUM_WARPS_Y + M * (k_tile_base / 128))[lane_id];
            reinterpret_cast<float2*>(smem_tileAs)[lane_id] = make_float2(val.x * tileBs, val.y * tileBs);
        }       

        __syncthreads();

        // Use matrix cores to compute 32×32 per‑warp fragment for two 64‑column halves
        auto frag_acc_block = rocwmma::fragment<rocwmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float>();
        rocwmma::fill_fragment(frag_acc_block, 0.f);

        auto frag_matA = rocwmma::fragment<rocwmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, __hip_fp8_e4m3_fnuz, rocwmma::row_major>();
        auto frag_matB = rocwmma::fragment<rocwmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, __hip_fp8_e4m3_fnuz, rocwmma::col_major>(); 

        int warp_idx_y = threadIdx.y;
        int warp_idx_x = threadIdx.x / THREADS_PER_WARP;

        #pragma unroll 2
        for (int k_inner = 0; k_inner < 64; k_inner += FRAG_K) {           
            rocwmma::load_matrix_sync(frag_matA, &smem_tileA[0][0] + (warp_idx_y * FRAG_M) * lda + k_inner, lda);            
            rocwmma::load_matrix_sync(frag_matB, &smem_tileB[0][0] + k_inner + ldb * (warp_idx_x * FRAG_N), ldb);
            rocwmma::mma_sync(frag_acc_block, frag_matA, frag_matB, frag_acc_block);
        }

        __syncthreads();

        // SECOND 64 columns of this 128‑slice
        #pragma unroll
        for (int kmini_load = 0; kmini_load < 64; kmini_load += 16){
            #pragma unroll
            for (int idx = 0; idx < 2; idx++) {
                int offset = idx * 64;
                smem_tileA[lane_id + offset][kmini_load + warp_id] = 
                    A_ptr[blockIdx.y * FRAG_M * NUM_WARPS_Y + lane_id + offset + M * (k_tile_base + kmini_load + 64 + warp_id)];
                smem_tileB[lane_id + offset][kmini_load + warp_id] = 
                    B_ptr[(k_tile_base + kmini_load + 64 + warp_id) * N + blockIdx.x * FRAG_N * NUM_WARPS_X + lane_id + offset];
            }
        }
        __syncthreads();

        // Scale‑and‑accumulate into the master fragment
        #pragma unroll
        for (int k_inner = 0; k_inner < 64; k_inner += FRAG_K) {           
            rocwmma::load_matrix_sync(frag_matA, &smem_tileA[0][0] + (warp_idx_y * FRAG_M) * lda + k_inner, lda);            
            rocwmma::load_matrix_sync(frag_matB, &smem_tileB[0][0] + k_inner + ldb * (warp_idx_x * FRAG_N), ldb);
            rocwmma::mma_sync(frag_acc_block, frag_matA, frag_matB, frag_acc_block);
        }

        // Apply scaling 
        #pragma unroll
        for (int i = 0; i < 4; ++i){
            #pragma unroll
            for (int j = 0; j < 4; ++j){
                frag_acc_master.x[i * 4 + j] += smem_tileAs[warp_idx_y * FRAG_M + 4 * (lane_id / 32) + 8 * i + j] * frag_acc_block.x[i * 4 + j];
            }
        }            

        __syncthreads();
    } 

    int warp_idx_x = threadIdx.x / THREADS_PER_WARP;    
    int col_base_C = (blockIdx.x * NUM_WARPS_X + warp_idx_x) * FRAG_N;
    if (col_base_C >= N) return;    

    int warp_idx_y = threadIdx.y;
    int row_base_C = (blockIdx.y * NUM_WARPS_Y + warp_idx_y) * FRAG_M;
    if (row_base_C >= M) return;

    // FP32 to BF16 conversion and writeback
    auto frag_result = rocwmma::fragment<rocwmma::accumulator, FRAG_M, FRAG_N, UNUSED_PLACEHOLDER, rocwmma::bfloat16_t>();

    #pragma unroll
    for (int i = 0; i < frag_acc_master.num_elements; ++i) {
        frag_result.x[i] = (rocwmma::bfloat16_t)frag_acc_master.x[i];
    }

    rocwmma::store_matrix_sync(C_ptr + row_base_C * N + col_base_C, frag_result, N, rocwmma::mem_row_major);
}

// Host stub callable from Python
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int M = a.size(0);
    int N = b.size(0);
    int K = a.size(1);
    int num_warp_x = NUM_WARPS_X;
    int num_warp_y = NUM_WARPS_Y;
    dim3 block_size((THREADS_PER_WARP * num_warp_x), num_warp_y);
    dim3 grid_size((N + num_warp_x * FRAG_N - 1) / (num_warp_x * FRAG_N), 
                   (M + num_warp_y * FRAG_M - 1) / (num_warp_y * FRAG_M));

    custom_kernel<<<grid_size, block_size, 0, 0>>> (
        (__hip_fp8_e4m3_fnuz*)a.data_ptr(), 
        (__hip_fp8_e4m3_fnuz*)b.data_ptr(), 
        as.data_ptr<float>(), 
        bs.data_ptr<float>(), 
        (rocwmma::bfloat16_t*)c.data_ptr(), 
        M, N, K
    );
    //C10_CUDA_CHECK(cudaGetLastError());
}
"""

import os

os.environ["CXX"] = "clang++"

# -----------------------------------------------------------------------------
# 3. Compile the extension
# -----------------------------------------------------------------------------
module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)

# -----------------------------------------------------------------------------
# 4. Custom kernel which is need by the evaluator
# -----------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c
