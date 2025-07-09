# FP8 GEMM Kernel — AMD MI300X
Compact HIP + rocWMMA kernel that multiplies FP8 matrices and stores BF16.  
Tests show **well above 400× speed‑up** compared with a straightforward FP32 implementation.

## Technology Stack:
- Programming Language - HIP (C++), Python
- GPU - AMD MI300X

## Build / Run
The competition was run through Discord submission to the servers.
To submit on Discord, make sure the first line of `fp8gemm.py` reads

```python
#!POPCORN leaderboard amd-fp8-mm
```

Full submission guide:  
<https://gpu-mode.github.io/discord-cluster-manager/docs/category/submitting-your-first-kernel>

## Kernel Highlights
* **Block tile** — each thread block computes a 128 × 128 patch of *C*.  
* **Matrix‑core op** — 32 × 32 × 16 MFMA per warp.  
* **Shared memory** — K dimension is processed in two halves for load/compute overlap.  
* **Scaling** — per‑row factor from *A* times block factor from *B*, fused once per tile.

## Results
The kernel achieved more than **400x** speedup over the HIP baseline.

## Challenge Reference
Entry for the FP8 GEMM task in the AMD Developer Challenge 2025.  
Details: <https://www.datamonsters.com/amd-developer-challenge-2025>
