# Limitations

* **No GPU: performance is BLOCKED.**
  * Nothing here was timed on a GPU, and no GFLOP/s or GB/s figures are claimed.
  * `bench/gpu_bench.cu` is the harness that would produce them. It builds and links for sm_80 and sm_89 (`make gpu-bench CUDA_ARCH=sm_89`) but has never run. The owner's RTX 4060 run is described in [OWNER_ACTIONS.md](OWNER_ACTIONS.md).
  * [analysis.md](analysis.md) has compiler resource reports and modelled memory behaviour instead.
* **The tensor-core fragment layouts were checked only against NVIDIA's documentation.**
  * The emulator's model of `mma.m16n8k16` and `ldmatrix` follows the PTX ISA tables. The tests show the kernel and the model agree, and a mutated layout fails.
  * If the model misreads the documentation in the same way the kernel does, the tests cannot catch it. The GPU benchmark's `mma_probe` and odd-shape checks would, but they have not run on hardware.
  * The emulator sums the 16 products of each mma in fp32 in k order. Real tensor cores use an unspecified order and truncate inside the adder. The test tolerance, (K + 2)·2⁻²³·Σ|aᵢₖbₖⱼ|, is set to cover both; the emulator's actual error is about 1 % of it.
* **The tensor-core GEMM is a first, simple version.**
  * It has no `cp.async` multi-stage pipeline, no swizzled shared layout (the A-tile stores have 2-way bank conflicts), no split-K, no autotuning and no bf16/FP8.
  * It uses 122–126 registers, so only 2 blocks fit per SM.
  * On H100 it uses Ampere-style `mma.sync`. Hopper needs `wgmma` and TMA to reach full tensor-core throughput.
  * A and B are row-major fp16; C is fp32. There is no fp16 output and no fused epilogue beyond α/β.
  * There is no cuBLAS comparison, because the harness does not link cuBLAS.
* **The emulated metrics are a model.**
  * Sector counts and bank-conflict wavefronts follow the documented hardware rules. They ignore caches, replays, the L1/L2 hierarchy, instruction issue and scheduling.
  * They show *why* one variant should be faster, not *how much*.
* **Precision.** The original kernels are fp32 only. The only reduced-precision kernel is the fp16 tensor-core GEMM.
* **Backward passes.**
  * Only softmax and LayerNorm have backward kernels. Attention has none.
  * The LayerNorm dγ/dβ kernel uses one thread per column and loops over all rows. With many rows and few columns it uses few threads. A production kernel would split the rows across blocks and add a second reduction pass.
* **The flash kernel is FlashAttention-1-style:**
  * one thread per query row;
  * 245 registers per thread, so 12.5 % occupancy;
  * no split-K / sequence parallelism and no backward pass.
* **The register-blocked SGEMM** has no double buffering (cp.async), no warp tiling and no autotuning of tile sizes.
* **The emulator** runs blocks sequentially and threads with OS barriers. It cannot reveal inter-block races, warp-divergence effects, or memory-ordering bugs outside barriers.
* **The toolchain is clang plus NVIDIA's ptxas and libnvfatbin** from PyPI, not nvcc. nvcc may make different register and unrolling decisions. The nvcc build line in the owner script has not been tried here, because the PyPI wheels do not ship nvcc.
* **The SASS disassembler** (`cuobjdump`/`nvdisasm`) comes from the CUDA 13.2 wheels, because NVIDIA does not publish CUDA 12 versions on PyPI. The cubins themselves come from ptxas 12.9.
