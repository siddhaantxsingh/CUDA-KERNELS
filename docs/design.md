# Design

## Kernels (`include/kc/`)

All kernels are row-major and header-only. They are fp32, except the tensor-core GEMM (fp16 inputs). Each comes with a host launcher. They are written as progressions, so each step's effect can be seen in [analysis.md](analysis.md).

| File | Kernels | Technique |
|---|---|---|
| `reduce.cuh` | atomic → shared-memory tree → float4 + warp shuffles, two-pass | grid-stride loops, `__shfl_xor_sync`, deterministic second pass |
| `transpose.cuh` | naive → shared tile → padded shared tile | coalescing through shared memory; bank-conflict padding |
| `gemm.cuh` | naive → 32×32 shared tiles → 128×128×8 register-blocked (8×8 per thread), contiguous and *split* layouts | register blocking, transposed A tile, float4 shared loads, conflict-free split micro-tiles, float4 epilogue |
| `softmax.cuh` | online softmax, Welford LayerNorm (+ naive E[x²]−E[x]² variant) | warp per row; single-pass statistics merged with shuffles |
| `hgemm.cuh` + `mma.cuh` | tensor-core GEMM: fp16 A, B (raw bits in `uint16_t`), fp32 accumulate and C; 128×128×32 block tile, 8 warps of 64×32, `ldmatrix.x4` (A) and `ldmatrix.x4.trans` (B) feeding `mma.sync.m16n8k16` | inline PTX; 16-byte padded shared rows for conflict-free `ldmatrix`; 16-byte global loads when K, N are multiples of 8, element loads with zero fill otherwise |
| `backward.cuh` | softmax backward; LayerNorm backward (dx per row; dγ, dβ per column, deterministic) | warp per row, shuffles; LayerNorm reuses the forward Welford statistics code |
| `attention.cuh` | tiled "flash" attention (online softmax, optional causal mask) and an unfused baseline that materialises scores | IO-aware tiling; Q/O staged through padded shared memory |

* **Correctness under all shapes.** Every kernel handles arbitrary sizes (guards at tile edges). Softmax never overflows: it handles inputs up to FLT_MAX and masks `-inf`. LayerNorm uses Welford/Chan merging, so a large mean does not cancel the variance.
* **`KC_UNROLL`** (`#pragma unroll` under CUDA) marks fixed-trip loops over per-thread arrays. Without it ptxas put those arrays in local memory; see [analysis.md](analysis.md).

## One source, three builds

`include/kc/compat.h` selects the environment:

* **`clang++ -x cuda` / `nvcc`.** Real CUDA. `KC_G(x)` and `KC_S(x)` expand to `x`, and `KC_LAUNCH` expands to `<<<grid, block>>>`.
* **Host C++ (`g++`).** `include/kc/simt_emu.h` supplies `threadIdx`, `__syncthreads`, shuffles, atomics and `float4`, and runs the kernels on the CPU.
* **Host C++ with tracing.** The same, with `kc_emu::g_trace` set: every `KC_G` / `KC_S` access is recorded.

## The emulator

* **Threads.** A thread block runs as `blockDim` real OS threads, from a persistent pool. Blocks run one after another.
* **Barriers.** `__syncthreads()` is a `std::barrier` across the block, so a missing barrier produces a real data race, which the tests detect (checked by mutation).
* **Shared memory.** `__shared__` becomes `static`: shared by the threads of the running block, and safe because blocks do not overlap.
* **Shuffles** go through a per-warp exchange buffer between two per-warp barriers. They assume the whole warp takes part, as every shuffle in these kernels does.
* **Grouping accesses into warp instructions.** The tracer groups the recorded accesses by (warp, call site, nth execution of that site by the thread). A lane that skips a predicated access simply has no entry, so it is excluded from that instruction, as on the hardware.
* **Global memory.** It counts the distinct 32-byte sectors per instruction and the distinct bytes used.
* **Shared memory.**
  * Accesses are split into phases: 32 lanes for 4-byte, 16 for 8-byte, 8 for 16-byte accesses.
  * Each phase costs as many wavefronts as the largest number of distinct 4-byte words that fall in any one of the 32 banks. Broadcasts are free.
  * The minimum is one wavefront per phase.
* **Tensor-core instructions.** `ldmatrix` and `mma.m16n8k16` are modelled at warp level, through a per-warp exchange buffer between two warp barriers, like shuffles:
  * `ldmatrix`: each lane posts its row address. Each lane then picks up its two 16-bit elements per matrix, following the PTX ISA layout, transposed for `.trans`. The tracer records it as one 16-byte shared access per lane.
  * `mma`: each lane posts its A and B fragments. The model rebuilds the 16×16 and 16×8 matrices from the documented fragment layout. Each lane then computes its 4 outputs in fp32, in k order.
  * A kernel that loads fragments in the wrong layout therefore gives wrong numbers on the emulator.
* **Validation.** `tests/test_tracer.cpp` pins this model to known patterns: stride 1/2/32, misalignment, float4, broadcast, predication, and bank strides 1/2/32/33.

## Device toolchain without a GPU

`tools/fetch_cuda.sh` assembles a CUDA toolkit root from NVIDIA's PyPI wheels: headers, libdevice, `ptxas`, cudart and libnvfatbin.

* clang 18 compiles the kernels to PTX.
* `ptxas` assembles them to SASS for sm_80, sm_89 and sm_90.
* `cuobjdump -sass` (CUDA 13.2 PyPI wheel; it calls `nvdisasm`) disassembles the cubins; `tools/sass_check.py` counts HMMA/LDSM per kernel. `make ptx` then reports registers, shared memory and spills.
* clang also needs a `fatbinary` program to embed device code in a host executable. The wheels do not include one, so `tools/fatbinary_shim.c` provides it on top of NVIDIA's libnvfatbin.
* With that, `make gpu-bench` links a complete CUDA program, `bench/gpu_bench.cu`, with an `.nv_fatbin` section. On this machine it exits with "CUDA driver version is insufficient", because there is no GPU or driver.
