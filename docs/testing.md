# Testing

`make test` builds and runs two programs. On a shared 4-vCPU machine this took about 2–4 minutes on 2026-10-15; see [evidence/2026-10-15/make_test.log](evidence/2026-10-15/make_test.log).

## `tests/test_tracer.cpp` (20 checks)

These pin the access model to patterns with known answers:

* global memory:
  * consecutive floats need 4 sectors;
  * stride 2 is 50 % efficient;
  * stride 32 needs one sector per lane;
  * a 4-byte misalignment needs 5 sectors;
  * float4 accesses need 16 sectors;
  * a broadcast needs 1 sector;
  * a predicated access forms one request per warp;
* shared memory:
  * strides 1 and 33 are conflict-free;
  * stride 2 is a 2-way conflict and stride 32 a 32-way conflict;
  * a broadcast is conflict-free;
  * consecutive float4s take 4 wavefronts, and float4s at stride 2 are a 2-way conflict.

## `tests/test_kernels.cpp` (150 checks: the original 100 plus 50 new ones)

Every kernel runs on the emulator and is compared with a double-precision reference.

| Kernel | Shapes | Tolerance |
|---|---|---|
| reductions | n = 1 … 2²⁰, odd sizes | fp32 error bound (serial terms + tree depth) × Σ\|x\|; the two-pass variant is bit-identical across runs |
| transpose (3 variants) | 1×1, 32×32, 33×65, 100×7, 256×512 | exact |
| SGEMM (4 variants) | 1×1×1, 128×128×8, 129×130×17, 64×300×64 (α = 0.5, β = 2), 256³ (β = 1), 5×257×3 (α = −1) | per element: (K + 2) · 6·10⁻⁸ · Σ\|aᵢₖbₖⱼ\| |
| softmax | 1×1 … 64×4096, values σ = 10, one 3·10³⁸, one −inf, all −inf row → NaN | 10⁻⁵ relative |
| LayerNorm | 1×1 … 8×768, 4×4096, mean 10⁴ | 10⁻⁴ + input-rounding term |
| attention D = 32, 64 | N = 1, 37, 130, 200; causal and not; BH up to 3 | 2·10⁻⁵ absolute |

| fp16 conversion | all 65 536 fp16 values; 2 M random floats and edge cases | exact; compared with GCC's `_Float16` |
| `mma.m16n8k16` probe (one warp, fragments gathered from the PTX tables) | 16×16×8, small integers | exact |
| tensor-core GEMM (`hgemm_mma`) | 1×1×1, 16×8×16, 128×128×32, 129×130×17, 64×300×64 (α = 0.5, β = 2), 256³ (β = 1), 5×257×3 (α = −1), 200×136×40, 33×72×8 (β = 0.5), 130×264×520 | (K + 2)·2⁻²³·Σ\|aᵢₖbₖⱼ\| (+ β term), against a double reference on the fp16-rounded inputs |
| softmax backward | 1×1 … 16×4096 | y·(\|dy\| + Σy\|dy\|)·(4·10⁻⁶ + (cols + 4)·6·10⁻⁸) |
| LayerNorm backward dx, dγ, dβ | 1×1 … 4×4096, 300×64, mean 3 | 4·(cols + 16)·6·10⁻⁸ × term magnitudes (dx); 4·(rows + cols + 16)·6·10⁻⁸·Σ\|dy·x̂\| (dγ) |

### Tensor-core GEMM tolerance

* Inputs are rounded to fp16 on the host. The reference uses the same fp16 values in double precision.
* A product of two fp16 values is exact in fp32. The only errors are the K fp32 additions and the α/β epilogue, so |error| ≤ (K + 2)·u·Σ|aᵢₖbₖⱼ|.
* u is 2⁻²³ instead of 2⁻²⁴ because real tensor cores truncate inside their adder (Fasi et al., "Numerical behavior of NVIDIA tensor cores", 2021). The same bound is used on the GPU.
* For 256³ the emulator's worst error is 0.009 of the bound (4·10⁻⁵ absolute).
* Against a reference built from the *unrounded* fp32 inputs, the error is about 500× larger (0.02). A check requires it to be more than 30× larger, which shows the inputs really go through fp16.
* Two more checks:
  * the element-load path (forced by a misaligned A) must equal the vector path bit for bit;
  * NaNs placed after A and B must not reach C. This proves no element past the K tail is loaded.

### Backward passes

* The kernels are compared with analytic formulas evaluated in double precision.
* The analytic formulas are checked separately against central finite differences of L = Σ dy·f(x), with f evaluated in double precision:
  * softmax: dx, h = 10⁻⁵, at 3 positions per row;
  * LayerNorm: dx on the first 2 rows, plus dγ and dβ on a 5×7 case.

The suite also checks two things about the naive LayerNorm (E[x²] − E[x]²): it must break down at mean 10⁴ (error more than 100× the bound), and it must pass on centred data. This shows why the Welford version exists.

**Mutation checks**, done by hand. Each of these changes makes the suite fail:

* removing the online-softmax rescale in attention or in softmax;
* breaking the Welford merge;
* removing a `__syncthreads()` from the tiled SGEMM (a race the emulator exposes);
* an off-by-one column in the split SGEMM epilogue;
* the causal key range in flash attention;
* dropping a tail element in the vectorised reduction.

Mutations of the new kernels were run by script; see [evidence/2026-10-15/mutations.log](evidence/2026-10-15/mutations.log). Each of these makes the suite fail:

* loading B with `ldmatrix` without `.trans`;
* swapping the B fragments of neighbouring tiles;
* a wrong lane-to-row mapping for A;
* removing a `__syncthreads()`;
* a wrong epilogue row for c2/c3;
* swapping a1 and a2 in the emulator's mma model;
* dropping a term from either backward formula;
* removing the K-tail guard in either load path. The first attempt at this mutation was *not* caught, because B's zero fill hid it. That led to the NaN-padding check.

## Device compilation (`make ptx`)

All 23 kernel instantiations compile with clang 18 and ptxas 12.9 for sm_80, sm_89 and sm_90, with zero stack and zero spills. `tools/sass_check.py` then disassembles the cubins and fails if the tensor-core kernels contain no HMMA instruction. CI runs this too.

## Not tested

* **Execution on a GPU.** There is no GPU here. `bench/gpu_bench.cu` checks GEMM results against the CPU when run on one, but it has never run; see [limitations](limitations.md).
* **Divergent warps and memory ordering.** The emulator runs each thread to completion between barriers. It does not reproduce warp-divergent scheduling or the GPU's weaker memory ordering outside barriers. The kernels here only communicate through barriers and shuffles.
