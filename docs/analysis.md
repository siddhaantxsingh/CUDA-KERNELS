# Static and emulated analysis

No NVIDIA GPU was available, so nothing here is a timing. There are two kinds of evidence:

1. **Compiler output.** Every kernel was compiled as real CUDA by clang 18 and NVIDIA `ptxas` 12.9 for sm_80 (A100), sm_89 (Ada: RTX 40xx, L4) and sm_90 (H100) with `make ptx`. That gives registers, shared memory, stack and spills per kernel. Theoretical occupancy follows from the per-SM limits in the CUDA programming guide.
2. **Memory-access traces.** `make analyze` runs each kernel on the CPU SIMT emulator with tracing on; see [design.md](design.md#the-emulator). From the addresses each warp instruction touches, it derives:
   * the 32-byte global-memory sectors per request, and *coalescing* (distinct bytes used ÷ bytes in the touched sectors);
   * the shared-memory wavefronts, and the *conflict factor* (wavefronts ÷ the minimum for the access width; 1 = conflict-free, 32 = fully serialised).

   These follow the documented hardware rules. They are what Nsight Compute reports as sectors per request and bank conflicts, *computed from a model*. They are not measured. Caches are not modelled, so *bytes requested* counts every instruction's bytes, including those an L1 or L2 hit would serve.

## Memory behaviour (emulator traces)

| kernel | problem | global requests | sectors | coalescing | bytes requested | shared requests | conflict factor |
|---|---|---|---|---|---|---|---|
| reduce: atomic | n=262144 | 8,192 | 32,768 | 100.0 % | 1.05 MB | 0 | 1.00 |
| reduce: smem tree | n=262144 | 8,192 | 32,768 | 100.0 % | 1.05 MB | 33,792 | 1.00 |
| reduce: float4 + shuffles, two-pass | n=262144 | 2,313 | 33,057 | 99.3 % | 1.05 MB | 2,313 | 1.00 |
| transpose: naive | 1024x1024 | 65,536 | 1,179,648 | 22.2 % | 8.39 MB | 0 | 1.00 |
| transpose: smem tile | 1024x1024 | 65,536 | 262,144 | 100.0 % | 8.39 MB | 65,536 | 16.50 |
| transpose: smem tile, padded | 1024x1024 | 65,536 | 262,144 | 100.0 % | 8.39 MB | 65,536 | 1.00 |
| sgemm: naive | 256x256x256 | 1,050,624 | 2,629,632 | 82.5 % | 69.47 MB | 0 | 1.00 |
| sgemm: smem 32x32 | 256x256x256 | 34,816 | 139,264 | 100.0 % | 4.46 MB | 1,081,344 | 1.00 |
| sgemm: register 8x8 | 256x256x256 | 10,240 | 98,304 | 41.7 % | 1.31 MB | 40,960 | 1.47 |
| sgemm: register 8x8, split | 256x256x256 | 8,704 | 40,960 | 100.0 % | 1.31 MB | 40,960 | 1.00 |
| hgemm: tensor-core mma.m16n8k16 + ldmatrix (fp16 in, fp32 acc) | 256x256x256 | 2,048 | 24,576 | 100.0 % | 0.79 MB | 4,096 | 1.12 |
| rows: softmax (online) | 256x1024 | 24,576 | 98,304 | 100.0 % | 3.15 MB | 0 | 1.00 |
| rows: layernorm (Welford) | 256x1024 | 40,960 | 163,840 | 100.0 % | 5.24 MB | 0 | 1.00 |
| rows: softmax backward | 256x1024 | 40,960 | 163,840 | 100.0 % | 5.24 MB | 0 | 1.00 |
| rows: layernorm backward (dx, dgamma, dbeta) | 256x1024 | 98,880 | 344,832 | 95.7 % | 10.56 MB | 0 | 1.00 |
| attention: flash (tiled, online softmax) | BH=1 N=512 D=64 | 18,432 | 73,728 | 100.0 % | 2.36 MB | 1,069,056 | 1.00 |
| attention: naive (scores in global memory) | BH=1 N=512 D=64 | 2,122,752 | 35,422,208 | 12.5 % | 141.69 MB | 0 | 1.00 |

What the traces show:

* **Transpose.**
  * The naive kernel's strided stores use 22 % of each sector.
  * Staging through a shared tile makes every global access fully coalesced. But reading a tile column puts all 32 lanes on one bank: 32-way conflicts on half the shared accesses (16.5 on average).
  * One padding column removes them: conflict factor 1.00.
* **SGEMM.**
  * Global bytes requested fall from 69 MB (naive) to 4.5 MB (32×32 tiles) to 1.3 MB (128×128 register tiles). Every operand fetched from global memory is reused 32 and then 128 times.
  * The register-blocked kernel with contiguous 8×8 micro-tiles has 2-way bank conflicts on its B reads, because lanes tx and tx+4 hit the same banks. Its scalar C stores are 42 % coalesced.
  * The *split* layout (two 4-wide halves 64 apart) and float4 stores bring both to ideal: conflict factor 1.00 and 100 % coalescing.
  * Padding the transposed A tile by 4 floats removed an 8-way conflict on the A-tile stores. The tracer found it, and it is fixed in both variants.
* **Attention.**
  * The tiled (flash) kernel requests 2.4 MB at N = 512, against 142 MB for the unfused kernel, which writes the N × N scores to global memory and reads them back.
  * Staging Q and O through padded shared memory makes the flash kernel's global accesses 100 % coalesced. The first version loaded one row per thread, which used one sector per lane.
  * The unfused kernel's per-thread row accesses are 12.5 % coalesced.
* **Tensor-core GEMM (fp16 in, fp32 accumulate).**
  * Per element of A and B it requests 2 bytes instead of 4, and each 128×128×32 block tile feeds 32 `mma.m16n8k16` per warp per k-tile.
  * The `ldmatrix` reads are conflict-free (conflict factor 1.00 on that site). This comes from padding every shared row by 16 bytes, so the 8 row addresses of each 8×8 matrix fall in distinct banks.
  * The 16-byte stores of the A tile into shared memory are 2-way conflicted: two rows of 4 chunks share one 8-lane phase, and the 80-byte row stride puts them on overlapping banks. That gives the 1.12 kernel-wide factor. An XOR swizzle, as CUTLASS uses, would remove it; it is not done here.
* **Backward passes.** Softmax and LayerNorm backward read and write fully coalesced. The LayerNorm dγ/dβ kernel is one thread per column, looping over rows. Its per-row mean and rstd loads are broadcasts, which lowers the coalescing figure slightly.
* **Reduction.** All three variants read the input coalesced. The difference between them is the number of atomics and barriers (atomic: one atomic per element; tree: 9 barriers and one atomic per block; two-pass: no atomics, deterministic), which this model does not price.

## Registers, shared memory and occupancy (ptxas)

| kernel | arch | registers | shared memory | stack / spills | threads per block | blocks per SM | occupancy | limited by |
|---|---|---|---|---|---|---|---|---|
| `kc::attn_naive_kernel<64>` | sm_80 | 31 | 0 B | 0 / 0 B | 64 | 32 | 100.0 % | threads |
| `kc::attn_naive_kernel<64>` | sm_89 | 39 | 0 B | 0 / 0 B | 64 | 24 | 100.0 % | threads |
| `kc::attn_naive_kernel<64>` | sm_90 | 31 | 0 B | 0 / 0 B | 64 | 32 | 100.0 % | threads |
| `kc::flash_attn_kernel<32, 64, 32>` | sm_80 | 127 | 16,640 B | 0 / 0 B | 64 | 8 | 25.0 % | registers |
| `kc::flash_attn_kernel<32, 64, 32>` | sm_89 | 167 | 16,640 B | 0 / 0 B | 64 | 5 | 20.8 % | shared memory |
| `kc::flash_attn_kernel<32, 64, 32>` | sm_90 | 127 | 16,640 B | 0 / 0 B | 64 | 8 | 25.0 % | registers |
| `kc::flash_attn_kernel<64, 64, 32>` | sm_80 | 245 | 33,024 B | 0 / 0 B | 64 | 4 | 12.5 % | registers |
| `kc::flash_attn_kernel<64, 64, 32>` | sm_89 | 247 | 33,024 B | 0 / 0 B | 64 | 3 | 12.5 % | shared memory |
| `kc::flash_attn_kernel<64, 64, 32>` | sm_90 | 247 | 33,024 B | 0 / 0 B | 64 | 4 | 12.5 % | registers |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, false>` | sm_80 | 123 | 18,944 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, false>` | sm_89 | 122 | 18,944 B | 0 / 0 B | 256 | 2 | 33.3 % | registers |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, false>` | sm_90 | 123 | 18,944 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, true>` | sm_80 | 124 | 18,944 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, true>` | sm_89 | 126 | 18,944 B | 0 / 0 B | 256 | 2 | 33.3 % | registers |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, true>` | sm_90 | 126 | 18,944 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::layernorm_bwd_cols_kernel` | sm_80 | 32 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::layernorm_bwd_cols_kernel` | sm_89 | 35 | 0 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::layernorm_bwd_cols_kernel` | sm_90 | 32 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::layernorm_bwd_rows_kernel` | sm_80 | 32 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::layernorm_bwd_rows_kernel` | sm_89 | 38 | 0 B | 0 / 0 B | 128 | 12 | 100.0 % | threads |
| `kc::layernorm_bwd_rows_kernel` | sm_90 | 32 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::layernorm_warp_kernel<false>` | sm_80 | 30 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::layernorm_warp_kernel<false>` | sm_89 | 31 | 0 B | 0 / 0 B | 128 | 12 | 100.0 % | threads |
| `kc::layernorm_warp_kernel<false>` | sm_90 | 32 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::layernorm_warp_kernel<true>` | sm_80 | 32 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::layernorm_warp_kernel<true>` | sm_89 | 33 | 0 B | 0 / 0 B | 128 | 12 | 100.0 % | threads |
| `kc::layernorm_warp_kernel<true>` | sm_90 | 32 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::mma_probe_kernel` | sm_80 | 24 | 0 B | 0 / 0 B | 32 | 32 | 50.0 % | blocks |
| `kc::mma_probe_kernel` | sm_89 | 26 | 0 B | 0 / 0 B | 32 | 24 | 50.0 % | blocks |
| `kc::mma_probe_kernel` | sm_90 | 24 | 0 B | 0 / 0 B | 32 | 32 | 50.0 % | blocks |
| `kc::reduce_atomic_kernel` | sm_80 | 8 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_atomic_kernel` | sm_89 | 8 | 0 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::reduce_atomic_kernel` | sm_90 | 8 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_final_kernel<256>` | sm_80 | 16 | 32 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_final_kernel<256>` | sm_89 | 16 | 32 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::reduce_final_kernel<256>` | sm_90 | 16 | 32 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_partials_kernel<256>` | sm_80 | 17 | 32 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_partials_kernel<256>` | sm_89 | 17 | 32 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::reduce_partials_kernel<256>` | sm_90 | 18 | 32 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_smem_kernel<256>` | sm_80 | 10 | 1,024 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::reduce_smem_kernel<256>` | sm_89 | 10 | 1,024 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::reduce_smem_kernel<256>` | sm_90 | 10 | 1,024 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::sgemm_naive_kernel` | sm_80 | 32 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::sgemm_naive_kernel` | sm_89 | 40 | 0 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::sgemm_naive_kernel` | sm_90 | 32 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::sgemm_reg_kernel<128, 128, 8, 8, 8, false>` | sm_80 | 128 | 8,320 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::sgemm_reg_kernel<128, 128, 8, 8, 8, false>` | sm_89 | 130 | 8,320 B | 0 / 0 B | 256 | 1 | 16.7 % | registers |
| `kc::sgemm_reg_kernel<128, 128, 8, 8, 8, false>` | sm_90 | 130 | 8,320 B | 0 / 0 B | 256 | 1 | 12.5 % | registers |
| `kc::sgemm_reg_kernel<128, 128, 8, 8, 8, true>` | sm_80 | 128 | 8,320 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::sgemm_reg_kernel<128, 128, 8, 8, 8, true>` | sm_89 | 128 | 8,320 B | 0 / 0 B | 256 | 2 | 33.3 % | registers |
| `kc::sgemm_reg_kernel<128, 128, 8, 8, 8, true>` | sm_90 | 128 | 8,320 B | 0 / 0 B | 256 | 2 | 25.0 % | registers |
| `kc::sgemm_smem_kernel<32>` | sm_80 | 32 | 8,192 B | 0 / 0 B | 1024 | 2 | 100.0 % | threads |
| `kc::sgemm_smem_kernel<32>` | sm_89 | 38 | 8,192 B | 0 / 0 B | 1024 | 1 | 66.7 % | threads |
| `kc::sgemm_smem_kernel<32>` | sm_90 | 32 | 8,192 B | 0 / 0 B | 1024 | 2 | 100.0 % | threads |
| `kc::softmax_bwd_warp_kernel` | sm_80 | 25 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::softmax_bwd_warp_kernel` | sm_89 | 27 | 0 B | 0 / 0 B | 128 | 12 | 100.0 % | threads |
| `kc::softmax_bwd_warp_kernel` | sm_90 | 24 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::softmax_warp_kernel` | sm_80 | 24 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::softmax_warp_kernel` | sm_89 | 23 | 0 B | 0 / 0 B | 128 | 12 | 100.0 % | threads |
| `kc::softmax_warp_kernel` | sm_90 | 24 | 0 B | 0 / 0 B | 128 | 16 | 100.0 % | threads |
| `kc::transpose_naive_kernel` | sm_80 | 20 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::transpose_naive_kernel` | sm_89 | 20 | 0 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::transpose_naive_kernel` | sm_90 | 20 | 0 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::transpose_tiled_kernel<0>` | sm_80 | 27 | 4,096 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::transpose_tiled_kernel<0>` | sm_89 | 26 | 4,096 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::transpose_tiled_kernel<0>` | sm_90 | 26 | 4,096 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::transpose_tiled_kernel<1>` | sm_80 | 25 | 4,224 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |
| `kc::transpose_tiled_kernel<1>` | sm_89 | 26 | 4,224 B | 0 / 0 B | 256 | 6 | 100.0 % | threads |
| `kc::transpose_tiled_kernel<1>` | sm_90 | 26 | 4,224 B | 0 / 0 B | 256 | 8 | 100.0 % | threads |

* **Tensor cores in the SASS.** `tools/sass_check.py` disassembles each cubin with `cuobjdump -sass` (from NVIDIA's PyPI wheels). It fails the build if a tensor-core kernel has no HMMA instruction:

| kernel | arch | HMMA (form) | LDSM | instructions |
|---|---|---|---|---|
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, false>` | sm_80 | 32 (HMMA.16816.F32) | 12 | 1144 |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, true>` | sm_80 | 32 (HMMA.16816.F32) | 12 | 952 |
| `kc::mma_probe_kernel` | sm_80 | 1 (HMMA.16816.F32) | 0 | 56 |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, false>` | sm_89 | 32 (HMMA.16816.F32) | 12 | 1144 |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, true>` | sm_89 | 32 (HMMA.16816.F32) | 12 | 952 |
| `kc::mma_probe_kernel` | sm_89 | 1 (HMMA.16816.F32) | 0 | 56 |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, false>` | sm_90 | 32 (HMMA.16816.F32) | 12 | 1232 |
| `kc::hgemm_mma_kernel<128, 128, 32, 64, 32, true>` | sm_90 | 32 (HMMA.16816.F32) | 12 | 1048 |
| `kc::mma_probe_kernel` | sm_90 | 1 (HMMA.16816.F32) | 0 | 56 |

  Per k-tile of 32, each warp issues 32 HMMA.16816.F32 (2 k-steps × 4 × 4 tiles) and 12 LDSM (2 × (4 for A + 2 transposed for B)). On sm_90 these are the Ampere-style warp-level instructions. Hopper's full tensor-core rate needs `wgmma` and TMA, which this kernel does not use.
* **The tensor-core GEMM** uses 122–126 registers (64 fp32 accumulators per thread), so 2 blocks of 256 threads fit per SM: 25 % occupancy on sm_80/sm_90, 33 % on sm_89, where an SM holds only 1536 threads.
* **No kernel uses stack or spills.** The first versions of the register-blocked SGEMM and the flash kernel had 288- and 640-byte stack frames: their per-thread arrays were in local memory because loops were not unrolled. `#pragma unroll` on the fixed-trip loops (`KC_UNROLL`) fixed that, as ptxas confirms.
* **The register-blocked SGEMM** uses 128 registers per thread, so 2 blocks (25 % occupancy) fit on sm_80. That is the usual trade for an 8×8 micro-tile: latency is hidden by instruction-level parallelism (64 independent FMAs per k) rather than by warps. On sm_90 the non-split variant needs 130 registers and drops to 1 block per SM.
* **The flash kernel** keeps q (64), o (64) and 32 scores per thread in registers: 245 registers and 12.5 % occupancy at D = 64. That is the cost of the thread-per-row design. FlashAttention-2 splits a row across a warp and uses tensor cores; see [limitations](limitations.md).
