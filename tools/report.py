"""Render docs/analysis.md from results/analysis.json and results/resources.json."""
import json

A = json.load(open("results/analysis.json"))
R = json.load(open("results/resources.json"))
S = json.load(open("results/sass.json"))
L = ["# Static and emulated analysis\n",
     "No NVIDIA GPU was available, so nothing here is a timing. There are two kinds of evidence:\n",
     "1. **Compiler output.** Every kernel was compiled as real CUDA by clang 18 and NVIDIA `ptxas` 12.9 for sm_80 (A100), "
     "sm_89 (Ada: RTX 40xx, L4) and sm_90 (H100) with `make ptx`. That gives registers, shared memory, stack and spills per kernel. Theoretical occupancy "
     "follows from the per-SM limits in the CUDA programming guide.",
     "2. **Memory-access traces.** `make analyze` runs each kernel on the CPU SIMT emulator with tracing on; see "
     "[design.md](design.md#the-emulator). From the addresses each warp instruction touches, it derives:",
     "   * the 32-byte global-memory sectors per request, and *coalescing* (distinct bytes used ÷ bytes in the touched sectors);",
     "   * the shared-memory wavefronts, and the *conflict factor* (wavefronts ÷ the minimum for the access width; 1 = "
     "conflict-free, 32 = fully serialised).\n",
     "   These follow the documented hardware rules. They are what Nsight Compute reports as sectors per request and bank "
     "conflicts, *computed from a model*. They are not measured. Caches are not modelled, so *bytes requested* counts every "
     "instruction's bytes, including those an L1 or L2 hit would serve.\n",
     "## Memory behaviour (emulator traces)\n",
     "| kernel | problem | global requests | sectors | coalescing | bytes requested | shared requests | conflict factor |",
     "|---|---|---|---|---|---|---|---|"]
for r in A:
    L.append(f"| {r['group']}: {r['kernel']} | {r['shape']} | {r['global_requests']:,} | {r['global_sectors']:,} | "
             f"{100 * r['coalescing']:.1f} % | {r['global_bytes'] / 1e6:,.2f} MB | {r['shared_requests']:,} | {r['conflict_factor']:.2f} |")
L += ["",
      "What the traces show:\n",
      "* **Transpose.**",
      "  * The naive kernel's strided stores use 22 % of each sector.",
      "  * Staging through a shared tile makes every global access fully coalesced. But reading a tile column puts all 32 lanes on "
      "one bank: 32-way conflicts on half the shared accesses (16.5 on average).",
      "  * One padding column removes them: conflict factor 1.00.",
      "* **SGEMM.**",
      "  * Global bytes requested fall from 69 MB (naive) to 4.5 MB (32×32 tiles) to 1.3 MB (128×128 register tiles). Every "
      "operand fetched from global memory is reused 32 and then 128 times.",
      "  * The register-blocked kernel with contiguous 8×8 micro-tiles has 2-way bank conflicts on its B reads, because lanes tx "
      "and tx+4 hit the same banks. Its scalar C stores are 42 % coalesced.",
      "  * The *split* layout (two 4-wide halves 64 apart) and float4 stores bring both to ideal: conflict factor 1.00 and 100 % "
      "coalescing.",
      "  * Padding the transposed A tile by 4 floats removed an 8-way conflict on the A-tile stores. The tracer found it, and it "
      "is fixed in both variants.",
      "* **Attention.**",
      "  * The tiled (flash) kernel requests 2.4 MB at N = 512, against 142 MB for the unfused kernel, which writes the "
      "N × N scores to global memory and reads them back.",
      "  * Staging Q and O through padded shared memory makes the flash kernel's global accesses 100 % coalesced. The first "
      "version loaded one row per thread, which used one sector per lane.",
      "  * The unfused kernel's per-thread row accesses are 12.5 % coalesced.",
      "* **Tensor-core GEMM (fp16 in, fp32 accumulate).**",
      "  * Per element of A and B it requests 2 bytes instead of 4, and each 128×128×32 block tile feeds 32 `mma.m16n8k16` "
      "per warp per k-tile.",
      "  * The `ldmatrix` reads are conflict-free (conflict factor 1.00 on that site). This comes from padding every shared "
      "row by 16 bytes, so the 8 row addresses of each 8×8 matrix fall in distinct banks.",
      "  * The 16-byte stores of the A tile into shared memory are 2-way conflicted: two rows of 4 chunks share one 8-lane "
      "phase, and the 80-byte row stride puts them on overlapping banks. That gives the 1.12 kernel-wide factor. An XOR "
      "swizzle, as CUTLASS uses, would remove it; it is not done here.",
      "* **Backward passes.** Softmax and LayerNorm backward read and write fully coalesced. The LayerNorm dγ/dβ kernel is "
      "one thread per column, looping over rows. Its per-row mean and rstd loads are broadcasts, which lowers the coalescing "
      "figure slightly.",
      "* **Reduction.** All three variants read the input coalesced. The difference between them is the number of atomics and "
      "barriers (atomic: one atomic per element; tree: 9 barriers and one atomic per block; two-pass: no atomics, "
      "deterministic), which this model does not price.\n",
      "## Registers, shared memory and occupancy (ptxas)\n",
      "| kernel | arch | registers | shared memory | stack / spills | threads per block | blocks per SM | occupancy | limited by |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in sorted(R, key=lambda r: (r["kernel"], r["arch"])):
    L.append(f"| `{r['kernel']}` | {r['arch']} | {r['registers']} | {r['smem_bytes']:,} B | {r['stack_bytes']} / {r['spill_bytes']} B | "
             f"{r['threads_per_block']} | {r['blocks_per_sm']} | {100 * r['occupancy']:.1f} % | {r['limiter']} |")
L += ["",
      "* **Tensor cores in the SASS.** `tools/sass_check.py` disassembles each cubin with `cuobjdump -sass` (from NVIDIA's "
      "PyPI wheels). It fails the build if a tensor-core kernel has no HMMA instruction:\n",
      "| kernel | arch | HMMA (form) | LDSM | instructions |",
      "|---|---|---|---|---|"]
for r in S:
    if r["hmma"]:
        L.append(f"| `{r['kernel']}` | {r['arch']} | {r['hmma']} ({', '.join(r['hmma_forms'])}) | {r['ldsm']} | {r['instructions']} |")
L += ["",
      "  Per k-tile of 32, each warp issues 32 HMMA.16816.F32 (2 k-steps × 4 × 4 tiles) and 12 LDSM (2 × (4 for A + 2 "
      "transposed for B)). On sm_90 these are the Ampere-style warp-level instructions. Hopper's full tensor-core rate needs "
      "`wgmma` and TMA, which this kernel does not use.",
      "* **The tensor-core GEMM** uses 122–126 registers (64 fp32 accumulators per thread), so 2 blocks of 256 threads fit "
      "per SM: 25 % occupancy on sm_80/sm_90, 33 % on sm_89, where an SM holds only 1536 threads.",
      "* **No kernel uses stack or spills.** The first versions of the register-blocked SGEMM and the flash kernel had 288- and "
      "640-byte stack frames: their per-thread arrays were in local memory because loops were not unrolled. `#pragma unroll` "
      "on the fixed-trip loops (`KC_UNROLL`) fixed that, as ptxas confirms.",
      "* **The register-blocked SGEMM** uses 128 registers per thread, so 2 blocks (25 % occupancy) fit on sm_80. That is "
      "the usual trade for an 8×8 micro-tile: latency is hidden by instruction-level parallelism (64 independent FMAs per k) "
      "rather than by warps. On sm_90 the non-split variant needs 130 registers and drops to 1 block per SM.",
      "* **The flash kernel** keeps q (64), o (64) and 32 scores per thread in registers: 245 registers and 12.5 % "
      "occupancy at D = 64. That is the cost of the thread-per-row design. FlashAttention-2 splits a row across a warp and uses "
      "tensor cores; see [limitations](limitations.md).",
      ""]
open("docs/analysis.md", "w").write("\n".join(L))
