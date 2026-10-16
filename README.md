# CUDA kernels (project 8)

This project has fp32 CUDA kernels written as optimisation progressions:

* reduction, transpose and SGEMM: from naive to register-blocked with conflict-free shared memory;
* online softmax and Welford LayerNorm;
* IO-aware ("flash") attention;
* a **tensor-core GEMM**: fp16 inputs, fp32 accumulation, `mma.sync.m16n8k16` and `ldmatrix` in inline PTX;
* **backward passes** for softmax and LayerNorm (dx, dγ, dβ).

It also has a **CPU SIMT emulator** that runs the same source unchanged for testing and traces every memory access. The emulator includes a warp-level model of the `mma`/`ldmatrix` fragment layouts. A toolchain compiles the kernels for A100 (sm_80), RTX 40xx (sm_89) and H100 (sm_90) without a GPU.

**Status:**

* **Functionally verified** on the emulator.
* **Compiles as real CUDA** for sm_80, sm_89 and sm_90, with no spills. The tensor-core GEMM contains `HMMA.16816.F32` in the SASS for all three.
* **GPU performance is BLOCKED:** there is no GPU here, so no timings are claimed. The owner's RTX 4060 run is scripted in [OWNER_ACTIONS](docs/OWNER_ACTIONS.md). See [limitations](docs/limitations.md).

## What the evidence shows

From [docs/analysis.md](docs/analysis.md). The memory figures come from modelled traces, not measurements.

| Kernel | Before → after |
|---|---|
| transpose: shared tile → padded | bank-conflict factor 16.5 → **1.0**, 100 % coalesced (naive: 22 %) |
| SGEMM: naive → 32×32 tiles → register-blocked | global bytes requested 69 MB → 4.5 MB → **1.3 MB** (256³) |
| SGEMM register-blocked: contiguous → split micro-tile | B-read conflicts 2-way → **none**; C stores 42 % → **100 %** coalesced |
| attention: unfused → flash | 142 MB → **2.4 MB** requested (N = 512, D = 64); 12.5 % → 100 % coalesced |
| tensor-core GEMM vs fp16-rounded double reference (emulator, 256³) | worst error 0.009 × the bound (K+2)·2⁻²³·Σ\|ab\|; `ldmatrix` reads conflict-free (modelled); 32 HMMA + 12 LDSM per k-tile per warp in the SASS; 122–126 registers, 2 blocks/SM |
| ptxas, all 23 kernels, sm_80 + sm_89 + sm_90 | **0 bytes stack, 0 spills** (the first SGEMM and flash versions spilled 288 B and 640 B per thread to local memory until their loops were unrolled) |

## Usage

```sh
make test             # emulator: 20 tracer checks + 150 kernel checks vs double-precision references
make analyze          # traced memory behaviour per kernel (JSON)
tools/fetch_cuda.sh   # CUDA toolkit root from NVIDIA PyPI wheels (ptxas, headers, libdevice, cudart, libnvfatbin, cuobjdump)
make ptx              # clang + ptxas for sm_80/sm_89/sm_90 -> registers, smem, spills, occupancy, HMMA counts
make gpu-bench CUDA_ARCH=sm_89   # links a real CUDA benchmark (needs a GPU to run)
tools/owner_rtx4060.sh           # owner only: tests + 3 benchmark runs + nvidia-smi dmon on an RTX 4060 (WSL2)
python3 tools/report.py   # regenerate docs/analysis.md from results/
```

## Documentation

* [Design](docs/design.md): the kernels, the emulator and its access model, and the device toolchain.
* [Testing](docs/testing.md).
* [Analysis](docs/analysis.md).
* [Limitations](docs/limitations.md).
* [Requirements matrix](docs/requirements.md).
* [Owner actions](docs/OWNER_ACTIONS.md): the GPU run only the owner can do.
* Evidence logs: [docs/evidence/2026-10-15/](docs/evidence/2026-10-15/).

## Scorecard (2026-10-15)

| Area | Score | Why not higher |
|---|---|---|
| Fresher-portfolio strength | 7/10 | Tensor cores, an emulator that models fragment layouts, and SASS checks are uncommon, but there is no measured GPU speed. |
| Engineering maturity | 7/10 | 150 emulator checks, mutation runs and CI, but nothing has run on real hardware and the fragment model is only checked against documentation. |
| Deployment readiness | 3/10 | This is a kernel library with a benchmark harness, not a packaged library. It has never executed on a GPU. |
| Completion of intended scope | 6/10 | GPU timing is BLOCKED, there is no cuBLAS comparison, no bf16/FP8, no cp.async pipeline and no attention backward. |
