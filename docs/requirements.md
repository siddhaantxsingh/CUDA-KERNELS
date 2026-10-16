# Requirements matrix — Project 8 (CUDA kernels)

Evidence logs from 2026-10-15 are in [evidence/2026-10-15/](evidence/2026-10-15/).

| Requirement | Status | Evidence |
|---|---|---|
| Reduction, transpose, SGEMM, softmax, LayerNorm, attention kernels | VERIFIED (functional, on emulator) | `tests/test_kernels.cpp`: the original 100 checks vs double references still pass (`evidence/2026-10-15/make_test.log`) |
| Optimisation progressions (coalescing, shared memory, bank conflicts, register blocking, vectorisation, warp shuffles, online softmax, IO-aware attention) | VERIFIED (modelled) | `docs/analysis.md`: traced sectors / conflicts per variant |
| Tensor-core GEMM: fp16 in, fp32 accumulate, `mma.sync.m16n8k16` + `ldmatrix` (inline PTX) | VERIFIED on the emulator; NOT run on a GPU | `include/kc/hgemm.cuh`; 15 checks: 10 shapes incl. 1×1×1, 129×130×17, 5×257×3, 130×264×520, α/β, element-load path, K-tail NaN padding; mutation log `evidence/2026-10-15/mutations.log` |
| Emulator model of the mma / ldmatrix fragment layouts (PTX ISA m16n8k16 .f16) | VERIFIED against the PTX ISA tables (by construction + exact-integer probe); hardware confirmation BLOCKED | `include/kc/simt_emu.h`; `mma_probe` check; the GPU benchmark repeats the probe on hardware |
| Tensor-core kernels compile to HMMA for sm_80, sm_89, sm_90 | VERIFIED | `make ptx` → `results/sass.json`: 32 × `HMMA.16816.F32` and 12 `LDSM` per kernel per arch; `evidence/2026-10-15/sass_hgemm_excerpt.txt`; 122–126 registers, 0 stack, 0 spills (`evidence/2026-10-15/device_build.log`) |
| fp16 conversion helpers | VERIFIED | all 65 536 fp16 values round-trip; float→fp16 matches GCC `_Float16` on 2 M random floats + edge cases |
| Backward passes: softmax and LayerNorm (dx, dγ, dβ) | VERIFIED (functional, on emulator) | `include/kc/backward.cuh`; 33 checks vs analytic double references, and those references vs central finite differences |
| Kernels compile as real CUDA for current GPUs | VERIFIED | clang 18 + ptxas 12.9, sm_80, sm_89 and sm_90, 23 kernels, no spills (`make ptx`, CI) |
| sm_89 (RTX 40xx) build target | VERIFIED (compiles and links) | `make ptx` and `make gpu-bench CUDA_ARCH=sm_89` (`evidence/2026-10-15/gpu_bench_build_sm89.log`) |
| Resource usage and occupancy | VERIFIED (ptxas + calculator) | `results/resources.json`, `docs/analysis.md` |
| GPU benchmark harness (now with tensor-core GEMM, backward kernels and on-GPU correctness checks) | IMPLEMENTED BUT UNVERIFIED | `bench/gpu_bench.cu` links for sm_80 and sm_89; on this machine it exits with "CUDA driver version is insufficient" (`evidence/2026-10-15/gpu_bench_run_no_gpu.log`) |
| Owner run script for RTX 4060 / WSL2 | IMPLEMENTED BUT UNVERIFIED | `tools/owner_rtx4060.sh`, `docs/OWNER_ACTIONS.md`; only its syntax is checked (`bash -n`) |
| Measured GPU performance | BLOCKED | no GPU here; the owner runs `tools/owner_rtx4060.sh` |
| Comparison with cuBLAS | NOT IMPLEMENTED | the harness links only cudart, not cuBLAS |
| bf16 / FP8 tensor-core kernels, `wgmma` (Hopper) | NOT IMPLEMENTED | — |
| Attention backward | NOT IMPLEMENTED | — |
| CI (emulator tests, sm_80/89/90 build, HMMA check, sm_80 + sm_89 bench link) | VERIFIED | run 38821227301 green on commit bd5b12a; `.github/workflows/cuda-kernels.yml` |
