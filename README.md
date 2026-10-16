<div align="center">

# ⚡ CUDA-KERNELS

### Hand-written GPU kernels — from naive to tensor cores — with a CPU SIMT emulator that proves they are correct

**Reduction · transpose · SGEMM · online softmax · Welford LayerNorm · flash attention · `mma.sync` tensor-core GEMM · backward passes**

<br/>

![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![PTX](https://img.shields.io/badge/inline_PTX-mma.sync_·_ldmatrix-6A1B9A?style=for-the-badge)
![Clang](https://img.shields.io/badge/clang_18_+_ptxas_12.9-262D3A?style=for-the-badge&logo=llvm&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2E7D32?style=for-the-badge)

![Targets](https://img.shields.io/badge/targets-sm__80_·_sm__89_·_sm__90-1565C0?style=flat-square)
![Checks](https://img.shields.io/badge/emulator_checks-170%2F170_passing-2E7D32?style=flat-square)
![Spills](https://img.shields.io/badge/23_kernels-0_spills_·_0_stack-2E7D32?style=flat-square)
![HMMA](https://img.shields.io/badge/SASS-HMMA.16816.F32_verified-76B900?style=flat-square)
![Attention](https://img.shields.io/badge/flash_attention-60×_less_traffic-E65100?style=flat-square)
![GPU timing](https://img.shields.io/badge/GPU_timing-pending_hardware_run-F9A825?style=flat-square)

</div>

---

## ⚡ TL;DR

This is a **kernel engineering lab**. Every kernel is written as an *optimisation progression*, so each step — coalescing, shared-memory tiling, bank-conflict padding, register blocking, vectorisation, warp shuffles, IO-aware fusion, tensor cores — has a measurable effect.

- 🧮 **A tensor-core GEMM in raw inline PTX.** fp16 in, fp32 accumulate, `ldmatrix.x4` / `ldmatrix.x4.trans` feeding `mma.sync.m16n8k16`. The SASS confirms **32 × `HMMA.16816.F32` + 12 × `LDSM` per warp per k-tile** on A100, RTX 40xx and H100.
- 🖥️ **A CPU SIMT emulator** that runs the *same source* unchanged: real OS threads per block, `std::barrier` for `__syncthreads`, warp shuffles, and a warp-level model of the `mma` / `ldmatrix` fragment layouts. A kernel that loads fragments in the wrong layout gives wrong numbers.
- 🔬 **A memory-access tracer** that computes what Nsight Compute would report — 32-byte sectors per request and shared-memory bank-conflict wavefronts — from a model of the hardware rules.
- 🧪 **170 checks** against double-precision references with derived error bounds, plus **mutation testing**: deliberately broken kernels (missing barrier, wrong fragment mapping, missing `.trans`, …) must fail the suite.
- 🛠️ **A GPU toolchain without a GPU**: clang 18 + NVIDIA `ptxas` and `libnvfatbin` assembled from PyPI wheels, plus a custom `fatbinary` shim, link a real CUDA benchmark binary.

> [!IMPORTANT]
> **No GPU timings are claimed.** This was built on a machine without an NVIDIA GPU. Correctness is verified on the emulator, the code is compiled to real SASS for three architectures, and memory behaviour is *modelled*. The benchmark harness is built and scripted for an RTX 4060 run ([`docs/OWNER_ACTIONS.md`](docs/OWNER_ACTIONS.md)); measured GFLOP/s will be added after that run.

---

## 🗺️ The optimisation progressions

```mermaid
flowchart TB
    subgraph RED["Σ Reduction"]
        direction LR
        R1["atomic per element"]:::naive --> R2["shared-memory tree"]:::mid --> R3["float4 + warp shuffles<br/>deterministic two-pass"]:::best
    end
    subgraph TR["⤡ Transpose"]
        direction LR
        T1["naive<br/>22% coalesced"]:::naive --> T2["shared tile<br/>16.5× bank conflicts"]:::mid --> T3["padded tile<br/>100% · conflict-free"]:::best
    end
    subgraph GE["▦ SGEMM"]
        direction LR
        G1["naive<br/>69 MB"]:::naive --> G2["32×32 smem tiles<br/>4.5 MB"]:::mid --> G3["128×128 register-blocked<br/>1.3 MB"]:::mid2 --> G4["split micro-tiles<br/>conflict-free · 100%"]:::best
    end
    subgraph AT["◎ Attention"]
        direction LR
        A1["unfused<br/>142 MB · 12.5%"]:::naive --> A2["flash (tiled, online softmax)<br/>2.4 MB · 100%"]:::best
    end
    subgraph TC["🧮 Tensor cores"]
        direction LR
        H1["fp32 SGEMM"]:::mid --> H2["fp16 mma.sync.m16n8k16<br/>+ ldmatrix · fp32 acc"]:::tc
    end

    RED ~~~ TR ~~~ GE ~~~ AT ~~~ TC

    classDef naive fill:#C62828,stroke:#B71C1C,color:#fff
    classDef mid fill:#E65100,stroke:#BF360C,color:#fff
    classDef mid2 fill:#F9A825,stroke:#F57F17,color:#000
    classDef best fill:#2E7D32,stroke:#1B5E20,color:#fff
    classDef tc fill:#76B900,stroke:#4E7A00,color:#000
    style RED fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
    style TR fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
    style GE fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
    style AT fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
    style TC fill:transparent,stroke:#90A4AE,stroke-dasharray:5 5
```

Plus **row kernels** — warp-per-row online softmax and Welford LayerNorm (numerically safe up to `FLT_MAX`, `-inf` masking, no cancellation at mean 10⁴) — and **backward passes** for softmax and LayerNorm (dx, dγ, dβ).

| File | Kernels | Key technique |
|---|---|---|
| `reduce.cuh` | atomic → shared tree → float4 + shuffles, two-pass | grid-stride loops, `__shfl_xor_sync`, deterministic second pass |
| `transpose.cuh` | naive → shared tile → padded tile | coalescing through shared memory, bank-conflict padding |
| `gemm.cuh` | naive → 32×32 tiles → 128×128×8 register-blocked (8×8 per thread), contiguous and *split* | transposed A tile, float4 shared loads, conflict-free split micro-tiles, float4 epilogue |
| `softmax.cuh` | online softmax, Welford LayerNorm (+ naive E[x²]−E[x]² for contrast) | single-pass statistics merged with shuffles |
| `hgemm.cuh` + `mma.cuh` | tensor-core GEMM, fp16 A/B, fp32 C | inline PTX, 16-byte padded rows for conflict-free `ldmatrix`, vector or zero-filled element loads |
| `backward.cuh` | softmax backward, LayerNorm backward | warp per row; deterministic dγ/dβ |
| `attention.cuh` | flash attention (optional causal mask) + unfused baseline | IO-aware tiling, Q/O staged through padded shared memory |

---

## 🧮 Tensor-core GEMM — how the work is split

```mermaid
flowchart TB
    G["🟩 C = α·A·B + β·C<br/>A, B: fp16 row-major · C: fp32"]:::top
    BT["📦 Thread block · 256 threads<br/>128 × 128 output tile · k-tile 32<br/>A, B tiles in shared memory (+16 B row padding)"]:::block
    W["🧵 8 warps · each owns a 64 × 32 sub-tile<br/>64 fp32 accumulators per thread"]:::warp
    L["📥 ldmatrix.x4 (A) · ldmatrix.x4.trans (B)<br/>12 LDSM per warp per k-tile · conflict-free"]:::ld
    M["⚙️ mma.sync.m16n8k16.f32.f16.f16.f32<br/>2 k-steps × 4 × 4 tiles = 32 HMMA per k-tile"]:::mma
    E["📤 Epilogue: α/β · fp32 stores"]:::ep

    G --> BT --> W --> L --> M --> E

    classDef top fill:#37474F,stroke:#263238,color:#fff
    classDef block fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef warp fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef ld fill:#00838F,stroke:#006064,color:#fff
    classDef mma fill:#76B900,stroke:#4E7A00,color:#000
    classDef ep fill:#E65100,stroke:#BF360C,color:#fff
```

**Proof that it really uses tensor cores** — `cuobjdump -sass` on the sm_80 cubin ([full excerpt](docs/evidence/2026-10-15/sass_hgemm_excerpt.txt)):

```text
LDSM.16.M88.4  R24, [R2] ;
LDSM.16.MT88.4 R20, [R86] ;          ← ldmatrix .trans for B
LDSM.16.MT88.4 R8,  [R87] ;
HMMA.16816.F32 R76, R24.reuse, R20, R76 ;
HMMA.16816.F32 R72, R24.reuse, R22, R72 ;
HMMA.16816.F32 R68, R24.reuse, R8,  R68 ;
HMMA.16816.F32 R64, R24,       R10, R64 ;
```

`tools/sass_check.py` fails the build if any tensor-core kernel contains no HMMA instruction.

| Arch | HMMA (form) | LDSM | Registers | Shared mem | Blocks / SM | Occupancy |
|---|---|---|---|---|---|---|
| sm_80 (A100) | 32 × `HMMA.16816.F32` | 12 | 123–124 | 18,944 B | 2 | 25 % |
| sm_89 (RTX 40xx, L4) | 32 × `HMMA.16816.F32` | 12 | 122–126 | 18,944 B | 2 | 33 % |
| sm_90 (H100) | 32 × `HMMA.16816.F32` | 12 | 123–126 | 18,944 B | 2 | 25 % |

---

## 🖥️ One source, three builds

```mermaid
flowchart TB
    SRC["📄 include/kc/*.cuh<br/>one kernel source · KC_G() / KC_S() / KC_LAUNCH"]:::src

    SRC --> CUDA["⚡ clang++ -x cuda / nvcc<br/>real CUDA · #60;#60;#60;grid, block#62;#62;#62;"]:::gpu
    SRC --> EMU["🖥️ g++ + simt_emu.h<br/>threads · barriers · shuffles · mma model"]:::emu
    SRC --> TRC["🔬 g++ + tracer<br/>every global / shared access recorded"]:::trc

    CUDA --> O1["PTX → SASS for sm_80 / 89 / 90<br/>registers · spills · HMMA count"]:::out
    EMU --> O2["170 correctness checks<br/>vs double-precision references"]:::out
    TRC --> O3["sectors / request · coalescing<br/>bank-conflict wavefronts"]:::out

    classDef src fill:#37474F,stroke:#263238,color:#fff
    classDef gpu fill:#76B900,stroke:#4E7A00,color:#000
    classDef emu fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef trc fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef out fill:#ECEFF1,stroke:#90A4AE,color:#000
```

### Inside the emulator

| GPU concept | Emulated as |
|---|---|
| Thread block | `blockDim` real OS threads from a persistent pool; blocks run one after another |
| `__syncthreads()` | `std::barrier` across the block — a missing barrier is a **real data race** the tests catch |
| `__shared__` | `static` storage, safe because blocks don't overlap |
| Warp shuffles | per-warp exchange buffer between two warp barriers |
| `ldmatrix` | each lane posts a row address, then picks its two 16-bit elements per matrix from the PTX ISA layout (transposed for `.trans`) |
| `mma.m16n8k16` | each lane posts its A/B fragments; the model rebuilds the 16×16 and 16×8 matrices from the documented layout and returns each lane's 4 fp32 outputs |
| Global memory cost | distinct 32-byte sectors per warp instruction |
| Shared memory cost | phases of 32 / 16 / 8 lanes for 4 / 8 / 16-byte accesses; wavefronts = worst bank's distinct words; broadcasts free |

The tracer itself is pinned by 20 tests against patterns with known answers: strides 1/2/32, misalignment, float4, broadcast, predication, and bank strides 1/2/32/33.

---

## 📊 What the evidence shows

All memory figures come from **modelled traces** (`make analyze`), not hardware counters. They show *why* one variant should be faster, not *how much*.

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#E65100"}}}}%%
xychart-beta
    title "SGEMM 256³ — global bytes requested (MB)"
    x-axis ["naive", "32×32 smem tiles", "register 8×8", "8×8 split"]
    y-axis "MB requested" 0 --> 75
    bar [69.47, 4.46, 1.31, 1.31]
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#6A1B9A"}}}}%%
xychart-beta
    title "Attention N=512, D=64 — global bytes requested (MB)"
    x-axis ["unfused (scores in global mem)", "flash (tiled, online softmax)"]
    y-axis "MB requested" 0 --> 150
    bar [141.69, 2.36]
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#C62828"}}}}%%
xychart-beta
    title "Transpose 1024² — shared-memory bank-conflict factor (1 = ideal)"
    x-axis ["naive (no smem)", "smem tile", "smem tile + 1 pad column"]
    y-axis "conflict factor" 0 --> 18
    bar [1.0, 16.5, 1.0]
```

| Optimisation step | Before → after |
|---|---|
| Transpose: shared tile → padded | bank-conflict factor **16.5 → 1.0**, 100 % coalesced (naive: 22 %) |
| SGEMM: naive → 32×32 tiles → register-blocked | bytes requested **69 MB → 4.5 MB → 1.3 MB** (256³) |
| SGEMM register-blocked: contiguous → split micro-tile | B-read conflicts 2-way → **none**; C stores 42 % → **100 %** coalesced |
| Attention: unfused → flash | **142 MB → 2.4 MB** (≈ 60× less); 12.5 % → 100 % coalesced |
| Tensor-core GEMM (256³) | worst error **0.009 ×** its bound; `ldmatrix` reads conflict-free |
| ptxas, all 23 kernels × 3 archs | **0 bytes stack, 0 spills** |

> 🧠 **Found and fixed through the tools:**
> - The first register-blocked SGEMM and flash kernels **spilled 288 B and 640 B per thread** to local memory. ptxas exposed it; `#pragma unroll` on the fixed-trip loops (`KC_UNROLL`) brought both to zero.
> - The tracer found an **8-way conflict on the A-tile stores** of the register-blocked SGEMM; padding the transposed A tile by 4 floats removed it.
> - The first flash kernel loaded one row per thread (one sector per lane); staging Q/O through padded shared memory made it **100 % coalesced**.

---

## 🧪 Correctness — 170 checks, derived error bounds, mutation testing

```mermaid
flowchart LR
    T1["🔬 Tracer model<br/>20 checks<br/>known access patterns"]:::a --> T2["🧮 Kernels<br/>150 checks<br/>vs double references"]:::b --> T3["🧬 Mutations<br/>broken kernels<br/>must fail"]:::c --> T4["⚙️ Device build<br/>23 kernels × 3 archs<br/>0 spills · HMMA check"]:::d

    classDef a fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef b fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef c fill:#C62828,stroke:#B71C1C,color:#fff
    classDef d fill:#76B900,stroke:#4E7A00,color:#000
```

| Kernel | Shapes tested | Tolerance |
|---|---|---|
| Reductions | n = 1 … 2²⁰, odd sizes | fp32 error bound × Σ\|x\|; two-pass is **bit-identical** across runs |
| Transpose | 1×1, 33×65, 100×7, 256×512, … | exact |
| SGEMM (4 variants) | 1×1×1 → 256³, odd shapes like 129×130×17, 5×257×3, α/β | (K + 2)·6·10⁻⁸·Σ\|aᵢₖbₖⱼ\| |
| Softmax | up to 64×4096, one 3·10³⁸, `-inf`, all-`-inf` row → NaN | 10⁻⁵ relative |
| LayerNorm | up to 4×4096, mean 10⁴ | 10⁻⁴ + input-rounding term |
| Flash attention D = 32, 64 | N = 1, 37, 130, 200 · causal and not | 2·10⁻⁵ absolute |
| fp16 conversion | **all 65,536 fp16 values** + 2 M random floats | exact, vs GCC `_Float16` |
| Tensor-core GEMM | 10 shapes incl. 1×1×1, 129×130×17, 130×264×520 | (K + 2)·2⁻²³·Σ\|aᵢₖbₖⱼ\| — covers tensor-core truncation (Fasi et al., 2021) |
| Softmax / LayerNorm backward | up to 16×4096, 300×64 | analytic references, themselves checked against **central finite differences** |

**Clever checks worth calling out**
- 🧪 The tensor-core GEMM is also compared against the *unrounded* fp32 inputs, and the error must be **> 30× larger** — proving the data really goes through fp16.
- 🧪 **NaNs are placed after A and B** in memory; they must never reach C, proving no element past the K tail is loaded.
- 🧪 The naive E[x²]−E[x]² LayerNorm must **break down** at mean 10⁴ — showing why the Welford version exists.

### Mutation log ([evidence](docs/evidence/2026-10-15/mutations.log))

| Deliberate bug | Result |
|---|---|
| B loaded with `ldmatrix` **without `.trans`** | ❌ caught — 11 / 13 checks fail |
| B fragments of neighbouring n-tiles swapped | ❌ caught — 9 / 13 |
| A `ldmatrix` lane-to-row mapping wrong | ❌ caught — 10 / 13 |
| Missing `__syncthreads()` before the mma loop | ❌ caught — 11 / 13 |
| Epilogue row offset for c2/c3 wrong | ❌ caught — 12 / 13 |
| Emulator: a1/a2 swapped in the mma layout model | ❌ caught — 1 / 1 |
| LayerNorm / softmax backward missing a term | ❌ caught — 6 / 22 and 5 / 10 |
| K-tail guard removed (vector path) | ⚠️ **initially missed** — zero fill hid it → added the NaN-padding check → ❌ caught |
| K-tail guard removed (element path) | ❌ caught — 1 / 15 |

---

## 🛠️ A CUDA toolchain without a GPU

```mermaid
flowchart LR
    W["📦 NVIDIA PyPI wheels<br/>headers · libdevice · ptxas<br/>cudart · libnvfatbin · cuobjdump"]:::w
    C["🔧 clang 18<br/>-x cuda → PTX"]:::c
    P["⚙️ ptxas 12.9<br/>→ cubin sm_80/89/90"]:::p
    S["🔍 cuobjdump -sass<br/>+ sass_check.py"]:::s
    F["🧩 fatbinary_shim.c<br/>(on libnvfatbin)"]:::f
    B["🚀 gpu_bench<br/>real CUDA executable"]:::b

    W --> C --> P --> S
    P --> F --> B

    classDef w fill:#37474F,stroke:#263238,color:#fff
    classDef c fill:#1565C0,stroke:#0D47A1,color:#fff
    classDef p fill:#76B900,stroke:#4E7A00,color:#000
    classDef s fill:#6A1B9A,stroke:#4A148C,color:#fff
    classDef f fill:#E65100,stroke:#BF360C,color:#fff
    classDef b fill:#2E7D32,stroke:#1B5E20,color:#fff
```

The wheels don't include the `fatbinary` program that clang needs to embed device code, so `tools/fatbinary_shim.c` implements it on top of NVIDIA's `libnvfatbin`. The result is a fully linked CUDA benchmark with an `.nv_fatbin` section. On a GPU-less machine it correctly stops with *"CUDA driver version is insufficient"*.

---

## 🚀 Quick start

```sh
make test                         # emulator: 20 tracer checks + 150 kernel checks (~2 min)
make analyze                      # traced memory behaviour per kernel (JSON)
tools/fetch_cuda.sh               # assemble a CUDA toolkit root from NVIDIA PyPI wheels
make ptx                          # sm_80/89/90 → registers, smem, spills, occupancy, HMMA counts
make gpu-bench CUDA_ARCH=sm_89    # link the real CUDA benchmark (needs a GPU to run)
python3 tools/report.py           # regenerate docs/analysis.md from results/
```

On a machine with an NVIDIA GPU (RTX 4060 under WSL2 is scripted):

```sh
tools/owner_rtx4060.sh            # tests + build + 3 benchmark runs + nvidia-smi dmon logging
```

The benchmark first runs on-GPU correctness checks — including an `mma_probe` that confirms the emulator's fragment-layout model on real hardware — then times every kernel.

---

## 📁 Repository layout

```text
CUDA-KERNELS/
├── include/kc/       # header-only kernels + compat.h + simt_emu.h (emulator) + half.h
├── tests/            # test_tracer.cpp (20) · test_kernels.cpp (150)
├── bench/            # gpu_bench.cu — real GPU harness with on-device checks
├── tools/            # analyze · report.py · resources.py · sass_check.py
│                     # fetch_cuda.sh · fatbinary_shim.c · device_build.sh · owner_rtx4060.sh
├── results/          # analysis.json · resources.json · sass.json
└── docs/             # design · analysis · testing · limitations · requirements · evidence/
```

---

## 🧭 Status and roadmap

| | Status |
|---|---|
| Functional correctness (all kernels) | ✅ verified on the emulator — 170 / 170 |
| Compiles to real SASS, sm_80 / 89 / 90, 0 spills | ✅ verified |
| Tensor cores used (`HMMA.16816.F32` in SASS) | ✅ verified |
| Fragment-layout model confirmed on hardware | ⏳ pending GPU run (`mma_probe`) |
| Measured GFLOP/s, GB/s | ⏳ pending GPU run |
| cp.async multi-stage pipeline, XOR swizzle, split-K | 🔜 next — removes the remaining 2-way A-tile store conflict |
| FlashAttention-2 style (warp per row, tensor cores), attention backward | 🔜 planned |
| bf16 / FP8, Hopper `wgmma` + TMA, cuBLAS comparison | 🔜 planned |

Honest notes: the emulator's metrics ignore caches and scheduling; the flash kernel is FlashAttention-1-style (245 registers, 12.5 % occupancy at D = 64); the toolchain is clang + ptxas rather than nvcc. Full detail in [`docs/limitations.md`](docs/limitations.md) and [`docs/design.md`](docs/design.md).

---

<div align="center">

**Every optimisation has a number. Every number has a test. Every test is proven to catch bugs.**

<sub>MIT © siddhaantxsingh</sub>

</div>
