# Owner actions

Only the owner can do these steps: they need an NVIDIA GPU. Until they are done, GPU timing is **BLOCKED**.

## RTX 4060 laptop (sm_89) under WSL2

### 1. Prerequisites (once)

In Windows, install a recent NVIDIA Game Ready or Studio driver. It includes WSL support. Do **not** install a Linux driver inside WSL.

In the WSL2 Ubuntu shell:

```sh
nvidia-smi                      # must list "NVIDIA GeForce RTX 4060 Laptop GPU"
sudo apt-get update && sudo apt-get install -y build-essential g++-13 clang-18 python3-pip
sudo ln -sf /usr/bin/clang++-18 /usr/local/bin/clang++   # only if `clang++ --version` is not 18
```

You have two options for the compiler:

* the CUDA toolkit for WSL-Ubuntu from NVIDIA, which provides `nvcc`, then the script uses `nvcc -arch=sm_89`;
* nothing extra, then the script uses clang 18 with the PyPI toolkit root from `tools/fetch_cuda.sh` (the same flow as CI).

### 2. Run

```sh
cd PROJECTS/cuda-kernels
tools/owner_rtx4060.sh
```

The script does four things:

1. It records the GPU, driver and clocks.
2. It runs `make test` (the emulator, about 2 minutes).
3. It builds `bench/gpu_bench.cu` for sm_89.
4. It runs the benchmark three times while `nvidia-smi dmon -s pucm -d 1` records power, temperature, clocks and utilisation.

Plug the laptop in and close other GPU programs first.

### 3. Expected output

`make_test.log` ends with `150 checks, 0 failures`.

Each `runN.txt` has lines in this format. The numbers here are placeholders, not results:

```
device: NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, 24 SMs)
check      mma.m16n8k16 fragment layout (probe)             PASS (0)
check      hgemm 129x130x17 (error / bound)                 PASS (0.0x)
...
hgemm      mma.m16n8k16 + ldmatrix (fp16)   4096^3            x.xxxx ms   xxxxx.x GFLOP/s   xxx.x GB/s err 0.0x
sgemm      register 8x8, split              4096^3            ...
rows       layernorm backward (dx+dgamma+dbeta) 16384x4096    ...
0 correctness check failures
```

The last line of the script counts PASS/FAIL per check across the three runs.

If the build fails, or a `check` line says FAIL, save the folder anyway; it is the evidence.

### 4. What to save

Commit `results/gpu/<date>-rtx4060/`. It holds `system.txt`, `compiler.txt`, `build.log`, `make_test.log`, `run1..3.txt`, `run1..3.json` and `dmon.txt`.

After that, the GPU timing rows in [requirements.md](requirements.md) can change from BLOCKED to measured. Quote the median of the three runs, the device, the driver and the power/clock data from `dmon.txt`. A laptop GPU's clocks depend on power and temperature limits.

### What a pass would and would not show

* A PASS on `mma_probe` and the odd-shape `hgemm` checks confirms the emulator's fragment-layout model on real hardware.
* The GFLOP/s figures are for this simple kernel, without cp.async pipelining. They are not a cuBLAS comparison: the harness does not link cuBLAS.
