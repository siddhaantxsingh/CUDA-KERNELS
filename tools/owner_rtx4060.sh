#!/usr/bin/env bash
# Owner-only run: correctness + timing on an RTX 4060 (sm_89) under WSL2.
# Needs: an NVIDIA Windows driver with WSL support (nvidia-smi works inside
# WSL), g++ >= 12, and either the CUDA toolkit for WSL (nvcc) or clang 18
# (`sudo apt-get install -y clang-18 g++-13`; the script then fetches ptxas
# and cudart from PyPI with tools/fetch_cuda.sh).
# Usage, from the cuda-kernels folder:  tools/owner_rtx4060.sh
# Output: results/gpu/<date>-rtx4060/ (see docs/OWNER_ACTIONS.md)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/gpu/$(date +%Y-%m-%d)-rtx4060
mkdir -p "$OUT" build

{
  echo "date: $(date -Is)"; uname -a
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
  nvidia-smi
} > "$OUT/system.txt" 2>&1
grep -q "8.9" "$OUT/system.txt" || echo "warning: compute capability 8.9 not found in nvidia-smi output; check $OUT/system.txt"

# 1. CPU emulator tests (same as CI)
make test 2>&1 | tee "$OUT/make_test.log"

# 2. build the GPU benchmark for sm_89
if command -v nvcc >/dev/null; then
  nvcc --version > "$OUT/compiler.txt"
  nvcc -arch=sm_89 -O3 -std=c++17 -Iinclude bench/gpu_bench.cu -o build/gpu_bench_sm89 2>&1 | tee "$OUT/build.log"
else
  [ -x build/cuda-root/bin/ptxas ] || tools/fetch_cuda.sh
  clang++ --version > "$OUT/compiler.txt"
  make gpu-bench CUDA_ARCH=sm_89 2>&1 | tee "$OUT/build.log"
  cp build/gpu_bench build/gpu_bench_sm89
fi
# WSL2: libcuda.so lives in /usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

# 3. three benchmark runs with nvidia-smi dmon (power, temperature, clocks, utilisation) sampled every second
nvidia-smi dmon -s pucm -d 1 > "$OUT/dmon.txt" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null || true' EXIT
status=0
for i in 1 2 3; do
  uptime > "$OUT/run$i.txt"
  ./build/gpu_bench_sm89 "$OUT/run$i.json" >> "$OUT/run$i.txt" 2>&1 || status=$?
done
kill $DMON 2>/dev/null || true
grep -h "^check" "$OUT"/run*.txt | sort | uniq -c
echo "benchmark exit status of the last failing run: $status (0 = all correctness checks passed)"
echo "saved to $OUT; commit that folder"
exit $status
