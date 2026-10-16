#!/bin/sh
# Compile all kernels as real CUDA device code and report ptxas resource
# usage. Needs clang (>= 17) and a CUDA toolkit root (headers, libdevice,
# ptxas): CUDA_ROOT=/usr/local/cuda, or one assembled from the PyPI wheels by
# tools/fetch_cuda.sh. Usage: tools/device_build.sh OUT_DIR [ARCH...]
set -eu
OUT=${1:-build/device}
shift || true
ARCHS=${*:-sm_80 sm_89 sm_90}
CUDA_ROOT=${CUDA_ROOT:-build/cuda-root}
mkdir -p "$OUT"
# clang 18's CUDA mode cannot parse GCC 14's <limits> (__float128); use GCC 13's
# headers when they are installed
GCC_DIR=${GCC_DIR:-/usr/lib/gcc/x86_64-linux-gnu/13}
HOSTFLAGS=""
if [ -d "$GCC_DIR" ]; then HOSTFLAGS="--gcc-install-dir=$GCC_DIR"; fi
for arch in $ARCHS; do
  clang++ -x cuda --cuda-path="$CUDA_ROOT" --cuda-gpu-arch="$arch" --cuda-device-only -S -O3 -std=c++17 \
    -Wno-unknown-cuda-version $HOSTFLAGS -Iinclude tools/all_kernels.cu -o "$OUT/kernels_$arch.ptx"
  "$CUDA_ROOT/bin/ptxas" -arch="$arch" -O3 -v "$OUT/kernels_$arch.ptx" -o "$OUT/kernels_$arch.cubin" 2> "$OUT/ptxas_$arch.txt"
  echo "$arch: $(grep -c 'Compiling entry' "$OUT/ptxas_$arch.txt") kernels compiled"
done
