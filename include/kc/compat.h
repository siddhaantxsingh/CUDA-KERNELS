// Compiles the kernels either as real CUDA (nvcc or clang -x cuda) or, for
// host-only builds, against the CPU SIMT emulator.
#pragma once

#if defined(__CUDACC__)
#define KC_CUDA 1
#define KC_G(x) (x)
#define KC_S(x) (x)
#define KC_LAUNCH(kernel, grid, block, ...) (kernel)<<<(grid), (block)>>>(__VA_ARGS__)
// fully unroll fixed-trip loops so per-thread arrays stay in registers
#define KC_UNROLL _Pragma("unroll")
#else
#define KC_UNROLL
#include "simt_emu.h"
#endif

namespace kc {
constexpr unsigned FULL_MASK = 0xffffffffu;

template <class T> __device__ __forceinline__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}
template <class T> __device__ __forceinline__ T warp_max(T v) {
    for (int o = 16; o > 0; o >>= 1) {
        T u = __shfl_xor_sync(FULL_MASK, v, o);
        v = u > v ? u : v;
    }
    return v;
}
inline unsigned ceil_div(unsigned a, unsigned b) { return (a + b - 1) / b; }
}  // namespace kc
