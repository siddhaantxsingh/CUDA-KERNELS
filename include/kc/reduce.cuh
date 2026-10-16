// Sum reduction of n floats, in three steps of optimisation.
#pragma once
#include "compat.h"

namespace kc {

// 1. One atomic per element: correct, serialised on one address.
__global__ void reduce_atomic_kernel(const float* __restrict__ x, int n, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, KC_G(x[i]));
}

// 2. Shared-memory tree per block (sequential addressing: no divergence
//    inside warps until the last steps, no bank conflicts); one atomic per block.
template <int BS>
__global__ void reduce_smem_kernel(const float* __restrict__ x, int n, float* out) {
    __shared__ float s[BS];
    int t = threadIdx.x;
    int i = blockIdx.x * BS + t;
    KC_S(s[t]) = i < n ? KC_G(x[i]) : 0.0f;
    __syncthreads();
    for (int stride = BS / 2; stride > 0; stride >>= 1) {
        if (t < stride) KC_S(s[t]) += KC_S(s[t + stride]);
        __syncthreads();
    }
    if (t == 0) atomicAdd(out, KC_S(s[0]));
}

// 3. Grid-stride loop with float4 loads into registers, warp shuffles, one
//    shared slot per warp, and a deterministic second pass over the block
//    partials (same input -> bit-identical result).
template <int BS>
__global__ void reduce_partials_kernel(const float* __restrict__ x, int n, float* __restrict__ partials) {
    static_assert(BS % 32 == 0 && BS <= 1024, "block size");
    __shared__ float warp_sums[BS / 32];
    float acc = 0.0f;
    int tid = blockIdx.x * BS + threadIdx.x;
    int stride = gridDim.x * BS;
    int n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = tid; i < n4; i += stride) {
        float4 v = KC_G(x4[i]);
        acc += (v.x + v.y) + (v.z + v.w);
    }
    for (int i = 4 * n4 + tid; i < n; i += stride) acc += KC_G(x[i]);
    acc = warp_sum(acc);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) KC_S(warp_sums[warp]) = acc;
    __syncthreads();
    if (warp == 0) {
        float v = lane < BS / 32 ? KC_S(warp_sums[lane]) : 0.0f;
        v = warp_sum(v);
        if (lane == 0) KC_G(partials[blockIdx.x]) = v;
    }
}

template <int BS>
__global__ void reduce_final_kernel(const float* __restrict__ partials, int n, float* out) {
    __shared__ float warp_sums[BS / 32];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += BS) acc += KC_G(partials[i]);
    acc = warp_sum(acc);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) KC_S(warp_sums[warp]) = acc;
    __syncthreads();
    if (warp == 0) {
        float v = lane < BS / 32 ? KC_S(warp_sums[lane]) : 0.0f;
        v = warp_sum(v);
        if (lane == 0) KC_G(*out) = v;
    }
}

// ---------------------------------------------------------------- launchers
// `out` must be zeroed for the atomic variants. `x` must be 16-byte aligned
// for the vectorised variant. `partials` needs `blocks` floats.

inline void reduce_atomic(const float* x, int n, float* out) {
    auto k = reduce_atomic_kernel;
    KC_LAUNCH(k, dim3(ceil_div(n, 256)), dim3(256), x, n, out);
}

inline void reduce_smem(const float* x, int n, float* out) {
    auto k = reduce_smem_kernel<256>;
    KC_LAUNCH(k, dim3(ceil_div(n, 256)), dim3(256), x, n, out);
}

constexpr int REDUCE_BLOCKS = 256;
inline void reduce_two_pass(const float* x, int n, float* partials, float* out, int blocks = REDUCE_BLOCKS) {
    auto k1 = reduce_partials_kernel<256>;
    auto k2 = reduce_final_kernel<256>;
    KC_LAUNCH(k1, dim3(blocks), dim3(256), x, n, partials);
    KC_LAUNCH(k2, dim3(1), dim3(256), partials, blocks, out);
}

}  // namespace kc
