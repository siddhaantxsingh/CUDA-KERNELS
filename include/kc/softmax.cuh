// Row softmax and LayerNorm over a rows x cols row-major matrix, one warp
// per row, single pass over the input for the statistics.
#pragma once
#include <cmath>
#include "compat.h"

namespace kc {

// Online softmax: each lane keeps a running (max, sum of exp(x - max)) and
// lanes merge pairs with shuffles. Never overflows (x up to FLT_MAX), -inf
// entries are masked out, an all -inf row gives NaN (as PyTorch does).
__global__ void softmax_warp_kernel(const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;  // whole warps exit together: blockDim is a multiple of 32
    const float* xr = x + size_t(row) * cols;
    float m = -INFINITY, s = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float v = KC_G(xr[c]);
        if (v == -INFINITY) continue;
        if (v > m) {
            s = s * expf(m - v) + 1.0f;
            m = v;
        } else {
            s += expf(v - m);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float mo = __shfl_xor_sync(FULL_MASK, m, o);
        float so = __shfl_xor_sync(FULL_MASK, s, o);
        float mn = fmaxf(m, mo);
        s = (m == -INFINITY ? 0.0f : s * expf(m - mn)) + (mo == -INFINITY ? 0.0f : so * expf(mo - mn));
        m = mn;
    }
    float inv = 1.0f / s;
    float* yr = y + size_t(row) * cols;
    for (int c = lane; c < cols; c += 32) KC_G(yr[c]) = expf(KC_G(xr[c]) - m) * inv;
}

// Mean and (biased) variance of one row, computed by a whole warp: Welford
// per lane, then pairwise merges with shuffles (Chan et al.).
__device__ __forceinline__ void welford_row(const float* xr, int cols, int lane, float* mean, float* var) {
    float n = 0.0f, mu = 0.0f, m2 = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float v = KC_G(xr[c]);
        n += 1.0f;
        float d = v - mu;
        mu += d / n;
        m2 += d * (v - mu);
    }
    for (int o = 16; o > 0; o >>= 1) {
        float nb = __shfl_xor_sync(FULL_MASK, n, o);
        float mb = __shfl_xor_sync(FULL_MASK, mu, o);
        float m2b = __shfl_xor_sync(FULL_MASK, m2, o);
        float nn = n + nb;
        if (nn > 0.0f) {
            float d = mb - mu;
            mu += d * (nb / nn);
            m2 += m2b + d * d * (n * nb / nn);
            n = nn;
        }
    }
    *mean = mu;
    *var = m2 / cols;
}

// LayerNorm with Welford statistics merged across lanes (Chan et al.).
// NAIVE = true uses E[x^2] - E[x]^2 in fp32 instead, which loses all
// precision when |mean| >> std (kept as a demonstration).
template <bool NAIVE>
__global__ void layernorm_warp_kernel(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
                                      float* __restrict__ y, int rows, int cols, float eps) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* xr = x + size_t(row) * cols;
    float mean, var;
    if (NAIVE) {
        float s = 0.0f, ss = 0.0f;
        for (int c = lane; c < cols; c += 32) {
            float v = KC_G(xr[c]);
            s += v;
            ss += v * v;
        }
        s = warp_sum(s);
        ss = warp_sum(ss);
        mean = s / cols;
        var = fmaxf(ss / cols - mean * mean, 0.0f);
    } else {
        welford_row(xr, cols, lane, &mean, &var);
    }
    float rstd = rsqrtf(var + eps);
    float* yr = y + size_t(row) * cols;
    for (int c = lane; c < cols; c += 32) KC_G(yr[c]) = (KC_G(xr[c]) - mean) * rstd * KC_G(gamma[c]) + KC_G(beta[c]);
}

inline void softmax(const float* x, float* y, int rows, int cols) {
    auto k = softmax_warp_kernel;
    KC_LAUNCH(k, dim3(ceil_div(rows, 4)), dim3(128), x, y, rows, cols);
}
template <bool NAIVE = false>
inline void layernorm(const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps = 1e-5f) {
    auto k = layernorm_warp_kernel<NAIVE>;
    KC_LAUNCH(k, dim3(ceil_div(rows, 4)), dim3(128), x, gamma, beta, y, rows, cols, eps);
}

}  // namespace kc
