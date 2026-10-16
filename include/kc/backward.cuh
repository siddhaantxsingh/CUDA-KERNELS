// Backward passes for the row kernels in softmax.cuh (fp32, row-major).
#pragma once
#include <cmath>
#include "compat.h"
#include "softmax.cuh"

namespace kc {

// Softmax backward, one warp per row: given y = softmax(x) and dy = dL/dy,
//   dx = y * (dy - sum_j dy_j * y_j).
__global__ void softmax_bwd_warp_kernel(const float* __restrict__ y, const float* __restrict__ dy, float* __restrict__ dx, int rows,
                                        int cols) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;  // whole warps exit together
    const float* yr = y + size_t(row) * cols;
    const float* dyr = dy + size_t(row) * cols;
    float dot = 0.0f;
    for (int c = lane; c < cols; c += 32) dot += KC_G(yr[c]) * KC_G(dyr[c]);
    dot = warp_sum(dot);
    float* dxr = dx + size_t(row) * cols;
    for (int c = lane; c < cols; c += 32) KC_G(dxr[c]) = KC_G(yr[c]) * (KC_G(dyr[c]) - dot);
}

// LayerNorm backward, part 1 (one warp per row). With xh = (x - mean) * rstd
// and g = dy * gamma:
//   dx = rstd * (g - mean(g) - xh * mean(g * xh)).
// The row statistics are recomputed with the forward pass's Welford code
// (the forward kernel does not save them) and stored for part 2.
__global__ void layernorm_bwd_rows_kernel(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ dy,
                                          float* __restrict__ dx, float* __restrict__ mean_out, float* __restrict__ rstd_out, int rows,
                                          int cols, float eps) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* xr = x + size_t(row) * cols;
    const float* dyr = dy + size_t(row) * cols;
    float mean, var;
    welford_row(xr, cols, lane, &mean, &var);
    float rstd = rsqrtf(var + eps);
    float sg = 0.0f, sgx = 0.0f;
    for (int c = lane; c < cols; c += 32) {
        float g = KC_G(dyr[c]) * KC_G(gamma[c]);
        sg += g;
        sgx += g * (KC_G(xr[c]) - mean) * rstd;
    }
    sg = warp_sum(sg) / cols;
    sgx = warp_sum(sgx) / cols;
    float* dxr = dx + size_t(row) * cols;
    for (int c = lane; c < cols; c += 32) {
        float xh = (KC_G(xr[c]) - mean) * rstd;
        KC_G(dxr[c]) = rstd * (KC_G(dyr[c]) * KC_G(gamma[c]) - sg - xh * sgx);
    }
    if (lane == 0) {
        KC_G(mean_out[row]) = mean;
        KC_G(rstd_out[row]) = rstd;
    }
}

// LayerNorm backward, part 2: dgamma[c] = sum_r dy * xh, dbeta[c] = sum_r dy.
// One thread per column, rows in order: coalesced across the warp and
// deterministic (no atomics). Serial in the row count; a production kernel
// would split rows over blocks and add a second reduction pass.
__global__ void layernorm_bwd_cols_kernel(const float* __restrict__ x, const float* __restrict__ dy, const float* __restrict__ mean,
                                          const float* __restrict__ rstd, float* __restrict__ dgamma, float* __restrict__ dbeta, int rows,
                                          int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float sg = 0.0f, sb = 0.0f;
    for (int r = 0; r < rows; r++) {
        float d = KC_G(dy[size_t(r) * cols + c]);
        sg += d * (KC_G(x[size_t(r) * cols + c]) - KC_G(mean[r])) * KC_G(rstd[r]);
        sb += d;
    }
    KC_G(dgamma[c]) = sg;
    KC_G(dbeta[c]) = sb;
}

inline void softmax_backward(const float* y, const float* dy, float* dx, int rows, int cols) {
    auto k = softmax_bwd_warp_kernel;
    KC_LAUNCH(k, dim3(ceil_div(rows, 4)), dim3(128), y, dy, dx, rows, cols);
}
// mean and rstd: rows floats each (workspace, also useful to the caller)
inline void layernorm_backward(const float* x, const float* gamma, const float* dy, float* dx, float* dgamma, float* dbeta, float* mean,
                               float* rstd, int rows, int cols, float eps = 1e-5f) {
    auto k1 = layernorm_bwd_rows_kernel;
    KC_LAUNCH(k1, dim3(ceil_div(rows, 4)), dim3(128), x, gamma, dy, dx, mean, rstd, rows, cols, eps);
    auto k2 = layernorm_bwd_cols_kernel;
    KC_LAUNCH(k2, dim3(ceil_div(cols, 256)), dim3(256), x, dy, mean, rstd, dgamma, dbeta, rows, cols);
}

}  // namespace kc
