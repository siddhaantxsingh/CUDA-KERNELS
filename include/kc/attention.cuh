// Scaled dot-product attention forward, fp32: O = softmax(Q K^T * scale) V
// for Q, K, V, O of shape [BH, N, D] (BH = batch x heads), optional causal
// mask.
#pragma once
#include <cmath>
#include "compat.h"

namespace kc {

// Tiled, IO-aware ("flash") attention, one thread per query row.
// K and V are streamed through shared memory in tiles of BC keys; the
// N x N score matrix is never written to global memory. Per tile the thread
// computes BC scores, then rescales its running output once (online
// softmax). This is the FlashAttention-1 algorithm without tensor cores:
// a correctness- and traffic-faithful baseline, not a fast kernel.
template <int D, int BR, int BC>
__global__ void __launch_bounds__(BR)
    flash_attn_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O, int N,
                      float scale, int causal) {
    __shared__ float Ks[BC][D];
    __shared__ float Vs[BC][D];
    // Q and O pass through shared memory so global accesses are coalesced
    // (thread-per-row access would touch one sector per lane); the +1 pad
    // makes the per-thread row reads conflict-free.
    __shared__ float QOs[BR][D + 1];
    const int bh = blockIdx.y;
    const int row0 = blockIdx.x * BR;
    const int row = row0 + threadIdx.x;
    const size_t base = size_t(bh) * N * D;
    for (int e = threadIdx.x; e < BR * D; e += BR) {
        int r = e / D, c = e % D;
        KC_S(QOs[r][c]) = row0 + r < N ? KC_G(Q[base + size_t(row0 + r) * D + c]) : 0.0f;
    }
    __syncthreads();
    float q[D], o[D];
    KC_UNROLL
    for (int d = 0; d < D; d++) {
        q[d] = KC_S(QOs[threadIdx.x][d]) * scale;
        o[d] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    // with a causal mask this block never needs keys past its last row
    const int kv_end = causal ? min(N, int(blockIdx.x + 1) * BR) : N;
    for (int j0 = 0; j0 < kv_end; j0 += BC) {
        for (int e = threadIdx.x; e < BC * D; e += BR) {
            int r = e / D, c = e % D;
            bool ok = j0 + r < N;
            KC_S(Ks[r][c]) = ok ? KC_G(K[base + size_t(j0 + r) * D + c]) : 0.0f;
            KC_S(Vs[r][c]) = ok ? KC_G(V[base + size_t(j0 + r) * D + c]) : 0.0f;
        }
        __syncthreads();
        if (row < N) {
            float s[BC];
            float mt = -INFINITY;
            KC_UNROLL
            for (int j = 0; j < BC; j++) {
                int key = j0 + j;
                if (key >= N || (causal && key > row)) {
                    s[j] = -INFINITY;
                    continue;
                }
                float acc = 0.0f;
                KC_UNROLL
                for (int d = 0; d < D; d++) acc += q[d] * KC_S(Ks[j][d]);
                s[j] = acc;
                mt = fmaxf(mt, acc);
            }
            if (mt != -INFINITY) {
                float mn = fmaxf(m, mt);
                float corr = expf(m - mn);  // 0 on the first tile (m = -inf)
                l *= corr;
                KC_UNROLL
                for (int d = 0; d < D; d++) o[d] *= corr;
                KC_UNROLL
                for (int j = 0; j < BC; j++) {
                    if (s[j] == -INFINITY) continue;
                    float p = expf(s[j] - mn);
                    l += p;
                    KC_UNROLL
                    for (int d = 0; d < D; d++) o[d] += p * KC_S(Vs[j][d]);
                }
                m = mn;
            }
        }
        __syncthreads();
    }
    // (the last __syncthreads of the loop also ends all reads of QOs as Q)
    float inv = row < N ? 1.0f / l : 0.0f;
    KC_UNROLL
    for (int d = 0; d < D; d++) KC_S(QOs[threadIdx.x][d]) = o[d] * inv;
    __syncthreads();
    for (int e = threadIdx.x; e < BR * D; e += BR) {
        int r = e / D, c = e % D;
        if (row0 + r < N) KC_G(O[base + size_t(row0 + r) * D + c]) = KC_S(QOs[r][c]);
    }
}

// Unfused reference kernel: materialises the scores in a [BH, N, N] scratch
// buffer (written once, read twice), the way a sequence of separate
// GEMM / softmax / GEMM kernels does. Used to compare memory traffic.
template <int D>
__global__ void attn_naive_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
                                  float* __restrict__ S, int N, float scale, int causal) {
    const int bh = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const size_t base = size_t(bh) * N * D;
    float* srow = S + (size_t(bh) * N + row) * N;
    const int end = causal ? row + 1 : N;
    float m = -INFINITY;
    for (int j = 0; j < end; j++) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) acc += KC_G(Q[base + size_t(row) * D + d]) * KC_G(K[base + size_t(j) * D + d]);
        acc *= scale;
        KC_G(srow[j]) = acc;
        m = fmaxf(m, acc);
    }
    float l = 0.0f;
    for (int j = 0; j < end; j++) {
        float p = expf(KC_G(srow[j]) - m);
        KC_G(srow[j]) = p;
        l += p;
    }
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = 0; j < end; j++) acc += KC_G(srow[j]) * KC_G(V[base + size_t(j) * D + d]);
        KC_G(O[base + size_t(row) * D + d]) = acc / l;
    }
}

template <int D>
inline void flash_attention(const float* Q, const float* K, const float* V, float* O, int BH, int N, bool causal) {
    constexpr int BR = 64, BC = 32;
    auto k = flash_attn_kernel<D, BR, BC>;
    KC_LAUNCH(k, dim3(ceil_div(N, BR), BH), dim3(BR), Q, K, V, O, N, 1.0f / std::sqrt(float(D)), int(causal));
}

template <int D>
inline void attention_naive(const float* Q, const float* K, const float* V, float* O, float* S, int BH, int N, bool causal) {
    auto k = attn_naive_kernel<D>;
    KC_LAUNCH(k, dim3(ceil_div(N, 64), BH), dim3(64), Q, K, V, O, S, N, 1.0f / std::sqrt(float(D)), int(causal));
}

}  // namespace kc
