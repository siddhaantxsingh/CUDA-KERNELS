// SGEMM, row-major: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C.
#pragma once
#include "compat.h"

namespace kc {

// 1. One thread per output element; every operand comes from global memory.
__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B,
                                   float beta, float* __restrict__ C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += KC_G(A[size_t(row) * K + k]) * KC_G(B[size_t(k) * N + col]);
    float* c = &C[size_t(row) * N + col];
    KC_G(*c) = alpha * acc + (beta != 0.0f ? beta * KC_G(*c) : 0.0f);
}

// 2. 32x32 shared-memory tiles: each element of A and B is read from global
//    memory N/32 and M/32 times instead of N and M times.
template <int T>
__global__ void sgemm_smem_kernel(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta,
                                  float* __restrict__ C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        KC_S(As[ty][tx]) = (row < M && k0 + tx < K) ? KC_G(A[size_t(row) * K + k0 + tx]) : 0.0f;
        KC_S(Bs[ty][tx]) = (k0 + ty < K && col < N) ? KC_G(B[size_t(k0 + ty) * N + col]) : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; k++) acc += KC_S(As[ty][k]) * KC_S(Bs[k][tx]);
        __syncthreads();
    }
    if (row < M && col < N) {
        float* c = &C[size_t(row) * N + col];
        KC_G(*c) = alpha * acc + (beta != 0.0f ? beta * KC_G(*c) : 0.0f);
    }
}

// 3. Register blocking: a 128x128 block tile, BK = 8, 256 threads each
//    computing an 8x8 micro-tile from registers (64 FMAs per 16 shared
//    loads). A is stored transposed in shared memory so both operands are
//    read as contiguous float4s.
//    SPLIT = false: thread (tx, ty) owns rows ty*8..+7 and columns tx*8..+7.
//    Lanes tx and tx+4 then read float4s 128 bytes apart: same banks,
//    2-way conflict on every B read.
//    SPLIT = true: the micro-tile is split in two 4-wide halves 64 apart
//    (rows ty*4..+3 and 64+ty*4..+3, same for columns): consecutive lanes
//    read consecutive float4s, conflict-free; C is written as float4s when
//    N is a multiple of 4.
//    Rows of the transposed A tile are padded by 4 floats: the store
//    As[k][m] (lanes vary k fastest) would otherwise put 8 lanes on each of
//    4 banks.
template <int BM, int BN, int BK, int TM, int TN, bool SPLIT>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_reg_kernel(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta,
                     float* __restrict__ C) {
    constexpr int NT = (BM / TM) * (BN / TN);
    static_assert(TM == 8 && TN == 8, "micro-tile is 8x8 (two float4s per operand)");
    static_assert((BM * BK) % NT == 0 && (BK * BN) % NT == 0, "tile loads divide evenly");
    __shared__ float4 As[BK][BM / 4 + 1];
    __shared__ float4 Bs[BK][BN / 4];
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    float acc[TM][TN];
    KC_UNROLL
    for (int i = 0; i < TM; i++)
        KC_UNROLL
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // A tile: 8 consecutive threads read one 32-byte row segment
        KC_UNROLL
        for (int i = 0; i < BM * BK / NT; i++) {
            int e = tid + i * NT, m = e / BK, k = e % BK;
            int gm = bm + m, gk = k0 + k;
            float v = (gm < M && gk < K) ? KC_G(A[size_t(gm) * K + gk]) : 0.0f;
            KC_S(reinterpret_cast<float*>(As[k])[m]) = v;
        }
        // B tile: consecutive threads read consecutive columns
        KC_UNROLL
        for (int i = 0; i < BK * BN / NT; i++) {
            int e = tid + i * NT, k = e / BN, n = e % BN;
            int gk = k0 + k, gn = bn + n;
            float v = (gk < K && gn < N) ? KC_G(B[size_t(gk) * N + gn]) : 0.0f;
            KC_S(reinterpret_cast<float*>(Bs[k])[n]) = v;
        }
        __syncthreads();
        KC_UNROLL
        for (int k = 0; k < BK; k++) {
            float4 a0, a1, b0, b1;
            if (SPLIT) {
                a0 = KC_S(As[k][ty]);
                a1 = KC_S(As[k][BM / 8 + ty]);
                b0 = KC_S(Bs[k][tx]);
                b1 = KC_S(Bs[k][BN / 8 + tx]);
            } else {
                a0 = KC_S(As[k][2 * ty]);
                a1 = KC_S(As[k][2 * ty + 1]);
                b0 = KC_S(Bs[k][2 * tx]);
                b1 = KC_S(Bs[k][2 * tx + 1]);
            }
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            KC_UNROLL
            for (int i = 0; i < TM; i++)
                KC_UNROLL
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    KC_UNROLL
    for (int i = 0; i < TM; i++) {
        int r = SPLIT ? (i < 4 ? ty * 4 + i : BM / 2 + ty * 4 + i - 4) : ty * TM + i;
        int gr = bm + r;
        if (gr >= M) continue;
        KC_UNROLL
        for (int half = 0; half < 2; half++) {
            int c0 = SPLIT ? half * (BN / 2) + tx * 4 : tx * TN + half * 4;
            int gc = bn + c0;
            float* p = &C[size_t(gr) * N + gc];
            if (SPLIT && N % 4 == 0 && gc + 3 < N) {
                // 16-byte aligned: N % 4 == 0, gc % 4 == 0, C from cudaMalloc
                float4 v = make_float4(alpha * acc[i][4 * half], alpha * acc[i][4 * half + 1], alpha * acc[i][4 * half + 2],
                                       alpha * acc[i][4 * half + 3]);
                if (beta != 0.0f) {
                    float4 old = KC_G(*reinterpret_cast<float4*>(p));
                    v.x += beta * old.x;
                    v.y += beta * old.y;
                    v.z += beta * old.z;
                    v.w += beta * old.w;
                }
                KC_G(*reinterpret_cast<float4*>(p)) = v;
                continue;
            }
            KC_UNROLL
            for (int j = 0; j < 4; j++) {
                if (gc + j >= N) break;
                KC_G(p[j]) = alpha * acc[i][4 * half + j] + (beta != 0.0f ? beta * KC_G(p[j]) : 0.0f);
            }
        }
    }
}

// ---------------------------------------------------------------- launchers

inline void sgemm_naive(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    auto k = sgemm_naive_kernel;
    KC_LAUNCH(k, dim3(ceil_div(N, 32), ceil_div(M, 8)), dim3(32, 8), M, N, K, alpha, A, B, beta, C);
}
inline void sgemm_smem(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    auto k = sgemm_smem_kernel<32>;
    KC_LAUNCH(k, dim3(ceil_div(N, 32), ceil_div(M, 32)), dim3(32, 32), M, N, K, alpha, A, B, beta, C);
}
template <bool SPLIT>
inline void sgemm_reg(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    auto k = sgemm_reg_kernel<128, 128, 8, 8, 8, SPLIT>;
    KC_LAUNCH(k, dim3(ceil_div(N, 128), ceil_div(M, 128)), dim3(256), M, N, K, alpha, A, B, beta, C);
}

}  // namespace kc
