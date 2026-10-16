// Tensor-core GEMM, row-major: C[M,N] (fp32) = alpha * A[M,K] * B[K,N] + beta * C
// with A and B in fp16 (raw IEEE binary16 bits in uint16_t) and fp32
// accumulation, on mma.sync.m16n8k16 (sm_80 and newer).
//
// * Block tile BM x BN x BK = 128 x 128 x 32, 8 warps in a 2 x 4 grid, each
//   warp owning a 64 x 32 tile = 4 x 4 mma tiles of 16 x 8 (64 fp32
//   accumulators per thread).
// * Global -> shared: 16-byte vector loads when K and N are multiples of 8
//   (VEC), element loads otherwise; out-of-range elements are zero-filled, so
//   any M, N, K works.
// * Shared -> registers: ldmatrix.x4 for A, ldmatrix.x4.trans for B (B is
//   row-major K x N, the mma wants it "col"). Rows are padded by 8 halves
//   (16 bytes) so the 8 row addresses of each 8x8 matrix fall in distinct
//   banks.
// * Not done (see docs/limitations.md): cp.async multi-stage pipelining,
//   swizzled shared layouts, split-K, autotuning.
#pragma once
#include <cstdint>
#include "compat.h"
#include "mma.cuh"

namespace kc {

// 16-byte copies as single vector instructions (clang otherwise copies the
// uint4 struct through a local-memory temporary)
#if defined(KC_CUDA)
__device__ __forceinline__ uint4 load16(const uint4* p) { return __ldg(p); }
__device__ __forceinline__ void store16(uint4* p, uint4 v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(smem_addr(p)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 zero16() { return make_uint4(0u, 0u, 0u, 0u); }
#else
inline uint4 load16(const uint4* p) { return *p; }
inline void store16(uint4* p, uint4 v) { *p = v; }
inline uint4 zero16() { return uint4{0u, 0u, 0u, 0u}; }
#endif

template <int BM, int BN, int BK, int WM, int WN, bool VEC>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
    hgemm_mma_kernel(int M, int N, int K, float alpha, const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, float beta,
                     float* __restrict__ C) {
    constexpr int WARPS_N = BN / WN, NT = (BM / WM) * (BN / WN) * 32;
    constexpr int MT = WM / 16, NTL = WN / 8;  // mma tiles per warp
    constexpr int AP = BK + 8, BP = BN + 8;    // padded row lengths (halves)
    static_assert(BK % 16 == 0 && WM % 16 == 0 && WN % 16 == 0, "tile shapes");
    static_assert((BM * BK / 8) % NT == 0 && (BK * BN / 8) % NT == 0, "vector tile loads divide evenly");
    struct alignas(16) Smem {
        uint16_t a[BM * AP];
        uint16_t b[BK * BP];
    };
    __shared__ Smem sm;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    float acc[MT][NTL][4];
    KC_UNROLL
    for (int i = 0; i < MT; i++)
        KC_UNROLL
        for (int j = 0; j < NTL; j++)
            KC_UNROLL
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (VEC) {
            // K % 8 == 0 and N % 8 == 0: an 8-half chunk is either fully in range or fully out
            KC_UNROLL
            for (int i = 0; i < BM * BK / 8 / NT; i++) {
                int c = tid + i * NT, m = c / (BK / 8), k = (c % (BK / 8)) * 8;
                int gm = bm + m, gk = k0 + k;
                uint4 v = (gm < M && gk < K) ? load16(&KC_G(*reinterpret_cast<const uint4*>(&A[size_t(gm) * K + gk]))) : zero16();
                store16(&KC_S(*reinterpret_cast<uint4*>(&sm.a[m * AP + k])), v);
            }
            KC_UNROLL
            for (int i = 0; i < BK * BN / 8 / NT; i++) {
                int c = tid + i * NT, k = c / (BN / 8), n = (c % (BN / 8)) * 8;
                int gk = k0 + k, gn = bn + n;
                uint4 v = (gk < K && gn < N) ? load16(&KC_G(*reinterpret_cast<const uint4*>(&B[size_t(gk) * N + gn]))) : zero16();
                store16(&KC_S(*reinterpret_cast<uint4*>(&sm.b[k * BP + n])), v);
            }
        } else {
            for (int e = tid; e < BM * BK; e += NT) {
                int m = e / BK, k = e % BK, gm = bm + m, gk = k0 + k;
                KC_S(sm.a[m * AP + k]) = (gm < M && gk < K) ? KC_G(A[size_t(gm) * K + gk]) : uint16_t(0);
            }
            for (int e = tid; e < BK * BN; e += NT) {
                int k = e / BN, n = e % BN, gk = k0 + k, gn = bn + n;
                KC_S(sm.b[k * BP + n]) = (gk < K && gn < N) ? KC_G(B[size_t(gk) * N + gn]) : uint16_t(0);
            }
        }
        __syncthreads();
        KC_UNROLL
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[MT][4], b[NTL][2];
            KC_UNROLL
            for (int i = 0; i < MT; i++)  // matrices: rows 0-7 / 8-15 x cols 0-7 / 8-15 -> a0..a3
                ldmatrix_x4(a[i], &sm.a[(wm * WM + i * 16 + lane % 16) * AP + kk + (lane / 16) * 8]);
            KC_UNROLL
            for (int j = 0; j < NTL; j += 2) {  // k 0-7 / 8-15 x two n8 tiles -> b0, b1 of tiles j, j+1
                uint32_t r[4];
                ldmatrix_x4_trans(r, &sm.b[(kk + lane % 16) * BP + wn * WN + j * 8 + (lane / 16) * 8]);
                b[j][0] = r[0];
                b[j][1] = r[1];
                b[j + 1][0] = r[2];
                b[j + 1][1] = r[3];
            }
            KC_UNROLL
            for (int i = 0; i < MT; i++)
                KC_UNROLL
                for (int j = 0; j < NTL; j++) mma_m16n8k16(acc[i][j], a[i], b[j]);
        }
        __syncthreads();
    }
    // epilogue: lane holds (g, 2t..2t+1) and (g+8, 2t..2t+1) of each 16x8 tile
    const int g = lane / 4, t = lane % 4;
    KC_UNROLL
    for (int i = 0; i < MT; i++)
        KC_UNROLL
        for (int j = 0; j < NTL; j++)
            KC_UNROLL
            for (int h = 0; h < 2; h++) {
                int gr = bm + wm * WM + i * 16 + g + 8 * h, gc = bn + wn * WN + j * 8 + 2 * t;
                if (gr >= M || gc >= N) continue;
                float* p = &C[size_t(gr) * N + gc];
                float v0 = alpha * acc[i][j][2 * h], v1 = alpha * acc[i][j][2 * h + 1];
                if (VEC) {  // N even, gc even: 8-byte aligned and gc + 1 < N
                    float2 v = {v0, v1};
                    if (beta != 0.0f) {
                        float2 old = KC_G(*reinterpret_cast<float2*>(p));
                        v.x += beta * old.x;
                        v.y += beta * old.y;
                    }
                    KC_G(*reinterpret_cast<float2*>(p)) = v;
                } else {
                    KC_G(p[0]) = v0 + (beta != 0.0f ? beta * KC_G(p[0]) : 0.0f);
                    if (gc + 1 < N) KC_G(p[1]) = v1 + (beta != 0.0f ? beta * KC_G(p[1]) : 0.0f);
                }
            }
}

// The vector path needs 16-byte aligned rows: K % 8 == 0, N % 8 == 0 and
// 16-byte aligned A and B (cudaMalloc gives 256), 8-byte aligned C.
inline void hgemm_mma(int M, int N, int K, float alpha, const uint16_t* A, const uint16_t* B, float beta, float* C) {
    dim3 grid(ceil_div(N, 128), ceil_div(M, 128)), block(256);
    bool vec = K % 8 == 0 && N % 8 == 0 && reinterpret_cast<uintptr_t>(A) % 16 == 0 && reinterpret_cast<uintptr_t>(B) % 16 == 0 &&
               reinterpret_cast<uintptr_t>(C) % 8 == 0;
    if (vec) {
        auto k = hgemm_mma_kernel<128, 128, 32, 64, 32, true>;
        KC_LAUNCH(k, grid, block, M, N, K, alpha, A, B, beta, C);
    } else {
        auto k = hgemm_mma_kernel<128, 128, 32, 64, 32, false>;
        KC_LAUNCH(k, grid, block, M, N, K, alpha, A, B, beta, C);
    }
}

}  // namespace kc
