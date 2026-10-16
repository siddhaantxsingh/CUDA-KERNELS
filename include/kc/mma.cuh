// Warp-level tensor-core primitives (sm_80+): ldmatrix and
// mma.sync.m16n8k16 with fp16 inputs and fp32 accumulators.
// Under CUDA they are inline PTX; on the host they call the emulator's models
// in simt_emu.h, which follow the PTX ISA fragment layouts.
#pragma once
#include <cstdint>
#include "compat.h"

namespace kc {

#if defined(KC_CUDA)
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    uint32_t a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
// four 8x8 b16 matrices; lanes 8i..8i+7 address the rows of matrix i
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const uint16_t* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p))
                 : "memory");
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const uint16_t* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p))
                 : "memory");
}
// d (16x8 fp32) += a (16x16 fp16, row) * b (16x8 fp16, col)
__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#else
inline void ldmatrix_x4(uint32_t* r, const uint16_t* p) { kc_emu::ldmatrix(r, 4, false, p); }
inline void ldmatrix_x4_trans(uint32_t* r, const uint16_t* p) { kc_emu::ldmatrix(r, 4, true, p); }
inline void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) { kc_emu::mma_m16n8k16(d, a, b); }
#endif

// One warp, one mma: D[16x8] = A[16x16] * B[16x8], all row-major in global
// memory. Fragments are gathered element by element from the PTX ISA layout
// tables (no ldmatrix), so this checks the mma fragment layout on its own;
// the GPU benchmark runs it to confirm the layout on real hardware.
__global__ void mma_probe_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ B, float* __restrict__ D) {
    int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    auto pack = [](uint16_t lo, uint16_t hi) { return uint32_t(lo) | (uint32_t(hi) << 16); };
    uint32_t a[4], b[2];
    a[0] = pack(KC_G(A[g * 16 + 2 * t]), KC_G(A[g * 16 + 2 * t + 1]));
    a[1] = pack(KC_G(A[(g + 8) * 16 + 2 * t]), KC_G(A[(g + 8) * 16 + 2 * t + 1]));
    a[2] = pack(KC_G(A[g * 16 + 2 * t + 8]), KC_G(A[g * 16 + 2 * t + 9]));
    a[3] = pack(KC_G(A[(g + 8) * 16 + 2 * t + 8]), KC_G(A[(g + 8) * 16 + 2 * t + 9]));
    b[0] = pack(KC_G(B[(2 * t) * 8 + g]), KC_G(B[(2 * t + 1) * 8 + g]));
    b[1] = pack(KC_G(B[(2 * t + 8) * 8 + g]), KC_G(B[(2 * t + 9) * 8 + g]));
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16(d, a, b);
    KC_G(D[g * 8 + 2 * t]) = d[0];
    KC_G(D[g * 8 + 2 * t + 1]) = d[1];
    KC_G(D[(g + 8) * 8 + 2 * t]) = d[2];
    KC_G(D[(g + 8) * 8 + 2 * t + 1]) = d[3];
}
inline void mma_probe(const uint16_t* A, const uint16_t* B, float* D) {
    auto k = mma_probe_kernel;
    KC_LAUNCH(k, dim3(1), dim3(32), A, B, D);
}

}  // namespace kc
