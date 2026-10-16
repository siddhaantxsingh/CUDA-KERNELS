// The emulator's access model on patterns with known answers.
#include <cstdio>
#include <cstdlib>

#include "kc/compat.h"

static int failures = 0;
#define EXPECT_NEAR(a, b, what)                                                            \
    do {                                                                                   \
        double _a = (a), _b = (b);                                                         \
        if (std::fabs(_a - _b) > 1e-9) {                                                   \
            failures++;                                                                    \
            std::printf("FAIL %s: got %g want %g\n", what, _a, _b);                        \
        } else {                                                                           \
            std::printf("ok   %s = %g\n", what, _a);                                       \
        }                                                                                  \
    } while (0)

__global__ void global_stride(const float* x, float* y, int stride) { y[threadIdx.x] = KC_G(x[threadIdx.x * stride]); }
__global__ void global_float4(const float4* x, float* y) { y[threadIdx.x] = KC_G(x[threadIdx.x]).x; }
__global__ void global_offset(const float* x, float* y) { y[threadIdx.x] = KC_G(x[threadIdx.x + 1]); }
__global__ void global_same(const float* x, float* y) { y[threadIdx.x] = KC_G(x[0]); }
__global__ void global_predicated(const float* x, float* y) {
    if (threadIdx.x % 2 == 0) y[threadIdx.x] = KC_G(x[threadIdx.x]);
}

template <int STRIDE>
__global__ void shared_stride(float* y) {
    __shared__ float s[32 * 33];
    s[threadIdx.x] = 1.0f;
    __syncthreads();
    y[threadIdx.x] = KC_S(s[(threadIdx.x * STRIDE) % (32 * 33)]);
}
__global__ void shared_broadcast(float* y) {
    __shared__ float s[32];
    s[threadIdx.x] = 1.0f;
    __syncthreads();
    y[threadIdx.x] = KC_S(s[0]);
}
template <int STRIDE4>
__global__ void shared_float4(float* y) {
    __shared__ float4 s[32 * 4];
    __syncthreads();
    y[threadIdx.x] = KC_S(s[threadIdx.x * STRIDE4]).x;
}

template <class F>
static kc_emu::Trace traced(F f) {
    kc_emu::Trace t;
    kc_emu::g_trace = &t;
    f();
    kc_emu::g_trace = nullptr;
    return t;
}

int main() {
    alignas(256) static float x[32 * 64 + 8];
    static float y[32];
    alignas(256) static float4 x4[32];
    auto t = traced([&] { KC_LAUNCH(global_stride, dim3(1), dim3(32), x, y, 1); });
    EXPECT_NEAR(t.global_sectors, 4, "consecutive floats: 4 sectors");
    EXPECT_NEAR(t.coalescing(), 1.0, "consecutive floats: efficiency");
    t = traced([&] { KC_LAUNCH(global_stride, dim3(1), dim3(32), x, y, 2); });
    EXPECT_NEAR(t.global_sectors, 8, "stride 2: 8 sectors");
    EXPECT_NEAR(t.coalescing(), 0.5, "stride 2: efficiency");
    t = traced([&] { KC_LAUNCH(global_stride, dim3(1), dim3(32), x, y, 32); });
    EXPECT_NEAR(t.global_sectors, 32, "stride 32: one sector per lane");
    t = traced([&] { KC_LAUNCH(global_offset, dim3(1), dim3(32), x, y); });
    EXPECT_NEAR(t.global_sectors, 5, "misaligned by 4 bytes: 5 sectors");
    t = traced([&] { KC_LAUNCH(global_float4, dim3(1), dim3(32), x4, y); });
    EXPECT_NEAR(t.global_sectors, 16, "float4 per lane: 16 sectors");
    EXPECT_NEAR(t.coalescing(), 1.0, "float4: efficiency");
    t = traced([&] { KC_LAUNCH(global_same, dim3(1), dim3(32), x, y); });
    EXPECT_NEAR(t.global_sectors, 1, "broadcast: 1 sector");
    EXPECT_NEAR(t.global_requests, 1, "broadcast: 1 request");
    t = traced([&] { KC_LAUNCH(global_predicated, dim3(1), dim3(64), x, y); });
    EXPECT_NEAR(t.global_requests, 2, "predicated: one request per warp");
    EXPECT_NEAR(t.coalescing(), 0.5, "predicated: half the lanes");

    t = traced([&] { KC_LAUNCH(shared_stride<1>, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.conflict_factor(), 1.0, "shared stride 1: conflict-free");
    t = traced([&] { KC_LAUNCH(shared_stride<2>, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.conflict_factor(), 2.0, "shared stride 2: 2-way");
    t = traced([&] { KC_LAUNCH(shared_stride<32>, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.conflict_factor(), 32.0, "shared stride 32: 32-way");
    t = traced([&] { KC_LAUNCH(shared_stride<33>, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.conflict_factor(), 1.0, "shared stride 33: conflict-free");
    t = traced([&] { KC_LAUNCH(shared_broadcast, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.conflict_factor(), 1.0, "shared broadcast");
    t = traced([&] { KC_LAUNCH(shared_float4<1>, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.shared_wavefronts, 4, "float4 consecutive: 4 wavefronts (one per quarter warp)");
    EXPECT_NEAR(t.conflict_factor(), 1.0, "float4 consecutive: conflict-free");
    t = traced([&] { KC_LAUNCH(shared_float4<2>, dim3(1), dim3(32), y); });
    EXPECT_NEAR(t.conflict_factor(), 2.0, "float4 stride 2: 2-way");
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
