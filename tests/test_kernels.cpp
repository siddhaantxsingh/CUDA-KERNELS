// Functional tests: every kernel runs unchanged on the CPU SIMT emulator and
// is compared with a double-precision reference.
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "kc/attention.cuh"
#include "kc/backward.cuh"
#include "kc/gemm.cuh"
#include "kc/half.h"
#include "kc/hgemm.cuh"
#include "kc/reduce.cuh"
#include "kc/softmax.cuh"
#include "kc/transpose.cuh"

static int failures = 0, checks = 0;
#define CHECK(cond, ...)                                       \
    do {                                                       \
        checks++;                                              \
        if (!(cond)) {                                         \
            failures++;                                        \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
            std::printf(__VA_ARGS__);                          \
            std::printf("\n");                                 \
        }                                                      \
    } while (0)

static std::mt19937 rng(12345);
static std::vector<float> randn(size_t n, float mean = 0.0f, float sd = 1.0f) {
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// ------------------------------------------------------------------ reduce

static void test_reduce() {
    for (int n : {1, 3, 31, 1000, 4097, 100003, 1 << 20}) {
        auto x = randn(n);
        double ref = 0, absum = 0;
        for (float v : x) {
            ref += v;
            absum += std::fabs(v);
        }
        // fp32 bound: each element passes through at most
        // (serial terms per thread + tree depth) roundings
        auto bound = [&](double serial) { return (serial + 20) * 6e-8 * absum + 1e-30; };
        float out = 0;
        if (n <= 100003) {  // one block per 256 elements: slow to emulate beyond this
            kc::reduce_atomic(x.data(), n, &out);
            CHECK(std::fabs(out - ref) <= bound(n), "atomic n=%d got %.7g want %.7g", n, out, ref);
            out = 0;
            kc::reduce_smem(x.data(), n, &out);
            CHECK(std::fabs(out - ref) <= bound(n / 256.0 + 1), "smem n=%d got %.7g want %.7g", n, out, ref);
        }
        std::vector<float> partials(kc::REDUCE_BLOCKS);
        float a = 0, b = 0;
        kc::reduce_two_pass(x.data(), n, partials.data(), &a);
        kc::reduce_two_pass(x.data(), n, partials.data(), &b);
        CHECK(std::fabs(a - ref) <= bound(n / (4.0 * 256 * 256) * 4 + 2), "two-pass n=%d got %.7g want %.7g", n, a, ref);
        CHECK(std::memcmp(&a, &b, 4) == 0, "two-pass is deterministic");
    }
}

// ------------------------------------------------------------------ transpose

static void test_transpose() {
    for (auto [r, c] : std::vector<std::pair<int, int>>{{1, 1}, {32, 32}, {33, 65}, {100, 7}, {256, 512}}) {
        auto in = randn(size_t(r) * c);
        std::vector<std::function<void(const float*, float*, int, int)>> fs = {kc::transpose_naive, kc::transpose_tiled, kc::transpose_padded};
        for (size_t f = 0; f < fs.size(); f++) {
            std::vector<float> out(in.size(), -1.0f);
            fs[f](in.data(), out.data(), r, c);
            bool ok = true;
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++) ok &= out[size_t(j) * r + i] == in[size_t(i) * c + j];
            CHECK(ok, "transpose variant %zu %dx%d", f, r, c);
        }
    }
}

// ------------------------------------------------------------------ gemm

static void test_gemm() {
    using F = std::function<void(int, int, int, float, const float*, const float*, float, float*)>;
    std::vector<std::pair<const char*, F>> fs = {
        {"naive", kc::sgemm_naive}, {"smem", kc::sgemm_smem}, {"reg", kc::sgemm_reg<false>}, {"reg-split", kc::sgemm_reg<true>}};
    struct S { int m, n, k; float alpha, beta; };
    for (S s : std::vector<S>{{1, 1, 1, 1, 0}, {128, 128, 8, 1, 0}, {129, 130, 17, 1, 0}, {64, 300, 64, 0.5f, 2.0f}, {256, 256, 256, 1, 1}, {5, 257, 3, -1, 0}}) {
        auto A = randn(size_t(s.m) * s.k), B = randn(size_t(s.k) * s.n), C0 = randn(size_t(s.m) * s.n);
        std::vector<double> ref(size_t(s.m) * s.n), mag(size_t(s.m) * s.n);
        for (int i = 0; i < s.m; i++)
            for (int j = 0; j < s.n; j++) {
                double acc = 0, a = 0;
                for (int k = 0; k < s.k; k++) {
                    acc += double(A[size_t(i) * s.k + k]) * B[size_t(k) * s.n + j];
                    a += std::fabs(double(A[size_t(i) * s.k + k]) * B[size_t(k) * s.n + j]);
                }
                ref[size_t(i) * s.n + j] = s.alpha * acc + s.beta * C0[size_t(i) * s.n + j];
                mag[size_t(i) * s.n + j] = std::fabs(s.alpha) * a + std::fabs(s.beta * C0[size_t(i) * s.n + j]);
            }
        for (auto& [name, f] : fs) {
            auto C = C0;
            f(s.m, s.n, s.k, s.alpha, A.data(), B.data(), s.beta, C.data());
            double worst = 0;
            for (size_t i = 0; i < C.size(); i++) worst = std::max(worst, std::fabs(C[i] - ref[i]) / ((s.k + 2) * 6e-8 * mag[i] + 1e-30));
            CHECK(worst <= 1.0, "%s %dx%dx%d: error %.2f x bound", name, s.m, s.n, s.k, worst);
        }
    }
}

// ------------------------------------------------------------------ softmax / layernorm

static void test_softmax() {
    for (auto [rows, cols] : std::vector<std::pair<int, int>>{{1, 1}, {3, 5}, {7, 32}, {9, 1000}, {64, 4096}}) {
        auto x = randn(size_t(rows) * cols, 0.0f, 10.0f);
        x[0] = 3e38f;  // would overflow exp() in a naive implementation
        if (cols > 2) x[1] = -INFINITY;
        std::vector<float> y(x.size());
        kc::softmax(x.data(), y.data(), rows, cols);
        double worst = 0;
        for (int r = 0; r < rows; r++) {
            double m = -INFINITY, s = 0;
            for (int c = 0; c < cols; c++) m = std::max(m, double(x[size_t(r) * cols + c]));
            for (int c = 0; c < cols; c++) s += std::exp(double(x[size_t(r) * cols + c]) - m);
            for (int c = 0; c < cols; c++) {
                double want = std::exp(double(x[size_t(r) * cols + c]) - m) / s;
                worst = std::max(worst, std::fabs(y[size_t(r) * cols + c] - want) / (want * 1e-5 + 1e-12));
            }
        }
        CHECK(worst <= 1.0, "softmax %dx%d error %.2f x bound", rows, cols, worst);
    }
    std::vector<float> x = {-INFINITY, -INFINITY, -INFINITY}, y(3);
    kc::softmax(x.data(), y.data(), 1, 3);
    CHECK(std::isnan(y[0]), "all -inf row gives NaN");
}

static void test_layernorm() {
    for (auto [rows, cols, mean] : std::vector<std::tuple<int, int, float>>{{1, 1, 0}, {5, 17, 0}, {8, 768, 0}, {4, 4096, 3}, {8, 1024, 1e4f}}) {
        auto x = randn(size_t(rows) * cols, mean, 1.0f);
        auto g = randn(cols, 1.0f, 0.1f), b = randn(cols, 0.0f, 0.1f);
        std::vector<float> y(x.size()), yn(x.size());
        kc::layernorm<false>(x.data(), g.data(), b.data(), y.data(), rows, cols);
        kc::layernorm<true>(x.data(), g.data(), b.data(), yn.data(), rows, cols);
        double worst = 0, worst_naive = 0;
        for (int r = 0; r < rows; r++) {
            double mu = 0, var = 0;
            for (int c = 0; c < cols; c++) mu += x[size_t(r) * cols + c];
            mu /= cols;
            for (int c = 0; c < cols; c++) var += std::pow(x[size_t(r) * cols + c] - mu, 2);
            var /= cols;
            for (int c = 0; c < cols; c++) {
                double want = (x[size_t(r) * cols + c] - mu) / std::sqrt(var + 1e-5) * g[c] + b[c];
                // fp32 input rounding alone moves (x - mean) by |x| * 6e-8
                double tol = 1e-4 + 4 * std::fabs(mean) * 6e-8 / std::sqrt(var + 1e-5) * std::fabs(g[c]);
                worst = std::max(worst, std::fabs(y[size_t(r) * cols + c] - want) / tol);
                worst_naive = std::max(worst_naive, std::fabs(yn[size_t(r) * cols + c] - want) / tol);
            }
        }
        CHECK(worst <= 1.0, "layernorm %dx%d mean %g: Welford error %.2f x bound", rows, cols, mean, worst);
        if (mean >= 1e4f) CHECK(worst_naive > 100, "naive E[x^2]-E[x]^2 should break down at mean %g (error %.1f x)", mean, worst_naive);
        if (mean == 0) CHECK(worst_naive <= 1.0, "naive variant is fine for centred data");
    }
}


// ------------------------------------------------------------------ fp16 conversion

static void test_half() {
    // every fp16 value survives fp16 -> fp32 -> fp16 (NaNs stay NaN)
    bool ok = true;
    for (uint32_t h = 0; h < 65536; h++) {
        float f = kc::f16_to_f32(uint16_t(h));
        bool nan = (h & 0x7c00u) == 0x7c00u && (h & 0x3ffu);
        ok &= nan ? std::isnan(f) : kc::f32_to_f16(f) == h;
        _Float16 ref;
        std::memcpy(&ref, &h, 2);
        ok &= nan || float(ref) == f || (f != f);
    }
    CHECK(ok, "fp16 -> fp32 -> fp16 round trip and fp16 -> fp32 agrees with the compiler's _Float16");
    // float -> fp16 rounding agrees with the compiler's _Float16 conversion (round to nearest even)
    std::uniform_int_distribution<uint32_t> bits;
    int bad = 0;
    for (int i = 0; i < 2000000; i++) {
        uint32_t u = bits(rng);
        // concentrate half the samples on the fp16 range, including subnormals and overflow
        if (i % 2) u = (u & 0x807fffffu) | ((100u + (u >> 23) % 50u) << 23);
        float f;
        std::memcpy(&f, &u, 4);
        if (std::isnan(f)) continue;
        _Float16 ref = static_cast<_Float16>(f);
        uint16_t rb;
        std::memcpy(&rb, &ref, 2);
        bad += kc::f32_to_f16(f) != rb;
    }
    for (float f : {65504.0f, 65519.99f, 65520.0f, 5.96e-8f, 2.98e-8f, 2.9802322e-8f, 1e-10f, -0.0f, INFINITY}) {
        _Float16 ref = static_cast<_Float16>(f);
        uint16_t rb;
        std::memcpy(&rb, &ref, 2);
        bad += kc::f32_to_f16(f) != rb;
    }
    CHECK(bad == 0, "float -> fp16 matches _Float16 conversion on 2M random floats and edge cases (%d mismatches)", bad);
}

// ------------------------------------------------------------------ tensor cores

static std::vector<uint16_t> to_half(const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); i++) h[i] = kc::f32_to_f16(v[i]);
    return h;
}

static void test_mma_probe() {
    // one mma with fragments gathered straight from the PTX layout tables:
    // small integers make every product and sum exact, so D must be exact
    std::vector<float> A(256), B(128);
    for (int i = 0; i < 256; i++) A[i] = float((i * 7) % 13 - 6);
    for (int i = 0; i < 128; i++) B[i] = float((i * 5) % 11 - 5);
    auto hA = to_half(A), hB = to_half(B);
    std::vector<float> D(128, -1.0f);
    kc::mma_probe(hA.data(), hB.data(), D.data());
    bool ok = true;
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 8; c++) {
            float ref = 0;
            for (int k = 0; k < 16; k++) ref += A[r * 16 + k] * B[k * 8 + c];
            ok &= D[r * 8 + c] == ref;
        }
    CHECK(ok, "mma.m16n8k16 fragment layout: one warp, exact integer 16x16x8 product");
}

// tolerance: products of fp16 values are exact in fp32; K fp32 additions plus
// the alpha/beta epilogue give |err| <= (K + 2) u sum|a_ik b_kj| (+ beta term),
// with u = 2^-24 for round-to-nearest. Tensor cores are documented by
// measurement (Fasi et al., 2021) to truncate in their internal adder, so we
// use u = 2^-23 to keep the bound valid for the hardware as well.
static void test_hgemm() {
    struct S { int m, n, k; float alpha, beta; };
    const double u = std::ldexp(1.0, -23);
    for (S s : std::vector<S>{{1, 1, 1, 1, 0},
                              {16, 8, 16, 1, 0},
                              {128, 128, 32, 1, 0},
                              {129, 130, 17, 1, 0},
                              {64, 300, 64, 0.5f, 2.0f},
                              {256, 256, 256, 1, 1},
                              {5, 257, 3, -1, 0},
                              {200, 136, 40, 1, 0},
                              {33, 72, 8, 1, 0.5f},
                              {130, 264, 520, 1, 0}}) {
        auto A = randn(size_t(s.m) * s.k), B = randn(size_t(s.k) * s.n), C0 = randn(size_t(s.m) * s.n);
        auto hA = to_half(A), hB = to_half(B);
        std::vector<double> ref(size_t(s.m) * s.n), mag(ref.size()), ref32(ref.size());
        for (int i = 0; i < s.m; i++)
            for (int j = 0; j < s.n; j++) {
                double acc = 0, a = 0, acc32 = 0;
                for (int k = 0; k < s.k; k++) {
                    double p = double(kc::f16_to_f32(hA[size_t(i) * s.k + k])) * kc::f16_to_f32(hB[size_t(k) * s.n + j]);
                    acc += p;
                    a += std::fabs(p);
                    acc32 += double(A[size_t(i) * s.k + k]) * B[size_t(k) * s.n + j];
                }
                size_t idx = size_t(i) * s.n + j;
                ref[idx] = s.alpha * acc + s.beta * C0[idx];
                ref32[idx] = s.alpha * acc32 + s.beta * C0[idx];
                mag[idx] = std::fabs(s.alpha) * a + std::fabs(s.beta * C0[idx]);
            }
        auto C = C0;
        kc::hgemm_mma(s.m, s.n, s.k, s.alpha, hA.data(), hB.data(), s.beta, C.data());
        double worst = 0, abs16 = 0, abs32 = 0;
        for (size_t i = 0; i < C.size(); i++) {
            double tol = (s.k + 2) * u * mag[i] + 1e-30;
            worst = std::max(worst, std::fabs(C[i] - ref[i]) / tol);
            abs16 = std::max(abs16, std::fabs(C[i] - ref[i]));
            abs32 = std::max(abs32, std::fabs(C[i] - ref32[i]));
        }
        CHECK(worst <= 1.0, "hgemm %dx%dx%d (alpha %g beta %g): error %.3f x bound", s.m, s.n, s.k, s.alpha, s.beta, worst);
        // the inputs really are rounded to fp16: the result is far closer to the
        // fp16-input reference than to the unrounded fp32-input one
        if (s.k >= 256)
            CHECK(abs32 > 30 * abs16, "hgemm %dx%dx%d: max error vs fp16-input reference %.2g, vs fp32-input reference %.2g", s.m, s.n, s.k, abs16, abs32);
        if (s.k >= 256) std::printf("  hgemm %dx%dx%d: max |error| vs fp16-input ref %.2g (%.3f x bound), vs fp32-input ref %.2g\n", s.m, s.n, s.k, abs16, worst, abs32);
    }
    // K tails must be zero-filled, never read: A and B are followed by fp16 NaNs,
    // which would poison the result if a K-tail element were loaded (NaN * 0 = NaN)
    for (auto [m, n, k] : std::vector<std::tuple<int, int, int>>{{200, 136, 40}, {129, 130, 17}}) {
        auto A = randn(size_t(m) * k), B = randn(size_t(k) * n);
        auto hA = to_half(A), hB = to_half(B);
        hA.resize(hA.size() + 256, 0x7e00);
        hB.resize(hB.size() + 256, 0x7e00);
        std::vector<float> C(size_t(m) * n);
        kc::hgemm_mma(m, n, k, 1, hA.data(), hB.data(), 0, C.data());
        bool finite = true;
        for (float v : C) finite &= std::isfinite(v);
        CHECK(finite, "hgemm %dx%dx%d: no element beyond the K tail is read (NaN padding)", m, n, k);
    }
    {  // misaligned A forces the element-load path although K and N are multiples of 8
        int m = 40, n = 48, k = 24;
        auto A = randn(size_t(m) * k + 1), B = randn(size_t(k) * n);
        auto hA = to_half(A), hB = to_half(B);
        std::vector<float> C(size_t(m) * n), C2(C.size());
        kc::hgemm_mma(m, n, k, 1, hA.data() + 1, hB.data(), 0, C.data());
        std::vector<uint16_t> hA2(hA.begin() + 1, hA.end());
        kc::hgemm_mma(m, n, k, 1, hA2.data(), hB.data(), 0, C2.data());
        CHECK(C == C2, "hgemm element-load path (misaligned A) equals the vector path bit for bit");
    }
}

// ------------------------------------------------------------------ backward passes

// double-precision references
static void softmax_ref(const double* x, double* y, int n) {
    double m = -INFINITY, s = 0;
    for (int i = 0; i < n; i++) m = std::max(m, x[i]);
    for (int i = 0; i < n; i++) s += (y[i] = std::exp(x[i] - m));
    for (int i = 0; i < n; i++) y[i] /= s;
}
static void layernorm_ref(const double* x, const double* g, const double* b, double* y, int n, double eps) {
    double mu = 0, var = 0;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mu) * (x[i] - mu);
    var /= n;
    for (int i = 0; i < n; i++) y[i] = (x[i] - mu) / std::sqrt(var + eps) * g[i] + b[i];
}

static void test_softmax_bwd() {
    for (auto [rows, cols] : std::vector<std::pair<int, int>>{{1, 1}, {3, 5}, {7, 32}, {9, 1000}, {16, 4096}}) {
        auto x = randn(size_t(rows) * cols, 0.0f, 3.0f), dy = randn(size_t(rows) * cols);
        std::vector<float> y(x.size()), dx(x.size());
        kc::softmax(x.data(), y.data(), rows, cols);
        kc::softmax_backward(y.data(), dy.data(), dx.data(), rows, cols);
        double worst = 0, worst_fd = 0;
        for (int r = 0; r < rows; r++) {
            std::vector<double> xd(x.begin() + size_t(r) * cols, x.begin() + size_t(r + 1) * cols), yd(cols);
            softmax_ref(xd.data(), yd.data(), cols);
            double dot = 0, sabs = 0;
            for (int c = 0; c < cols; c++) {
                dot += yd[c] * dy[size_t(r) * cols + c];
                sabs += yd[c] * std::fabs(dy[size_t(r) * cols + c]);
            }
            for (int c = 0; c < cols; c++) {
                double want = yd[c] * (dy[size_t(r) * cols + c] - dot);
                // y has ~1e-6 relative error from the forward kernel; the dot product adds cols roundings
                double tol = yd[c] * (std::fabs(dy[size_t(r) * cols + c]) + sabs) * (4e-6 + (cols + 4) * 6e-8) + 1e-30;
                worst = std::max(worst, std::fabs(dx[size_t(r) * cols + c] - want) / tol);
            }
            // the analytic formula against central differences of L = sum dy * softmax(x), in double
            for (int c : {0, cols / 2, cols - 1}) {
                const double h = 1e-5;
                std::vector<double> xp = xd, xm = xd, yp(cols), ym(cols);
                xp[c] += h;
                xm[c] -= h;
                softmax_ref(xp.data(), yp.data(), cols);
                softmax_ref(xm.data(), ym.data(), cols);
                double lp = 0, lm = 0;
                for (int j = 0; j < cols; j++) {
                    lp += dy[size_t(r) * cols + j] * yp[j];
                    lm += dy[size_t(r) * cols + j] * ym[j];
                }
                double fd = (lp - lm) / (2 * h), want = yd[c] * (dy[size_t(r) * cols + c] - dot);
                worst_fd = std::max(worst_fd, std::fabs(fd - want) / (1e-7 * (std::fabs(want) + sabs) + 1e-12));
            }
        }
        CHECK(worst <= 1.0, "softmax backward %dx%d: error %.3f x bound", rows, cols, worst);
        CHECK(worst_fd <= 1.0, "softmax backward %dx%d: analytic reference vs finite differences %.3f x bound", rows, cols, worst_fd);
    }
}

static void test_layernorm_bwd() {
    const double eps = 1e-5;
    for (auto [rows, cols, mean] : std::vector<std::tuple<int, int, float>>{{1, 1, 0}, {3, 5, 0}, {7, 32, 0}, {9, 1000, 0}, {16, 768, 3}, {4, 4096, 0}, {300, 64, 0}}) {
        auto x = randn(size_t(rows) * cols, mean, 1.0f), dy = randn(size_t(rows) * cols);
        auto g = randn(cols, 1.0f, 0.1f);
        std::vector<float> dx(x.size()), dg(cols), db(cols), mu(rows), rs(rows);
        kc::layernorm_backward(x.data(), g.data(), dy.data(), dx.data(), dg.data(), db.data(), mu.data(), rs.data(), rows, cols, float(eps));
        std::vector<double> dg_ref(cols, 0), db_ref(cols, 0), dg_mag(cols, 0), db_mag(cols, 0);
        double worst = 0, worst_fd = 0;
        for (int r = 0; r < rows; r++) {
            const float* xr = &x[size_t(r) * cols];
            const float* dyr = &dy[size_t(r) * cols];
            double m = 0, var = 0;
            for (int c = 0; c < cols; c++) m += xr[c];
            m /= cols;
            for (int c = 0; c < cols; c++) var += (xr[c] - m) * (xr[c] - m);
            var /= cols;
            double rstd = 1 / std::sqrt(var + eps), sg = 0, sgx = 0, ag = 0, agx = 0;
            std::vector<double> xh(cols);
            for (int c = 0; c < cols; c++) {
                xh[c] = (xr[c] - m) * rstd;
                double gg = double(dyr[c]) * g[c];
                sg += gg;
                sgx += gg * xh[c];
                ag += std::fabs(gg);
                agx += std::fabs(gg * xh[c]);
                dg_ref[c] += dyr[c] * xh[c];
                db_ref[c] += dyr[c];
                dg_mag[c] += std::fabs(dyr[c] * xh[c]);
                db_mag[c] += std::fabs(dyr[c]);
            }
            sg /= cols;
            sgx /= cols;
            for (int c = 0; c < cols; c++) {
                double want = rstd * (double(dyr[c]) * g[c] - sg - xh[c] * sgx);
                // each term carries ~(cols + 16) fp32 roundings (sums, Welford statistics);
                // (x - mean) loses |mean| * u to the rounding of the mean
                double terms = rstd * (std::fabs(double(dyr[c]) * g[c]) + ag / cols + (std::fabs(xh[c]) + 1) * agx / cols);
                double tol = 4 * (cols + 16) * 6e-8 * terms + 4 * std::fabs(mean) * 6e-8 * rstd * rstd * agx / cols * cols + 1e-30;
                worst = std::max(worst, std::fabs(dx[size_t(r) * cols + c] - want) / tol);
            }
            if (r < 2) {  // analytic reference vs central differences of L = sum dy * LN(x), in double
                std::vector<double> xd(xr, xr + cols), gd(g.begin(), g.end()), bd(cols, 0.0), yp(cols), ym(cols);
                for (int c : {0, cols / 2, cols - 1}) {
                    const double h = 1e-5;
                    auto loss = [&](std::vector<double>& xx) {
                        std::vector<double> yy(cols);
                        layernorm_ref(xx.data(), gd.data(), bd.data(), yy.data(), cols, eps);
                        double l = 0;
                        for (int j = 0; j < cols; j++) l += dyr[j] * yy[j];
                        return l;
                    };
                    xd[c] += h;
                    double lp = loss(xd);
                    xd[c] -= 2 * h;
                    double lm = loss(xd);
                    xd[c] += h;
                    double fd = (lp - lm) / (2 * h), want = rstd * (double(dyr[c]) * g[c] - sg - xh[c] * sgx);
                    worst_fd = std::max(worst_fd, std::fabs(fd - want) / (1e-6 * (std::fabs(want) + rstd * ag / cols) + 1e-12));
                }
            }
        }
        double worst_g = 0, worst_b = 0;
        for (int c = 0; c < cols; c++) {
            worst_g = std::max(worst_g, std::fabs(dg[c] - dg_ref[c]) / (4 * (rows + cols + 16) * 6e-8 * dg_mag[c] + 1e-30));
            worst_b = std::max(worst_b, std::fabs(db[c] - db_ref[c]) / ((rows + 2) * 6e-8 * db_mag[c] + 1e-30));
        }
        CHECK(worst <= 1.0, "layernorm backward dx %dx%d mean %g: error %.3f x bound", rows, cols, mean, worst);
        CHECK(worst_g <= 1.0 && worst_b <= 1.0, "layernorm backward dgamma/dbeta %dx%d: errors %.3f / %.3f x bound", rows, cols, worst_g, worst_b);
        CHECK(worst_fd <= 1.0, "layernorm backward %dx%d: analytic reference vs finite differences %.3f x bound", rows, cols, worst_fd);
    }
    {  // dgamma by finite differences on gamma (double), one small case
        int rows = 5, cols = 7;
        auto x = randn(size_t(rows) * cols), dy = randn(size_t(rows) * cols), g = randn(cols, 1.0f, 0.1f);
        std::vector<float> dx(x.size()), dg(cols), db(cols), mu(rows), rs(rows);
        kc::layernorm_backward(x.data(), g.data(), dy.data(), dx.data(), dg.data(), db.data(), mu.data(), rs.data(), rows, cols, float(eps));
        double worst = 0;
        for (int c = 0; c < cols; c++) {
            auto loss = [&](double gc, double bc) {
                double l = 0;
                for (int r = 0; r < rows; r++) {
                    std::vector<double> xd(&x[size_t(r) * cols], &x[size_t(r) * cols] + cols), gd(g.begin(), g.end()), bd(cols, 0.0), yy(cols);
                    gd[c] = gc;
                    bd[c] = bc;
                    layernorm_ref(xd.data(), gd.data(), bd.data(), yy.data(), cols, eps);
                    for (int j = 0; j < cols; j++) l += dy[size_t(r) * cols + j] * yy[j];
                }
                return l;
            };
            const double h = 1e-5;
            double fdg = (loss(g[c] + h, 0) - loss(g[c] - h, 0)) / (2 * h);
            double fdb = (loss(g[c], h) - loss(g[c], -h)) / (2 * h);
            worst = std::max({worst, std::fabs(dg[c] - fdg) / (1e-5 * (1 + std::fabs(fdg))), std::fabs(db[c] - fdb) / (1e-5 * (1 + std::fabs(fdb)))});
        }
        CHECK(worst <= 1.0, "layernorm backward dgamma/dbeta vs finite differences: %.3f x bound", worst);
    }
}

// ------------------------------------------------------------------ attention

template <int D>
static void test_attention_d() {
    for (auto [bh, n, causal] : std::vector<std::tuple<int, int, bool>>{{1, 1, false}, {2, 37, false}, {2, 37, true}, {3, 130, true}, {1, 200, false}}) {
        size_t sz = size_t(bh) * n * D;
        auto Q = randn(sz), K = randn(sz), V = randn(sz);
        std::vector<float> O(sz), On(sz), S(size_t(bh) * n * n);
        kc::flash_attention<D>(Q.data(), K.data(), V.data(), O.data(), bh, n, causal);
        kc::attention_naive<D>(Q.data(), K.data(), V.data(), On.data(), S.data(), bh, n, causal);
        double worst = 0, worst_n = 0;
        double scale = 1.0 / std::sqrt(double(D));
        for (int h = 0; h < bh; h++)
            for (int i = 0; i < n; i++) {
                std::vector<double> s(n);
                double m = -INFINITY;
                int end = causal ? i + 1 : n;
                for (int j = 0; j < end; j++) {
                    double acc = 0;
                    for (int d = 0; d < D; d++) acc += double(Q[(size_t(h) * n + i) * D + d]) * K[(size_t(h) * n + j) * D + d];
                    s[j] = acc * scale;
                    m = std::max(m, s[j]);
                }
                double l = 0;
                for (int j = 0; j < end; j++) l += (s[j] = std::exp(s[j] - m));
                for (int d = 0; d < D; d++) {
                    double o = 0;
                    for (int j = 0; j < end; j++) o += s[j] * V[(size_t(h) * n + j) * D + d];
                    o /= l;
                    size_t idx = (size_t(h) * n + i) * D + d;
                    worst = std::max(worst, std::fabs(O[idx] - o));
                    worst_n = std::max(worst_n, std::fabs(On[idx] - o));
                }
            }
        CHECK(worst < 2e-5, "flash D=%d bh=%d n=%d causal=%d max abs error %.3g", D, bh, n, int(causal), worst);
        CHECK(worst_n < 2e-5, "naive attention D=%d n=%d max abs error %.3g", D, n, worst_n);
    }
}

int main(int argc, char** argv) {
    std::vector<std::pair<std::string, void (*)()>> tests = {
        {"reduce", test_reduce},       {"transpose", test_transpose}, {"gemm", test_gemm},
        {"softmax", test_softmax},     {"layernorm", test_layernorm}, {"attention_d32", test_attention_d<32>},
        {"attention_d64", test_attention_d<64>},
        {"half", test_half},           {"mma_probe", test_mma_probe}, {"hgemm", test_hgemm},
        {"softmax_bwd", test_softmax_bwd}, {"layernorm_bwd", test_layernorm_bwd},
    };
    for (auto& [name, f] : tests) {
        if (argc > 1 && name.find(argv[1]) == std::string::npos) continue;
        int before = failures, c0 = checks;
        f();
        std::printf("%-16s %s (%d checks)\n", name.c_str(), failures == before ? "ok" : "FAILED", checks - c0);
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
