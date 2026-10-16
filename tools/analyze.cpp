// Runs each kernel on the SIMT emulator with access tracing and prints JSON:
// global requests, 32-byte sectors, coalescing efficiency, bytes, shared
// requests and bank-conflict wavefronts. Usage: analyze > results.json
#include <cstdio>
#include <cstdlib>
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

// 256-byte aligned like cudaMalloc, so sector counts match a real allocation
template <class T> struct Aligned {
    using value_type = T;
    Aligned() = default;
    template <class U> Aligned(const Aligned<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(std::aligned_alloc(256, ((n * sizeof(T) + 255) / 256) * 256)); }
    void deallocate(T* p, size_t) { std::free(p); }
    template <class U> bool operator==(const Aligned<U>&) const { return true; }
};
using Vec = std::vector<float, Aligned<float>>;

static Vec randv(size_t n) {
    static std::mt19937 rng(7);
    std::normal_distribution<float> d;
    Vec v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

static std::string sites_json(const kc_emu::Trace& t) {
    std::string out = "[";
    for (auto& [k, v] : t.sites) {
        std::string site = k.substr(k.find("include/") == std::string::npos ? 0 : k.find("include/") + 8);
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s{\"site\":\"%s\",\"requests\":%llu,\"sectors\":%llu,\"bytes\":%llu,\"wavefronts\":%llu}",
                      out.size() > 1 ? "," : "", site.c_str(), (unsigned long long)v.requests, (unsigned long long)v.sectors,
                      (unsigned long long)v.bytes, (unsigned long long)v.wavefronts);
        out += buf;
    }
    return out + "]";
}

static bool first = true;
static void report(const std::string& group, const std::string& name, const std::string& shape, double flops, double min_bytes,
                   const std::function<void()>& run) {
    kc_emu::Trace t;
    kc_emu::g_trace = &t;
    run();
    kc_emu::g_trace = nullptr;
    std::printf("%s{\"group\":\"%s\",\"kernel\":\"%s\",\"shape\":\"%s\",\"flops\":%.0f,\"min_bytes\":%.0f,"
                "\"global_requests\":%llu,\"global_sectors\":%llu,\"global_bytes\":%llu,\"coalescing\":%.4f,"
                "\"shared_requests\":%llu,\"shared_wavefronts\":%llu,\"conflict_factor\":%.3f,\"blocks\":%llu,\"threads\":%llu,\"sites\":%s}\n",
                first ? "[" : ",", group.c_str(), name.c_str(), shape.c_str(), flops, min_bytes, (unsigned long long)t.global_requests,
                (unsigned long long)t.global_sectors, (unsigned long long)t.global_bytes, t.coalescing(),
                (unsigned long long)t.shared_requests, (unsigned long long)t.shared_wavefronts, t.conflict_factor(),
                (unsigned long long)t.blocks, (unsigned long long)t.threads, sites_json(t).c_str());
    std::fflush(stdout);
    first = false;
}

int main() {
    {
        int n = 1 << 18;
        auto x = randv(n);
        float out = 0;
        Vec part(kc::REDUCE_BLOCKS);
        std::string sh = "n=262144";
        report("reduce", "atomic", sh, n, 4.0 * n, [&] { kc::reduce_atomic(x.data(), n, &out); });
        report("reduce", "smem tree", sh, n, 4.0 * n, [&] { kc::reduce_smem(x.data(), n, &out); });
        report("reduce", "float4 + shuffles, two-pass", sh, n, 4.0 * n, [&] { kc::reduce_two_pass(x.data(), n, part.data(), &out); });
    }
    {
        int r = 1024, c = 1024;
        auto in = randv(size_t(r) * c);
        Vec out(in.size());
        std::string sh = "1024x1024";
        double b = 8.0 * r * c;
        report("transpose", "naive", sh, 0, b, [&] { kc::transpose_naive(in.data(), out.data(), r, c); });
        report("transpose", "smem tile", sh, 0, b, [&] { kc::transpose_tiled(in.data(), out.data(), r, c); });
        report("transpose", "smem tile, padded", sh, 0, b, [&] { kc::transpose_padded(in.data(), out.data(), r, c); });
    }
    {
        int m = 256, n = 256, k = 256;
        auto A = randv(size_t(m) * k), B = randv(size_t(k) * n);
        Vec C(size_t(m) * n);
        std::string sh = "256x256x256";
        double f = 2.0 * m * n * k, b = 4.0 * (m * k + k * n + m * n);
        report("sgemm", "naive", sh, f, b, [&] { kc::sgemm_naive(m, n, k, 1, A.data(), B.data(), 0, C.data()); });
        report("sgemm", "smem 32x32", sh, f, b, [&] { kc::sgemm_smem(m, n, k, 1, A.data(), B.data(), 0, C.data()); });
        report("sgemm", "register 8x8", sh, f, b, [&] { kc::sgemm_reg<false>(m, n, k, 1, A.data(), B.data(), 0, C.data()); });
        report("sgemm", "register 8x8, split", sh, f, b, [&] { kc::sgemm_reg<true>(m, n, k, 1, A.data(), B.data(), 0, C.data()); });
    }
    {
        int m = 256, n = 256, k = 256;
        auto A = randv(size_t(m) * k), B = randv(size_t(k) * n);
        std::vector<uint16_t, Aligned<uint16_t>> hA(A.size()), hB(B.size());
        for (size_t i = 0; i < A.size(); i++) hA[i] = kc::f32_to_f16(A[i]);
        for (size_t i = 0; i < B.size(); i++) hB[i] = kc::f32_to_f16(B[i]);
        Vec C(size_t(m) * n);
        double f = 2.0 * m * n * k, b = 2.0 * (m * k + k * n) + 4.0 * m * n;
        report("hgemm", "tensor-core mma.m16n8k16 + ldmatrix (fp16 in, fp32 acc)", "256x256x256", f, b,
               [&] { kc::hgemm_mma(m, n, k, 1, hA.data(), hB.data(), 0, C.data()); });
    }
    {
        int rows = 256, cols = 1024;
        auto x = randv(size_t(rows) * cols), g = randv(cols), b = randv(cols);
        Vec y(x.size());
        std::string sh = "256x1024";
        report("rows", "softmax (online)", sh, 0, 8.0 * rows * cols, [&] { kc::softmax(x.data(), y.data(), rows, cols); });
        report("rows", "layernorm (Welford)", sh, 0, 8.0 * rows * cols, [&] { kc::layernorm<false>(x.data(), g.data(), b.data(), y.data(), rows, cols); });
        Vec dy = randv(x.size()), dx(x.size()), dg(cols), db(cols), mu(rows), rs(rows);
        report("rows", "softmax backward", sh, 0, 12.0 * rows * cols, [&] { kc::softmax_backward(y.data(), dy.data(), dx.data(), rows, cols); });
        report("rows", "layernorm backward (dx, dgamma, dbeta)", sh, 0, 12.0 * rows * cols,
               [&] { kc::layernorm_backward(x.data(), g.data(), dy.data(), dx.data(), dg.data(), db.data(), mu.data(), rs.data(), rows, cols); });
    }
    {
        int bh = 1, n = 512;
        constexpr int D = 64;
        size_t sz = size_t(bh) * n * D;
        auto Q = randv(sz), K = randv(sz), V = randv(sz);
        Vec O(sz), S(size_t(bh) * n * n);
        std::string sh = "BH=1 N=512 D=64";
        double f = 4.0 * bh * n * n * D, b = 16.0 * sz;
        report("attention", "flash (tiled, online softmax)", sh, f, b, [&] { kc::flash_attention<D>(Q.data(), K.data(), V.data(), O.data(), bh, n, false); });
        report("attention", "naive (scores in global memory)", sh, f, b, [&] { kc::attention_naive<D>(Q.data(), K.data(), V.data(), O.data(), S.data(), bh, n, false); });
    }
    std::printf("]\n");
}
