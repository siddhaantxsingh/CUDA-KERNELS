// CPU SIMT emulator: runs the CUDA kernels in this repository unchanged
// (same source) on the host, for functional testing without a GPU.
//
// * Each thread block runs as blockDim real OS threads; blocks run one after
//   another. __syncthreads() is a block-wide std::barrier, so data races
//   and missing barriers behave as they would on a GPU (or worse).
// * __shared__ variables are function statics: shared by the threads of
//   the running block, and safe because only one block runs at a time.
// * Warp shuffles exchange values through a per-warp buffer guarded by a
//   per-warp barrier; all lanes of the warp must take part (the kernels
//   here only shuffle with full warps).
// * Optional tracing (kc_emu::Trace): every access wrapped in KC_G (global)
//   or KC_S (shared) is recorded with its call site. Accesses are grouped
//   into warp instructions by (warp, call site, per-thread occurrence), so
//   lanes that skip a predicated access simply do not take part. From the
//   groups the tracer derives 32-byte global sectors per request and shared
//   memory bank-conflict wavefronts. This is a model of the hardware, not a
//   measurement (see docs/analysis.md).
#pragma once

#include <array>
#include <atomic>
#include <barrier>
#include <functional>
#include <semaphore>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "half.h"

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
#define __launch_bounds__(...)

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(8) float2 { float x, y; };
struct alignas(16) uint4 { uint32_t x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

using std::max;
using std::min;

namespace kc_emu {

inline thread_local dim3 tl_threadIdx, tl_blockIdx;
inline dim3 g_blockDim, g_gridDim;
inline thread_local unsigned tl_warp, tl_lane;

struct Access {
    const char* file;
    int line;
    uint32_t occurrence;
    uintptr_t addr;
    uint32_t bytes;
    uint8_t space;  // 0 global, 1 shared
    uint32_t tid;
};

struct SiteStats {
    uint64_t requests = 0, sectors = 0, bytes = 0, wavefronts = 0;
};

// Aggregated statistics of a traced launch.
struct Trace {
    std::map<std::string, SiteStats> sites;  // "file:line" -> totals
    uint64_t global_requests = 0, global_sectors = 0, global_bytes = 0;
    uint64_t global_load_bytes = 0, global_store_bytes = 0;
    uint64_t shared_requests = 0, shared_wavefronts = 0, shared_ideal = 0;
    uint64_t blocks = 0, threads = 0, barriers = 0;
    void add(const Trace& o) {
        global_requests += o.global_requests; global_sectors += o.global_sectors; global_bytes += o.global_bytes;
        global_load_bytes += o.global_load_bytes; global_store_bytes += o.global_store_bytes;
        shared_requests += o.shared_requests; shared_wavefronts += o.shared_wavefronts; shared_ideal += o.shared_ideal;
        blocks += o.blocks; threads += o.threads; barriers += o.barriers;
        for (auto& [k, v] : o.sites) {
            auto& d = sites[k];
            d.requests += v.requests; d.sectors += v.sectors; d.bytes += v.bytes; d.wavefronts += v.wavefronts;
        }
    }
    // distinct bytes requested / bytes in the touched 32-byte sectors
    // (lanes reading the same address count once: a broadcast is free)
    double coalescing() const { return global_sectors ? double(global_bytes) / (32.0 * global_sectors) : 1.0; }
    // wavefronts / the minimum for the access width (one per phase):
    // 1.0 = conflict-free, 32 = every lane serialised
    double conflict_factor() const { return shared_ideal ? double(shared_wavefronts) / shared_ideal : 1.0; }
};

inline Trace* g_trace = nullptr;
inline thread_local std::vector<Access>* tl_log = nullptr;
inline thread_local std::unordered_map<uint64_t, uint32_t>* tl_occ = nullptr;
inline thread_local bool tl_store = false;

struct Block {
    std::unique_ptr<std::barrier<>> bar;
    std::vector<std::unique_ptr<std::barrier<>>> warp_bar;
    std::vector<std::array<uint64_t, 32>> shfl;
    std::vector<std::array<std::array<uint32_t, 8>, 32>> wx;  // per-warp exchange for ldmatrix / mma
    std::atomic<uint64_t> barriers{0};
};
inline Block* g_block = nullptr;

inline void syncthreads() {
    if (tl_lane == 0 && tl_threadIdx.x + tl_threadIdx.y + tl_threadIdx.z == 0) g_block->barriers++;
    g_block->bar->arrive_and_wait();
}
inline void syncwarp() { g_block->warp_bar[tl_warp]->arrive_and_wait(); }

template <class T>
T shfl_generic(T v, int src_lane) {
    static_assert(sizeof(T) <= 8, "shuffle of at most 8 bytes");
    auto& buf = g_block->shfl[tl_warp];
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(T));
    buf[tl_lane] = bits;
    g_block->warp_bar[tl_warp]->arrive_and_wait();
    T out = v;
    if (src_lane >= 0 && src_lane < 32) {
        uint64_t b = buf[src_lane];
        std::memcpy(&out, &b, sizeof(T));
    }
    g_block->warp_bar[tl_warp]->arrive_and_wait();
    return out;
}

template <class T>
T& touch(T& x, uint8_t space, const char* file, int line);

// ---- warp-level matrix instructions (models of the PTX ISA definitions) ----
//
// ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16: lanes 8i..8i+7 give the
// addresses of the 8 rows (16 bytes each) of matrix i. Lane l receives, in
// register i, two 16-bit elements of matrix i: row l/4, columns 2(l%4) and
// 2(l%4)+1 (low half first). With .trans the matrix is transposed first,
// so the elements are rows 2(l%4) and 2(l%4)+1 of column l/4.
struct alignas(16) Row16 { uint32_t w[4]; };
inline void ldmatrix(uint32_t* r, int nmat, bool trans, const uint16_t* p) {
    auto& buf = g_block->wx[tl_warp];
    uintptr_t v = reinterpret_cast<uintptr_t>(p);
    std::memcpy(&buf[tl_lane][0], &v, sizeof v);
    // traced as one 16-byte shared access per lane: 8 lanes per phase, like the hardware's 8 rows per matrix
    if (tl_lane < unsigned(8 * nmat)) touch(*reinterpret_cast<Row16*>(const_cast<uint16_t*>(p)), 1, "ldmatrix", 0);
    g_block->warp_bar[tl_warp]->arrive_and_wait();
    auto row = [&](unsigned lane) {
        uintptr_t a;
        std::memcpy(&a, &buf[lane][0], sizeof a);
        return reinterpret_cast<const uint16_t*>(a);
    };
    unsigned g = tl_lane / 4, t = tl_lane % 4;
    for (int i = 0; i < nmat; i++) {
        uint16_t lo, hi;
        if (!trans) {
            lo = row(8 * i + g)[2 * t];
            hi = row(8 * i + g)[2 * t + 1];
        } else {
            lo = row(8 * i + 2 * t)[g];
            hi = row(8 * i + 2 * t + 1)[g];
        }
        r[i] = uint32_t(lo) | (uint32_t(hi) << 16);
    }
    g_block->warp_bar[tl_warp]->arrive_and_wait();
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, D = A * B + C with
// A 16x16, B 16x8, C/D 16x8. Fragment layouts (PTX ISA, "Matrix Fragments for
// mma.m16n8k16 with floating point type"), g = lane / 4, t = lane % 4, each
// .b32 register holding two fp16 values, lower column/row index in the low half:
//   A: a0 = (g, 2t..2t+1)  a1 = (g+8, 2t..)  a2 = (g, 2t+8..)  a3 = (g+8, 2t+8..)
//   B: b0 = (k = 2t..2t+1, n = g)            b1 = (k = 2t+8..2t+9, n = g)
//   C, D: c0, c1 = (g, 2t), (g, 2t+1)        c2, c3 = (g+8, 2t), (g+8, 2t+1)
// fp16 x fp16 products are exact in fp32; they are added to c in fp32, in k
// order. The hardware's internal summation order and rounding are not
// specified by NVIDIA, so this is a model; the tests' tolerance covers both.
inline void mma_m16n8k16(float* d, const uint32_t* a, const uint32_t* b) {
    auto& buf = g_block->wx[tl_warp];
    for (int i = 0; i < 4; i++) buf[tl_lane][i] = a[i];
    buf[tl_lane][4] = b[0];
    buf[tl_lane][5] = b[1];
    g_block->warp_bar[tl_warp]->arrive_and_wait();
    float A[16][16], B[16][8];
    for (unsigned l = 0; l < 32; l++) {
        unsigned g = l / 4, t = l % 4;
        auto lo = [](uint32_t v) { return kc::f16_to_f32(uint16_t(v & 0xffffu)); };
        auto hi = [](uint32_t v) { return kc::f16_to_f32(uint16_t(v >> 16)); };
        const auto& f = buf[l];
        A[g][2 * t] = lo(f[0]);         A[g][2 * t + 1] = hi(f[0]);
        A[g + 8][2 * t] = lo(f[1]);     A[g + 8][2 * t + 1] = hi(f[1]);
        A[g][2 * t + 8] = lo(f[2]);     A[g][2 * t + 9] = hi(f[2]);
        A[g + 8][2 * t + 8] = lo(f[3]); A[g + 8][2 * t + 9] = hi(f[3]);
        B[2 * t][g] = lo(f[4]);         B[2 * t + 1][g] = hi(f[4]);
        B[2 * t + 8][g] = lo(f[5]);     B[2 * t + 9][g] = hi(f[5]);
    }
    g_block->warp_bar[tl_warp]->arrive_and_wait();
    unsigned g = tl_lane / 4, t = tl_lane % 4;
    for (int i = 0; i < 4; i++) {
        unsigned r = g + (i >= 2 ? 8 : 0), c = 2 * t + (i & 1);
        float acc = d[i];
        for (int k = 0; k < 16; k++) acc += A[r][k] * B[k][c];
        d[i] = acc;
    }
}

template <class T>
T& touch(T& x, uint8_t space, const char* file, int line) {
    if (tl_log) {
        uint64_t key = (reinterpret_cast<uintptr_t>(file) << 20) ^ uint64_t(line) ^ (uint64_t(space) << 62);
        uint32_t occ = (*tl_occ)[key]++;
        unsigned tid = tl_threadIdx.x + g_blockDim.x * (tl_threadIdx.y + g_blockDim.y * tl_threadIdx.z);
        tl_log->push_back({file, line, occ, reinterpret_cast<uintptr_t>(&x), uint32_t(sizeof(T)), space, tid});
    }
    return x;
}

// shared-memory wavefronts for one warp request: lanes are split into
// phases (32 lanes for 4-byte, 16 for 8-byte, 8 for 16-byte accesses);
// within a phase, the number of distinct 4-byte words that map to the most
// loaded of the 32 banks is the number of wavefronts.
inline uint64_t shared_wavefronts(const std::vector<std::pair<unsigned, const Access*>>& lanes, uint64_t* ideal) {
    uint32_t bytes = lanes.empty() ? 4 : lanes[0].second->bytes;
    unsigned words = bytes >= 4 ? bytes / 4 : 1;
    unsigned per_phase = 32 / (words > 4 ? 4 : words);
    std::map<unsigned, std::vector<const Access*>> phases;
    for (auto& [lane, a] : lanes) phases[lane / per_phase].push_back(a);
    uint64_t total = 0;
    for (auto& [_, v] : phases) {
        std::array<std::set<uintptr_t>, 32> bank;
        for (const Access* a : v)
            for (unsigned w = 0; w < words; w++) {
                uintptr_t word = (a->addr >> 2) + w;
                bank[word % 32].insert(word);
            }
        size_t m = 1;
        for (auto& b : bank) m = std::max(m, b.size());
        total += m;
    }
    *ideal = phases.size();
    return total;
}

inline void analyse_block(std::vector<std::vector<Access>>& logs, unsigned warp_of_tid_div, Trace& t) {
    // group by (warp, space, file, line, occurrence)
    struct Key {
        unsigned warp; uint8_t space; const char* file; int line; uint32_t occ;
        bool operator<(const Key& o) const {
            return std::tie(warp, space, file, line, occ) < std::tie(o.warp, o.space, o.file, o.line, o.occ);
        }
    };
    std::map<Key, std::vector<std::pair<unsigned, const Access*>>> groups;
    for (auto& log : logs)
        for (auto& a : log) groups[{a.tid / warp_of_tid_div, a.space, a.file, a.line, a.occurrence}].push_back({a.tid % 32, &a});
    for (auto& [k, lanes] : groups) {
        auto& site = t.sites[std::string(k.file) + ":" + std::to_string(k.line)];
        site.requests++;
        if (k.space == 0) {
            std::set<uintptr_t> sectors, bytes;
            for (auto& [_, a] : lanes) {
                for (uintptr_t s = a->addr / 32; s <= (a->addr + a->bytes - 1) / 32; s++) sectors.insert(s);
                for (uintptr_t b = a->addr; b < a->addr + a->bytes; b++) bytes.insert(b);
            }
            t.global_requests++;
            t.global_sectors += sectors.size();
            t.global_bytes += bytes.size();
            site.sectors += sectors.size();
            site.bytes += bytes.size();
        } else {
            uint64_t ideal = 0;
            uint64_t w = shared_wavefronts(lanes, &ideal);
            t.shared_requests++;
            t.shared_wavefronts += w;
            t.shared_ideal += ideal;
            site.wavefronts += w;
        }
    }
}

// Persistent worker threads, one per thread of the largest block so far
// (creating OS threads per block dominated the run time). Each worker has
// its own semaphore so a launch wakes exactly the threads it needs.
class Pool {
   public:
    void run(unsigned n, const std::function<void(unsigned)>& f) {
        while (workers_.size() < n) {
            auto w = std::make_unique<Worker>();
            Worker* raw = w.get();
            w->th = std::thread([this, raw, id = unsigned(workers_.size())] {
                for (;;) {
                    raw->go.acquire();
                    if (stop_) return;
                    (*job_)(id);
                    if (remaining_.fetch_sub(1) == 1) done_.release();
                }
            });
            workers_.push_back(std::move(w));
        }
        job_ = &f;
        remaining_ = n;
        for (unsigned i = 0; i < n; i++) workers_[i]->go.release();
        done_.acquire();
    }
    ~Pool() {
        stop_ = true;
        for (auto& w : workers_) w->go.release();
        for (auto& w : workers_) w->th.join();
    }

   private:
    struct Worker {
        std::binary_semaphore go{0};
        std::thread th;
    };
    std::vector<std::unique_ptr<Worker>> workers_;
    const std::function<void(unsigned)>* job_ = nullptr;
    std::atomic<unsigned> remaining_{0};
    std::binary_semaphore done_{0};
    std::atomic<bool> stop_{false};
};
inline Pool g_pool;

template <class Kernel, class... Args>
void launch(dim3 grid, dim3 block, Kernel kernel, Args... args) {
    g_gridDim = grid;
    g_blockDim = block;
    unsigned n = block.x * block.y * block.z;
    unsigned nwarps = (n + 31) / 32;
    for (unsigned bz = 0; bz < grid.z; bz++)
        for (unsigned by = 0; by < grid.y; by++)
            for (unsigned bx = 0; bx < grid.x; bx++) {
                Block b;
                b.bar = std::make_unique<std::barrier<>>(n);
                for (unsigned w = 0; w < nwarps; w++) b.warp_bar.push_back(std::make_unique<std::barrier<>>(std::min(32u, n - 32 * w)));
                b.shfl.resize(nwarps);
                b.wx.resize(nwarps);
                g_block = &b;
                std::vector<std::vector<Access>> logs(g_trace ? n : 0);
                std::function<void(unsigned)> body = [&](unsigned t) {
                        tl_threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
                        tl_blockIdx = dim3(bx, by, bz);
                        tl_warp = t / 32;
                        tl_lane = t % 32;
                        std::unordered_map<uint64_t, uint32_t> occ;
                        if (g_trace) {
                            tl_log = &logs[t];
                            tl_occ = &occ;
                        }
                        kernel(args...);
                        tl_log = nullptr;
                        tl_occ = nullptr;
                };
                g_pool.run(n, body);
                if (g_trace) {
                    Trace t;
                    analyse_block(logs, 32, t);
                    t.blocks = 1;
                    t.threads = n;
                    t.barriers = b.barriers.load();
                    g_trace->add(t);
                }
                g_block = nullptr;
            }
}

}  // namespace kc_emu

#define threadIdx (::kc_emu::tl_threadIdx)
#define blockIdx (::kc_emu::tl_blockIdx)
#define blockDim (::kc_emu::g_blockDim)
#define gridDim (::kc_emu::g_gridDim)
#define __syncthreads() ::kc_emu::syncthreads()
#define __syncwarp(...) ::kc_emu::syncwarp()

template <class T> inline T __shfl_xor_sync(unsigned, T v, int m, int w = 32) {
    int lane = int(::kc_emu::tl_lane);
    int src = lane ^ m;
    if ((src / w) != (lane / w)) src = lane;
    return ::kc_emu::shfl_generic(v, src);
}
template <class T> inline T __shfl_down_sync(unsigned, T v, unsigned d, int w = 32) {
    int lane = int(::kc_emu::tl_lane);
    int src = lane + int(d);
    if ((src / w) != (lane / w)) src = lane;
    return ::kc_emu::shfl_generic(v, src);
}
template <class T> inline T __shfl_sync(unsigned, T v, int s, int w = 32) {
    int lane = int(::kc_emu::tl_lane);
    return ::kc_emu::shfl_generic(v, (lane / w) * w + (s % w));
}
inline float atomicAdd(float* p, float v) { return std::atomic_ref<float>(*p).fetch_add(v); }
inline int atomicAdd(int* p, int v) { return std::atomic_ref<int>(*p).fetch_add(v); }
inline float __expf(float x) { return std::exp(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline int __dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; i++) c += int(int8_t(a >> (8 * i))) * int(int8_t(b >> (8 * i)));
    return c;
}

#define KC_G(x) (::kc_emu::touch((x), 0, __FILE__, __LINE__))
#define KC_S(x) (::kc_emu::touch((x), 1, __FILE__, __LINE__))
#define KC_LAUNCH(kernel, grid, block, ...) ::kc_emu::launch((grid), (block), (kernel), __VA_ARGS__)
