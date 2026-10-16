// Instantiates every kernel for device compilation (tools/device_build.sh).
// The host function is never run.
#include "kc/attention.cuh"
#include "kc/backward.cuh"
#include "kc/gemm.cuh"
#include "kc/hgemm.cuh"
#include "kc/mma.cuh"
#include "kc/reduce.cuh"
#include "kc/softmax.cuh"
#include "kc/transpose.cuh"

void instantiate_all(float* a, float* b, float* c, float* d, float* e) {
    kc::reduce_atomic(a, 1, b);
    kc::reduce_smem(a, 1, b);
    kc::reduce_two_pass(a, 1, b, c);
    kc::transpose_naive(a, b, 1, 1);
    kc::transpose_tiled(a, b, 1, 1);
    kc::transpose_padded(a, b, 1, 1);
    kc::sgemm_naive(1, 1, 1, 1.0f, a, b, 0.0f, c);
    kc::sgemm_smem(1, 1, 1, 1.0f, a, b, 0.0f, c);
    kc::sgemm_reg<false>(1, 1, 1, 1.0f, a, b, 0.0f, c);
    kc::sgemm_reg<true>(1, 1, 1, 1.0f, a, b, 0.0f, c);
    kc::softmax(a, b, 1, 1);
    kc::layernorm<false>(a, b, c, d, 1, 1);
    kc::layernorm<true>(a, b, c, d, 1, 1);
    kc::flash_attention<32>(a, b, c, d, 1, 1, false);
    kc::flash_attention<64>(a, b, c, d, 1, 1, false);
    kc::attention_naive<64>(a, b, c, d, e, 1, 1, false);
    // sm_80+ only: fp16 tensor-core GEMM (both load paths) and the layout probe
    auto* h = reinterpret_cast<const uint16_t*>(a);
    kc::hgemm_mma(1, 1, 1, 1.0f, h, h, 0.0f, c);
    kc::mma_probe(h, h, c);
    kc::softmax_backward(a, b, c, 1, 1);
    kc::layernorm_backward(a, b, c, d, e, e, e, e, 1, 1);
}
