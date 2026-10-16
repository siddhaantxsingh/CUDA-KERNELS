// Out-of-place transpose of a rows x cols row-major float matrix.
#pragma once
#include "compat.h"

namespace kc {

constexpr int TILE = 32, TROWS = 8;  // 32x32 tile, 32x8 threads, 4 elements each

// Reads are coalesced, writes are strided by `rows` (one sector per element).
__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    int c = blockIdx.x * TILE + threadIdx.x;
    int r0 = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += TROWS) {
        int r = r0 + j;
        if (r < rows && c < cols) KC_G(out[size_t(c) * rows + r]) = KC_G(in[size_t(r) * cols + c]);
    }
}

// Stage through shared memory so both global reads and writes are coalesced.
// PAD = 0 reads a tile column: all 32 lanes hit the same bank (32-way
// conflict). PAD = 1 shifts each row by one bank.
template <int PAD>
__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + PAD];
    int c = blockIdx.x * TILE + threadIdx.x;
    int r = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += TROWS)
        if (r + j < rows && c < cols) KC_S(tile[threadIdx.y + j][threadIdx.x]) = KC_G(in[size_t(r + j) * cols + c]);
    __syncthreads();
    int oc = blockIdx.y * TILE + threadIdx.x;  // column of out = row of in
    int orow = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += TROWS)
        if (orow + j < cols && oc < rows) KC_G(out[size_t(orow + j) * rows + oc]) = KC_S(tile[threadIdx.x][threadIdx.y + j]);
}

inline dim3 transpose_grid(int rows, int cols) { return dim3(ceil_div(cols, TILE), ceil_div(rows, TILE)); }

inline void transpose_naive(const float* in, float* out, int rows, int cols) {
    auto k = transpose_naive_kernel;
    KC_LAUNCH(k, transpose_grid(rows, cols), dim3(TILE, TROWS), in, out, rows, cols);
}
inline void transpose_tiled(const float* in, float* out, int rows, int cols) {
    auto k = transpose_tiled_kernel<0>;
    KC_LAUNCH(k, transpose_grid(rows, cols), dim3(TILE, TROWS), in, out, rows, cols);
}
inline void transpose_padded(const float* in, float* out, int rows, int cols) {
    auto k = transpose_tiled_kernel<1>;
    KC_LAUNCH(k, transpose_grid(rows, cols), dim3(TILE, TROWS), in, out, rows, cols);
}

}  // namespace kc
