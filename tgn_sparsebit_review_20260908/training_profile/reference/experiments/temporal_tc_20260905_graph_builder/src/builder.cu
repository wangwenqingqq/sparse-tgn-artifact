// Exact bounded symmetric multiplicity histogram -> sorted CSR and bitmap.
#include <cuda_runtime.h>
#include <stdint.h>
using I = int64_t;

__global__ void gb_scatter(int m, int stride, const I* edges, int* hist) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < m) {
        int u = int(edges[k]), v = int(edges[m + k]);
        atomicAdd(hist + u * stride + v, 1);
        atomicAdd(hist + v * stride + u, 1);  // Self-loops count twice.
    }
}

__global__ void gb_count_bits(int n, int stride, const int* hist,
                               int* counts, uint32_t* bits) {
    __shared__ int warp_counts[8];
    int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    int count = 0, words = (n + 31) / 32;
    for (int tile = 0; tile < n; tile += 256) {
        int z = tile + t;
        bool nonzero = z < n && hist[r * stride + z] != 0;
        uint32_t mask = __ballot_sync(0xffffffff, nonzero);
        if (lane == 0 && z < n) bits[r * words + z / 32] = mask;
        count += int(nonzero);
    }
    for (int delta = 16; delta; delta /= 2)
        count += __shfl_down_sync(0xffffffff, count, delta);
    if (lane == 0) warp_counts[warp] = count;
    __syncthreads();
    if (t == 0) {
        int total = 0;
        for (int w = 0; w < 8; ++w) total += warp_counts[w];
        counts[r] = total;
    }
}

__global__ void gb_scan(int n, const int* counts, I* rp) {
    __shared__ int sums[8], prefix[8], carry;
    int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (t == 0) { carry = 0; rp[0] = 0; }
    __syncthreads();
    for (int tile = 0; tile < n; tile += 256) {
        int r = tile + t, value = r < n ? counts[r] : 0;
        for (int delta = 1; delta < 32; delta *= 2) {
            int previous = __shfl_up_sync(0xffffffff, value, delta);
            if (lane >= delta) value += previous;
        }
        if (lane == 31) sums[warp] = value;
        __syncthreads();
        if (t == 0) {
            int total = carry;
            for (int w = 0; w < 8; ++w) { prefix[w] = total; total += sums[w]; }
            carry = total;
        }
        __syncthreads();
        if (r < n) rp[r + 1] = I(prefix[warp] + value);
        __syncthreads();  // Prefix reads finish before the next tile overwrites it.
    }
}

__global__ void gb_emit(int n, int stride, const int* hist, const I* rp,
                         I* col, I* values) {
    __shared__ int sums[8], prefix[8], carry;
    int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    I row_base = rp[r];
    if (t == 0) carry = 0;
    __syncthreads();
    for (int tile = 0; tile < n; tile += 256) {
        int z = tile + t, value = z < n ? hist[r * stride + z] : 0;
        uint32_t mask = __ballot_sync(0xffffffff, value != 0);
        if (lane == 0) sums[warp] = __popc(mask);
        __syncthreads();
        if (t == 0) {
            int total = carry;
            for (int w = 0; w < 8; ++w) { prefix[w] = total; total += sums[w]; }
            carry = total;
        }
        __syncthreads();
        if (value != 0) {
            int rank = __popc(mask & ((1u << lane) - 1u));
            I at = row_base + prefix[warp] + rank;
            col[at] = z;
            values[at] = value;
        }
        __syncthreads();
    }
}

extern "C" int gb_build(int n, int m, const I* edges, int* hist, int* counts,
                         I* rp, I* col, I* values, uint32_t* bits, void* stream) {
    if (n < 1 || n > 2048 || m < 0 || m > 1000000) return int(cudaErrorInvalidValue);
    int stride = ((n + 31) / 32) * 32;
    cudaStream_t st = (cudaStream_t)stream;
    cudaError_t error;
    if (m) {
        gb_scatter<<<(m + 255) / 256, 256, 0, st>>>(m, stride, edges, hist);
        if ((error = cudaGetLastError()) != cudaSuccess) return int(error);
    }
    gb_count_bits<<<n, 256, 0, st>>>(n, stride, hist, counts, bits);
    if ((error = cudaGetLastError()) != cudaSuccess) return int(error);
    gb_scan<<<1, 256, 0, st>>>(n, counts, rp);
    if ((error = cudaGetLastError()) != cudaSuccess) return int(error);
    gb_emit<<<n, 256, 0, st>>>(n, stride, hist, rp, col, values);
    return int(cudaGetLastError());
}
