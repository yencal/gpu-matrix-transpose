#pragma once

__global__ void TransposeNaive(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        output[x * N + y] = input[y * N + x];
    }
}

template <int TILE_DIM>
__global__ void TransposeBankConflicts(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{ 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float smem[TILE_DIM][TILE_DIM];

    if (x < N && y < N) {
        smem[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        output[y * N + x] = smem[threadIdx.x][threadIdx.y];
    }
}

template <int TILE_DIM>
__global__ void TransposeNoBankConflicts(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float smem[TILE_DIM][TILE_DIM+1];

    if (x < N && y < N) {
        smem[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        output[y * N + x] = smem[threadIdx.x][threadIdx.y];
    }
}

template <int TILE_DIM, typename VecT>
__global__ void TransposeVectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    // NOTE: Assumes N is divisible by sizeof(VecT)/sizeof(float)
    // Caller must ensure this or handle tail elements separately

    static_assert(sizeof(VecT) % sizeof(float) == 0, 
                  "VecT must be a multiple of float size");

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int f = sizeof(VecT) / sizeof(float);
    const int width = N / f; // Width in vector space

    __shared__ VecT smem[TILE_DIM][TILE_DIM + 1];

    if (x < width && y < width) {
        smem[threadIdx.y][threadIdx.x] = reinterpret_cast<const VecT*>(input)[y*width+x];
    }
    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;

    if (x < width && y < width) {
        reinterpret_cast<VecT*>(output)[y*width+x] = smem[threadIdx.x][threadIdx.y];
    }
}