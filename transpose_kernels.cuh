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

template <int TILE_DIM>
__global__ void TransposeVec2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{   
    // Block size: (TILE_DIM/2, TILE_DIM) = (16, 32) for TILE=32
    // Each thread handles one float2 = 2 consecutive floats

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Load phase - vectorized read, unpack to shared memory
    int x = blockIdx.x * (TILE_DIM/2) + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int width = N / 2;
    
    if (x < width && y < N) {
        float2 data = reinterpret_cast<const float2*>(input)[y * width + x];
        tile[threadIdx.y][threadIdx.x * 2 + 0] = data.x;
        tile[threadIdx.y][threadIdx.x * 2 + 1] = data.y;
    }
    __syncthreads();
    
    // Store phase - gather from shared memory, vectorized write
    x = blockIdx.y * (TILE_DIM/2) + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < width && y < N) {
        float2 data;
        data.x = tile[threadIdx.x * 2 + 0][threadIdx.y];
        data.y = tile[threadIdx.x * 2 + 1][threadIdx.y];
        reinterpret_cast<float2*>(output)[y * width + x] = data;
    }
}

template <int TILE_DIM>
__global__ void TransposeVec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{   
    // Block size: (TILE_DIM/4, TILE_DIM) = (8, 32) for TILE=32
    // Each thread handles one float4 = 4 consecutive floats

    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Load phase - vectorized read, unpack to shared memory
    int x = blockIdx.x * (TILE_DIM/4) + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int width = N / 4;
    
    if (x < width && y < N) {
        float4 data = reinterpret_cast<const float4*>(input)[y * width + x];
        tile[threadIdx.y][threadIdx.x * 4 + 0] = data.x;
        tile[threadIdx.y][threadIdx.x * 4 + 1] = data.y;
        tile[threadIdx.y][threadIdx.x * 4 + 2] = data.z;
        tile[threadIdx.y][threadIdx.x * 4 + 3] = data.w;
    }
    __syncthreads();
    
    // Store phase - gather from shared memory, vectorized write
    x = blockIdx.y * (TILE_DIM/4) + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < width && y < N) {
        float4 data;
        data.x = tile[threadIdx.x * 4 + 0][threadIdx.y];
        data.y = tile[threadIdx.x * 4 + 1][threadIdx.y];
        data.z = tile[threadIdx.x * 4 + 2][threadIdx.y];
        data.w = tile[threadIdx.x * 4 + 3][threadIdx.y];
        reinterpret_cast<float4*>(output)[y * width + x] = data;
    }
}

template <int TILE_DIM>
__global__ void TransposeVec2_1DSmem(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    constexpr int STRIDE = TILE_DIM + 1;  // Bank conflict padding
    __shared__ float tile[TILE_DIM * STRIDE];
    
    int x = blockIdx.x * (TILE_DIM / 2) + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int width = N / 2;
    
    // Load: vectorized global read → vectorized smem store
    if (x < width && y < N) {
        float2 data = reinterpret_cast<const float2*>(input)[y * width + x];
        
        // Single 8-byte store to smem (addresses are consecutive)
        int smem_idx = threadIdx.y * STRIDE + threadIdx.x * 2;
        reinterpret_cast<float2*>(&tile[smem_idx])[0] = data;  // STS.64
    }
    __syncthreads();
    
    // Store: strided smem reads (can't vectorize) → vectorized global write
    x = blockIdx.y * (TILE_DIM / 2) + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < width && y < N) {
        float2 data;
        // Two separate loads - stride apart, unavoidable
        data.x = tile[(threadIdx.x * 2 + 0) * STRIDE + threadIdx.y];  // LDS.32
        data.y = tile[(threadIdx.x * 2 + 1) * STRIDE + threadIdx.y];  // LDS.32
        
        reinterpret_cast<float2*>(output)[y * width + x] = data;
    }
}

template <int TILE_DIM>
__global__ void TransposeVec4_1DSmem(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    constexpr int STRIDE = TILE_DIM + 4;  // Padding for bank conflicts
    __shared__ float tile[TILE_DIM * STRIDE];
    
    int x = blockIdx.x * (TILE_DIM / 4) + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    const int width = N / 4;
    
    if (x < width && y < N) {
        float4 data = reinterpret_cast<const float4*>(input)[y * width + x];
        
        // Single 16-byte store instead of 4× 4-byte
        int smem_idx = threadIdx.y * STRIDE + threadIdx.x * 4;
        reinterpret_cast<float4*>(&tile[smem_idx])[0] = data;  // STS.128
    }
    __syncthreads();
    
    x = blockIdx.y * (TILE_DIM / 4) + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < width && y < N) {
        float4 data;
        int base = threadIdx.x * 4;
        data.x = tile[(base + 0) * STRIDE + threadIdx.y];
        data.y = tile[(base + 1) * STRIDE + threadIdx.y];
        data.z = tile[(base + 2) * STRIDE + threadIdx.y];
        data.w = tile[(base + 3) * STRIDE + threadIdx.y];
        
        reinterpret_cast<float4*>(output)[y * width + x] = data;
    }
}