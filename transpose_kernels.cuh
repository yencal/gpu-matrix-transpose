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
    // CRITICAL: Block size is (32, 8) not (8, 32)
    // Now warps are contiguous: warp 0 is threadIdx.y=0, threadIdx.x=0-31
    
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int width = N / 4;
    
    // Load phase - each thread loads float4 vertically
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_base = blockIdx.y * (TILE_DIM / 4) + threadIdx.y;
    
    if (x < N && y_base < width) {
        // Load 4 consecutive rows from same column
        float4 data = reinterpret_cast<const float4*>(input)[y_base * N + x];
        int base_row = threadIdx.y * 4;
        tile[base_row + 0][threadIdx.x] = data.x;
        tile[base_row + 1][threadIdx.x] = data.y;
        tile[base_row + 2][threadIdx.x] = data.z;
        tile[base_row + 3][threadIdx.x] = data.w;
    }
    __syncthreads();
    
    // Store phase - read horizontally, write vertically
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y_base = blockIdx.x * (TILE_DIM / 4) + threadIdx.y;
    
    if (x < N && y_base < width) {
        float4 data;
        int base_col = threadIdx.y * 4;
        data.x = tile[threadIdx.x][base_col + 0];
        data.y = tile[threadIdx.x][base_col + 1];
        data.z = tile[threadIdx.x][base_col + 2];
        data.w = tile[threadIdx.x][base_col + 3];
        reinterpret_cast<float4*>(output)[y_base * N + x] = data;
    }
}