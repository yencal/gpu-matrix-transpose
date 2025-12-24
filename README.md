# GPU Matrix Transpose Benchmark

Systematic exploration of matrix transpose optimizations on NVIDIA GPUs, from naive implementation to state-of-the-art techniques achieving 87.7% of theoretical peak bandwidth.

## Quick Start
```bash
nvcc -O3 -std=c++17 -arch=sm_90 main.cu -o transpose_benchmark
./transpose_benchmark
```

## Results

Benchmarked on NVIDIA H100 80GB HBM3 with 32768×32768 matrix (8.6 GB read+write):
```
Kernel                               Block    Time(ms)      BW(GB/s)    % Peak
------------------------------------------------------------------------------
Naive                                32x32     18.0645         475.5     14.2%
Tiled                                32x32      8.6235         996.1     29.7%
TiledPadded                          32x32      5.0724        1693.4     50.5%
Coarsen2                             32x16      3.6014        2385.1     71.1%
Coarsen4                              32x8      3.0984        2772.4     82.7%
Coarsen8                              32x4      3.0951        2775.3     82.8%
Vec2                                 16x32      3.6041        2383.4     71.1%
Vec2Coarsen2                         16x16      3.0983        2772.5     82.7%
Vec2Coarsen4                          16x8      3.0953        2775.1     82.8%
Vec2Coarsen8                          16x4      3.0951        2775.4     82.8%
Vec4                                  8x32      3.0997        2771.2     82.7%
Vec4Coarsen2                          8x16      3.0966        2774.0     82.7%
Vec4Coarsen4                           8x8      3.0960        2774.5     82.8%
Vec4Coarsen8                           8x4      3.0972        2773.4     82.7%
Vec2_T64                             32x32      2.9910        2872.0     85.7%
Vec2_T64_Coarsen2                    32x16      2.9220        2939.8     87.7%
Vec2_T64_Coarsen4                     32x8      2.9345        2927.2     87.3%
Vec4_T64                             16x32      2.9228        2938.9     87.7%
Vec4_T64_Coarsen2                    16x16      2.9347        2927.0     87.3%
Vec4_T64_Coarsen4                     16x8      2.9213        2940.5     87.7%
------------------------------------------------------------------------------
Theoretical Peak: 3352.3 GB/s
Best Achieved: 2940.5 GB/s (87.7%)
```

## Optimization Progression

**Naive (14.2%)**: Direct transpose with coalesced reads but strided writes. Demonstrates baseline unoptimized performance.

**Tiled (29.7%)**: Uses 32×32 shared memory tiles to enable coalesced reads and writes. Suffers from 32-way bank conflicts on column access.

**TiledPadded (50.5%)**: Adds +1 padding to shared memory (33 columns instead of 32) to eliminate bank conflicts. First practical implementation.

**Coarsen4 (82.7%)**: Each thread processes 4 rows using a 32×8 thread block with loop. Amortizes index calculations and improves occupancy on modern GPUs.

**Vec4Coarsen4 (82.8%)**: Combines vectorized memory access (float4) with coarsening. Achieves 128-byte transactions on both reads and writes.

**Vec2_T64_Coarsen2 (87.7%)**: Uses 64×64 tiles with 32×16 blocks and float2 vectorization. Larger tiles reduce per-tile overhead. Best performer.

## Implementation Details

All kernels use the double-swap technique for transpose:
1. Swap shared memory indices: `tile[ty][tx]` → `tile[tx][ty]`
2. Swap block coordinates: `blockIdx.x, blockIdx.y` → `blockIdx.y, blockIdx.x`

Vectorized kernels unpack vectors to shared memory during load, then gather column values and repack during store. This enables vectorized global transactions while working with transposed data in shared memory.