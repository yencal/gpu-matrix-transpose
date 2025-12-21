# GPU Matrix Transpose Benchmark

Benchmark different matrix transpose implementations on NVIDIA GPUs, demonstrating the impact of memory coalescing, bank conflicts, and vectorization.

## Quick Start
```bash
nvcc -O3 -std=c++17 -arch=sm_90 main.cu -o transpose_benchmark
./transpose_benchmark
python3 plot_results.py
```

## What's Being Tested

This benchmark compares 5 transpose variants across matrix sizes from 32×32 to 32768×32768:

**Memory Access Patterns:**
- **Naive**: Direct transpose with uncoalesced writes
- **BankConflicts**: Shared memory with 32-way bank conflicts
- **NoBankConflicts**: Shared memory with +1 padding to avoid conflicts

**Vectorization:**
- **Float2**: 2-element vectorized loads/stores (64-bit transactions)
- **Float4**: 4-element vectorized loads/stores (128-bit transactions)

## NVIDIA GPU Bandwidth Plots

The figures below show measured memory bandwidth on different GPUs. All kernels use 32×32 thread blocks processing 32×32 tiles.

* **H100**
![H100 Bandwidth](figures/bandwidth_plot_h100.png)