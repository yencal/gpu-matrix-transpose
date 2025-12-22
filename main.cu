#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include "transpose_kernels.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct BenchmarkConfig {
    size_t N;
    dim3 block_dim;
    dim3 grid_dim;
    int num_warmup;
    int num_iterations;
};

struct BenchmarkResult {
    std::string label;
    size_t N;
    dim3 block_dim;
    dim3 grid_dim;
    float latency_ms;
    float bandwidth_GB_per_s;
};

void WriteCSV(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
    std::ofstream file(filename);
    file << "Label,N,BlockSize,GridSize,LatencyMs,BandwidthGBps\n";
    for (const auto& r : results) {
        file << r.label << ","
             << r.N << ","
             << r.block_dim.x << "x" << r.block_dim.y << ","
             << r.grid_dim.x << "x" << r.grid_dim.y << ","
             << r.latency_ms << ","
             << r.bandwidth_GB_per_s << "\n";
    }
    file.close();
    std::cout << "Results written to " << filename << std::endl;
}

void Initialize(
    float* __restrict__ matrix,
    float* __restrict__ gold,
    int N)
{
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            matrix[row*N + col] = row*N + col;
        }
    }

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            gold[row*N + col] = matrix[col*N + row];
        }
    }
}

void ValidateTranspose(
    const float* __restrict__ result,
    const float* __restrict__ reference,
    int N)
{
    for (size_t i = 0; i < N*N; ++i) {
        if (result[i] != reference[i]) {
            fprintf(stderr, "FAILED at index %zu: result=%f, expected=%f\n",
                    i, result[i], reference[i]);
            exit(EXIT_FAILURE);
        }
    }
}

template<typename KernelFunc>
BenchmarkResult RunTest(
    const std::string& label,
    KernelFunc kernel,
    const BenchmarkConfig& config)
{
    int N = config.N;
    size_t size_bytes = N * N * sizeof(float);

    // Host allocations
    float* h_input = new float[N*N];
    float* h_transposed = new float[N*N];
    float* h_gold = new float[N*N];
    Initialize(h_input, h_gold, N);

    // Device allocations
    float* d_input;
    float* d_transposed;
    CHECK_CUDA(cudaMalloc(&d_input, size_bytes));
    CHECK_CUDA(cudaMalloc(&d_transposed, size_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size_bytes, cudaMemcpyHostToDevice));

    // Validate first (fail fast)
    kernel<<<config.grid_dim, config.block_dim>>>(d_input, d_transposed, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_transposed, d_transposed, size_bytes, cudaMemcpyDeviceToHost));
    ValidateTranspose(h_transposed, h_gold, N);
    std::cout << "  ✓ " << label << " validation passed\n";

    // Warmup
    for (int i = 0; i < config.num_warmup; ++i) {
        kernel<<<config.grid_dim, config.block_dim>>>(d_input, d_transposed, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < config.num_iterations; ++i) {
        kernel<<<config.grid_dim, config.block_dim>>>(d_input, d_transposed, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / config.num_iterations;

    // Calculate bandwidth (read + write)
    size_t bytes_transferred = 2 * size_bytes;
    float bandwidth_GB_per_s = (bytes_transferred / (avg_time_ms / 1000.0)) / 1e9;

    std::cout << "  " << label << ": " << avg_time_ms << " ms, " 
              << bandwidth_GB_per_s << " GB/s\n";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    delete[] h_input;
    delete[] h_transposed;
    delete[] h_gold;
    cudaFree(d_input);
    cudaFree(d_transposed);
    
    return BenchmarkResult{
        label,
        config.N,
        config.block_dim,
        config.grid_dim,
        avg_time_ms,
        bandwidth_GB_per_s
    };
}

int main(int argc, char** argv)
{
    constexpr int TILE = 32;
    const dim3 BLOCK_DIM{TILE, TILE};
    const int NUM_WARMUP = 10;
    const int NUM_ITERATIONS = 100;
    const std::string CSV_FILENAME = "benchmark_results.csv";

    std::vector<BenchmarkResult> results;

    // for (int power = 5; power <= 15; ++power)
    for (int power = 5; power < 6; ++power)
    {
        size_t N = 1 << power;
        std::cout << "\n=== Testing " << N << "x" << N << " matrix ===\n";
     
        dim3 grid_dim{(N + TILE - 1) / TILE, (N + TILE - 1) / TILE};
        BenchmarkConfig config{N, BLOCK_DIM, grid_dim, NUM_WARMUP, NUM_ITERATIONS};

        // Scalar transpose kernels
        results.push_back(RunTest("Naive", TransposeNaive, config));
        results.push_back(RunTest("BankConflicts", TransposeBankConflicts<TILE>, config));
        results.push_back(RunTest("NoBankConflicts", TransposeNoBankConflicts<TILE>, config));

        // Vectorized kernels (only if N is divisible)
        if (N % 2 == 0) {
            BenchmarkConfig config_f2 = config;
            config_f2.block_dim = dim3{TILE/2, TILE};
            config_f2.grid_dim = dim3{(N/2 + TILE/2 - 1) / (TILE/2), (N + TILE - 1) / TILE};
            results.push_back(RunTest("Float2", TransposeVec2<TILE>, config_f2));
        }

        if (N % 4 == 0) {
            BenchmarkConfig config_f4 = config;
            config_f4.block_dim = dim3{TILE/4, TILE};
            config_f4.grid_dim = dim3{(N/4 + TILE/4 - 1) / (TILE/4), (N + TILE - 1) / TILE};
            results.push_back(RunTest("Float4", TransposeVec4<TILE>, config_f4));
        }
    }

    WriteCSV(results, CSV_FILENAME);
    std::cout << "\nBenchmark complete!\n";

    return EXIT_SUCCESS;
}