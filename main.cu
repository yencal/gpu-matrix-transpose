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

struct BenchmarkResult {
    std::string label;
    int N;
    dim3 block_dim;
    dim3 grid_dim;
    float latency_ms;
    float bandwidth_GB_per_s;
    float pct_of_peak;
};

struct DeviceInfo {
    std::string name;
    float peak_bandwidth_GBps;
};

DeviceInfo GetDeviceInfo(int device = 0)
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    // Peak BW = memoryClockRate (kHz) * memoryBusWidth (bits) * 2 (DDR) / 8 (bits->bytes) / 1e6 (to GB/s)
    float peak_bw = (prop.memoryClockRate * 1e3f) * (prop.memoryBusWidth / 8) * 2 / 1e9f;

    return {prop.name, peak_bw};
}

void Initialize(
    float* matrix,
    float* gold,
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
    const float* result,
    const float* reference,
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
    const KernelFunc& kernel,
    const dim3 block_dim,
    const dim3 grid_dim,
    const int N,
    const float peak_bandwidth,
    const int num_warmup = 10,
    const int num_iterations = 100)
{
    std::cout << " - Running Benchmark for: " << label << std::endl;
    
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
    kernel<<<grid_dim, block_dim>>>(d_input, d_transposed, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_transposed, d_transposed, size_bytes, cudaMemcpyDeviceToHost));
    ValidateTranspose(h_transposed, h_gold, N);

    // Warmup
    for (int i = 0; i < num_warmup; ++i) {
        kernel<<<grid_dim, block_dim>>>(d_input, d_transposed, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        kernel<<<grid_dim, block_dim>>>(d_input, d_transposed, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / num_iterations;

    // Calculate bandwidth (read + write)
    size_t bytes_transferred = 2 * size_bytes;
    float bandwidth_GB_per_s = (bytes_transferred / (avg_time_ms / 1000.0)) / 1e9;
    float pct_peak = (bandwidth_GB_per_s / peak_bandwidth) * 100.0f;

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
        N,
        block_dim,
        grid_dim,
        avg_time_ms,
        bandwidth_GB_per_s,
        pct_peak
    };
}

void PrintTable(const std::vector<BenchmarkResult>& results)
{
    std::cout << "\n";
    std::cout << std::left << std::setw(30) << "Kernel"
              << std::right << std::setw(12) << "Block"
              << std::setw(12) << "Time(ms)"
              << std::setw(14) << "BW(GB/s)"
              << std::setw(10) << "% Peak"
              << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& r : results) {
        std::string block_str = std::to_string(r.block_dim.x) + "x" + std::to_string(r.block_dim.y);
        std::cout << std::left << std::setw(30) << r.label
                  << std::right << std::setw(12) << block_str
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.latency_ms
                  << std::setw(14) << std::setprecision(1) << r.bandwidth_GB_per_s
                  << std::setw(9) << std::setprecision(1) << r.pct_of_peak << "%"
                  << "\n";
    }
    std::cout << std::string(78, '-') << "\n";
}

void WriteCSV(const std::vector<BenchmarkResult>& results, const std::string& filename)
{
    std::ofstream file(filename);
    file << "Kernel,BlockX,BlockY,LatencyMs,BandwidthGBps,PctPeak\n";
    for (const auto& r : results) {
        file << r.label << ","
             << r.block_dim.x << "," << r.block_dim.y << ","
             << r.latency_ms << ","
             << r.bandwidth_GB_per_s << ","
             << r.pct_of_peak << "\n";
    }
    std::cout << "Results written to " << filename << "\n";
}

int main(int argc, char** argv)
{
    constexpr int N = 32768;
    constexpr int TILE = 32;

    // Get device info
    DeviceInfo device = GetDeviceInfo();

    std::cout << "=== Transpose Benchmark: " << N << "x" << N << " ===" << std::endl;
    std::cout << "Device: " << device.name << std::endl;
    std::cout << "Theoretical Peak BW: " << std::fixed << std::setprecision(1) 
              << device.peak_bandwidth_GBps << " GB/s" << std::endl;
    std::cout << "Matrix size: " << (2.0 * N * N * sizeof(float)) / 1e9 << " GB (read+write)" << std::endl;

    std::vector<BenchmarkResult> results;
    float peak_bw = device.peak_bandwidth_GBps;
    const unsigned int grid_size = (N + TILE - 1) / TILE;

    // === Baseline kernels ===
    results.push_back(RunTest("Naive", TransposeNaive,
        {TILE, TILE}, {grid_size, grid_size}, N, peak_bw));
    results.push_back(RunTest("SMEM+BankConflicts", TransposeBankConflicts<TILE>,
        {TILE, TILE}, {grid_size, grid_size}, N, peak_bw));
    results.push_back(RunTest("SMEM+Padding", TransposeNoBankConflicts<TILE>,
        {TILE, TILE}, {grid_size, grid_size}, N, peak_bw));

    // === Coarsening sweep ===
    results.push_back(RunTest("CBLOCK_ROW_16", TransposeNoBankConflictsCoarsen<TILE, 16>,
        {TILE, 16}, {grid_size, grid_size}, N, peak_bw));

    results.push_back(RunTest("BLOCK_ROW_8", TransposeNoBankConflictsCoarsen<TILE, 8>,
        {TILE, 8}, {grid_size, grid_size}, N, peak_bw));

    results.push_back(RunTest("BLOCK_ROW_4", TransposeNoBankConflictsCoarsen<TILE, 4>,
        {TILE, 4}, {grid_size, grid_size}, N, peak_bw));

    // // Vectorized kernels (only if N is divisible)
    // if (N % 2 == 0) {
    //     BenchmarkConfig config_f2 = config;
    //     config_f2.block_dim = dim3{TILE/2, TILE};
    //     const unsigned int grid_x_f2 = (N/2 + TILE/2 - 1) / (TILE/2);
    //     const unsigned int grid_y_f2 = (N + TILE - 1) / TILE;
    //     config_f2.grid_dim = dim3{grid_x_f2, grid_y_f2};
    //     results.push_back(RunTest("Float2", TransposeVec2<TILE>, config_f2));
    // }

    // if (N % 4 == 0) {
    //     BenchmarkConfig config_f4 = config;
    //     config_f4.block_dim = dim3{TILE/4, TILE};
    //     const unsigned int grid_x_f4 = (N/4 + TILE/4 - 1) / (TILE/4);
    //     const unsigned int grid_y_f4 = (N + TILE - 1) / TILE;
    //     config_f4.grid_dim = dim3{grid_x_f4, grid_y_f4};
    //     results.push_back(RunTest("Float4", TransposeVec4<TILE>, config_f4));
    // }

    // === Print results ===
    PrintTable(results);
    WriteCSV(results, "transpose_comparison.csv");

    return EXIT_SUCCESS;
}