#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <string>

// CUDA_CHECK
#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                    __FILE__, __LINE__, cudaGetErrorString(err_));                  \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// CUDA_CHECK_KERNEL
// reuse CUDA_CEHCK
#define CUDA_CHECK_KERNEL()                  \
    do {                                     \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

__global__ void
saxpy(const float *x, float *y, long long n) {
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += (long long)gridDim.x * blockDim.x) {
        y[idx] = 2.0f * x[idx] + y[idx];
    }
}

int main(int argc, char **argv) {
    // cmdline input handling
    if (argc < 2) {
        fprintf(stderr, "用法：%s <n>\n", argv[0]);
        return 1;
    }
    long long element_count;
    try {
        element_count = std::stoll(argv[1]);
    } catch (const std::exception &) {
        fprintf(stderr, "用法：%s <n>（n 必须是非负整数）\n", argv[0]);
        return 1;
    }
    if (element_count < 0) {
        fprintf(stderr, "输入元素数目应不小于0");
        return 1;
    }
    // n==0 handling
    if (element_count == 0) {
        printf("SUM=0\n");
        return 0;
    }

    // get device prop for blockDim
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    size_t bytes = (size_t)element_count * sizeof(float);

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);

    // generate x and y
    for (long long i = 0; i < element_count; ++i) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = ((i % 1024) - 512) * 1.f;
    }

    // memcpy
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // kernel launch
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int blocksPerGrid = (int)((element_count + threadsPerBlock - 1) /
                              threadsPerBlock);

    GpuTimer timer;
    timer.start();
    saxpy<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, element_count);
    CUDA_CHECK_KERNEL();

    float kernel_ms = timer.stop_ms();

    // move to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // host-side sum
    double s = 0.0;
    for (long long i = 0; i < element_count; ++i) {
        s += h_y[i];
    }

    printf("SUM=%.0f n=%lld kernel_ms=%.3f\n", s, element_count, kernel_ms);
    return 0;
}
