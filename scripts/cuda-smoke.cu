#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
        return 1; \
    } \
} while (0)

__global__ void fill_values(int *values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    values[index] = index * 2;
}

int main() {
    cudaDeviceProp device{};
    CHECK_CUDA(cudaGetDeviceProperties(&device, 0));
    int *values = nullptr;
    int actual[256]{};
    CHECK_CUDA(cudaMalloc(&values, sizeof(actual)));
    fill_values<<<1, 256>>>(values);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(actual, values, sizeof(actual), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(values));
    for (int i = 0; i < 256; ++i) {
        if (actual[i] != i * 2) {
            std::fprintf(stderr, "Incorrect result at index %d\n", i);
            return 1;
        }
    }
    std::printf("PASS: %s, sm_%d%d, 256 GPU results verified\n", device.name, device.major, device.minor);
    return 0;
}
