#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int threads_per_block = 256;
constexpr int warp_width = 32;
constexpr int warmup_launches = 64;
constexpr int launches_per_sample = 64;
constexpr int samples_per_size = 12;
constexpr std::size_t mib = 1024 * 1024;
constexpr std::array<int, 3> sizes_mib = {64, 256, 512};

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void stream_read(const uint4 *input, unsigned count, unsigned *warp_checksums) {
    const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    uint4 sum{};
    for (unsigned i = thread; i < count; i += stride) {
        const uint4 value = __ldcg(input + i);
        sum.x ^= value.x;
        sum.y ^= value.y;
        sum.z ^= value.z;
        sum.w ^= value.w;
    }
    unsigned checksum = sum.x ^ sum.y ^ sum.z ^ sum.w;
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        checksum ^= __shfl_down_sync(0xffffffff, checksum, offset);
    }
    if (threadIdx.x % warp_width == 0) {
        warp_checksums[thread / warp_width] = checksum;
    }
}

int main() {
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    const int blocks = device.multiProcessorCount * 16;
    std::vector<unsigned> checksums(blocks * threads_per_block / warp_width);
    const std::size_t max_bytes = sizes_mib.back() * mib;
    std::vector<uint4> input(max_bytes / sizeof(uint4));
    std::mt19937 random(42);
    unsigned checksum = 0;
    std::array<unsigned, sizes_mib.size()> expected{};
    std::size_t size_index = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto &value = input[i];
        value.x = random();
        value.y = random();
        value.z = random();
        value.w = random();
        checksum ^= value.x ^ value.y ^ value.z ^ value.w;
        if ((i + 1) * sizeof(uint4) == sizes_mib[size_index] * mib) {
            expected[size_index++] = checksum;
        }
    }

    uint4 *device_input = nullptr;
    unsigned *device_checksums = nullptr;
    check(cudaMalloc(&device_input, max_bytes));
    const std::size_t checksum_bytes = checksums.size() * sizeof(unsigned);
    check(cudaMalloc(&device_checksums, checksum_bytes));
    check(cudaMemcpy(device_input, input.data(), max_bytes, cudaMemcpyHostToDevice));
    cudaEvent_t begin{}, end{};
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));

    std::puts("order,size_mib,sample,launches,batch_ms,read_gb_s");
    for (bool reverse : {false, true}) {
        for (std::size_t step = 0; step < sizes_mib.size(); ++step) {
            const std::size_t index = reverse ? sizes_mib.size() - 1 - step : step;
            const std::size_t bytes = sizes_mib[index] * mib;
            const auto count = static_cast<unsigned>(bytes / sizeof(uint4));
            auto launch = [&] { stream_read<<<blocks, threads_per_block>>>(device_input, count, device_checksums); };
            for (int i = 0; i < warmup_launches; ++i) {
                launch();
            }
            check(cudaGetLastError());
            check(cudaMemcpy(checksums.data(), device_checksums, checksum_bytes, cudaMemcpyDeviceToHost));
            unsigned actual = 0;
            for (const unsigned partial : checksums) {
                actual ^= partial;
            }
            if (actual != expected[index]) {
                std::fprintf(stderr, "Checksum failed for %d MiB\n", sizes_mib[index]);
                return EXIT_FAILURE;
            }

            for (int sample = 1; sample <= samples_per_size; ++sample) {
                check(cudaEventRecord(begin));
                for (int i = 0; i < launches_per_sample; ++i) {
                    launch();
                }
                check(cudaGetLastError());
                check(cudaEventRecord(end));
                check(cudaEventSynchronize(end));
                float elapsed_ms = 0;
                check(cudaEventElapsedTime(&elapsed_ms, begin, end));
                const double read_gb_s = double(bytes) * launches_per_sample / (elapsed_ms * 1e6);
                std::printf("%s,%d,%d,%d,%.6f,%.6f\n",
                            reverse ? "reverse" : "forward", sizes_mib[index], sample,
                            launches_per_sample, elapsed_ms, read_gb_s);
            }
        }
    }

    check(cudaEventDestroy(end));
    check(cudaEventDestroy(begin));
    check(cudaFree(device_checksums));
    check(cudaFree(device_input));
}
