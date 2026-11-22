#include "shape.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" __declspec(dllimport) cudaStream_t prefill_mmq_stream(ggml_backend_t backend);
extern "C" __declspec(dllimport) std::size_t prefill_mmq_packed_bytes();
extern "C" __declspec(dllimport) void prefill_mmq_pack(ggml_backend_t backend, const float * input, void * packed);
extern "C" __declspec(dllimport) void prefill_mmq_launch(ggml_backend_t backend, int width,
    const char * up, const char * gate, const void * packed, float * output);

using nlohmann::json;
using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;
constexpr std::array<int, 3> widths = {128, 64, 32};
constexpr std::array<int, 8> layers = {0, 5, 10, 15, 20, 25, 30, 34};
constexpr int passes = 18;
constexpr std::size_t output_count = token_columns * output_features;

void check_cuda(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(code));
    }
}

DeviceBuffer make_device_buffer(std::size_t bytes) {
    void * pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, bytes));
    return DeviceBuffer(pointer, &cudaFree);
}

DeviceBuffer read_weight(std::ifstream & model, const gguf_context * meta, const std::string & name) {
    const int64_t index = gguf_find_tensor(meta, name.c_str());
    const std::size_t bytes = ggml_row_size(GGML_TYPE_Q4_K, input_features) * output_features;
    if (index < 0 || gguf_get_tensor_type(meta, index) != GGML_TYPE_Q4_K || gguf_get_tensor_size(meta, index) != bytes) {
        throw std::runtime_error("Unexpected GGUF tensor " + name);
    }
    std::vector<char> host(bytes);
    model.seekg(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index));
    if (!model.read(host.data(), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Cannot read GGUF tensor " + name);
    }
    auto device = make_device_buffer(bytes);
    check_cuda(cudaMemcpy(device.get(), host.data(), bytes, cudaMemcpyHostToDevice));
    return device;
}

struct WeightPair {
    DeviceBuffer up;
    DeviceBuffer gate;
};

int main(int argc, char ** argv) try {
    if (argc != 3) {
        throw std::runtime_error("Usage: prefill-mmq-bench model.gguf output.json");
    }
    const std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(argv[1], {true, nullptr}), &gguf_free);
    std::ifstream model(argv[1], std::ios::binary);
    const std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cuda_init(0), &ggml_backend_free);
    if (!meta || !model || !backend) {
        throw std::runtime_error("Cannot open the model or initialize the CUDA backend.");
    }
    std::vector<WeightPair> weight_pairs;
    for (int layer : layers) {
        const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
        weight_pairs.push_back({read_weight(model, meta.get(), prefix + "up.weight"),
                                read_weight(model, meta.get(), prefix + "gate.weight")});
    }

    std::mt19937 random(42);
    std::vector<float> activations(input_features * token_columns);
    for (float & value : activations) {
        value = (static_cast<float>(random() >> 8) / 8388608.0f) - 1.0f;
    }
    auto input = make_device_buffer(activations.size() * sizeof(float));
    auto packed = make_device_buffer(prefill_mmq_packed_bytes());
    auto output = make_device_buffer(output_count * sizeof(float));
    check_cuda(cudaMemcpy(input.get(), activations.data(), activations.size() * sizeof(float), cudaMemcpyHostToDevice));
    const auto stream = prefill_mmq_stream(backend.get());
    check_cuda(cudaMemsetAsync(packed.get(), 0, prefill_mmq_packed_bytes(), stream));
    prefill_mmq_pack(backend.get(), static_cast<const float *>(input.get()), packed.get());
    check_cuda(cudaGetLastError());
    const auto launch = [&](int width, const WeightPair & weights) {
        prefill_mmq_launch(backend.get(), width, static_cast<const char *>(weights.up.get()),
            static_cast<const char *>(weights.gate.get()), packed.get(), static_cast<float *>(output.get()));
        check_cuda(cudaGetLastError());
    };
    const auto get_output = [&](std::vector<float> & host) {
        check_cuda(cudaMemcpyAsync(host.data(), output.get(), output_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
    };

    std::array<double, widths.size()> max_error = {};
    std::array<std::size_t, widths.size()> differing = {};
    std::vector<float> reference(output_count), candidate(output_count);
    for (const auto & pair : weight_pairs) {
        launch(widths.front(), pair);
        get_output(reference);
        for (std::size_t variant = 0; variant < widths.size(); ++variant) {
            launch(widths[variant], pair);
            get_output(candidate);
            for (std::size_t index = 0; index < output_count; ++index) {
                const double error = std::abs(double(reference[index]) - candidate[index]);
                if (!std::isfinite(error) || error > 1e-5 + 1e-6 * std::abs(reference[index])) {
                    throw std::runtime_error("Output mismatch at width " + std::to_string(widths[variant]));
                }
                max_error[variant] = std::max(max_error[variant], error);
                differing[variant] += error != 0;
            }
        }
    }

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    std::array<std::vector<double>, widths.size()> samples;
    for (int pass = 0; pass < passes; ++pass) {
        for (std::size_t position = 0; position < widths.size(); ++position) {
            const std::size_t variant = (pass + position) % widths.size();
            check_cuda(cudaEventRecord(start, stream));
            for (const auto & pair : weight_pairs) {
                launch(widths[variant], pair);
            }
            check_cuda(cudaEventRecord(stop, stream));
            check_cuda(cudaEventSynchronize(stop));
            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, stop));
            samples[variant].push_back(ms / weight_pairs.size());
        }
    }
    check_cuda(cudaEventDestroy(start));
    check_cuda(cudaEventDestroy(stop));

    json rows = json::array();
    for (std::size_t index = 0; index < widths.size(); ++index) {
        auto sorted = samples[index];
        std::sort(sorted.begin(), sorted.end());
        const double median = (sorted[passes / 2 - 1] + sorted[passes / 2]) / 2;
        rows.push_back({{"tile_width", widths[index]}, {"median_ms", median}, {"min_ms", sorted.front()},
            {"max_ms", sorted.back()}, {"sample_ms_per_launch", samples[index]},
            {"differing_values_vs_128", differing[index]}, {"max_abs_error_vs_128", max_error[index]}});
        std::cout << "Tile " << widths[index] << ": " << median << " ms per fused launch\n";
    }
    json result = {{"status", "passed"}, {"input_features", input_features}, {"output_features", output_features},
        {"tokens", token_columns}, {"layers", layers}, {"activation_seed", 42}, {"passes", passes}, {"variants", rows}};
    std::ofstream out(argv[2]);
    if (!(out << result.dump(2) << '\n')) {
        throw std::runtime_error("Cannot save isolated measurements.");
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
