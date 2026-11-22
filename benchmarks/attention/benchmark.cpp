#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cupti_range_profiler_util.h>
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

using nlohmann::json;

extern "C" __declspec(dllimport) cudaStream_t attention_bench_stream(ggml_backend_t backend);
extern "C" __declspec(dllimport) const char * attention_bench_path(ggml_tensor * output);
extern "C" __declspec(dllimport) void attention_bench_launch(ggml_backend_t backend, ggml_tensor * output);

constexpr int head_dim = 128;
constexpr int query_heads = 32;
constexpr int kv_heads = 8;
constexpr int layers = 8;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

std::vector<float> attention_reference(int depth, const std::vector<float> & query,
                                       const std::vector<ggml_fp16_t> & key,
                                       const std::vector<ggml_fp16_t> & value) {
    std::vector<float> output(query.size());
    std::vector<double> scores(depth);
    for (int head = 0; head < query_heads; ++head) {
        const auto group = head / (query_heads / kv_heads);
        for (int i = 0; i < depth; ++i) {
            double dot = 0;
            const auto offset = (i * kv_heads + group) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                dot += double(query[head * head_dim + d]) * ggml_fp16_to_fp32(key[offset + d]);
            }
            scores[i] = dot / std::sqrt(double(head_dim));
        }
        const auto shift = *std::max_element(scores.begin(), scores.end());
        std::array<double, head_dim> sum{};
        double normalization = 0;
        for (int i = 0; i < depth; ++i) {
            const auto weight = std::exp(scores[i] - shift);
            const auto offset = (i * kv_heads + group) * head_dim;
            normalization += weight;
            for (int d = 0; d < head_dim; ++d) {
                sum[d] += weight * ggml_fp16_to_fp32(value[offset + d]);
            }
        }
        for (int d = 0; d < head_dim; ++d) {
            output[head * head_dim + d] = float(sum[d] / normalization);
        }
    }
    return output;
}

json profile_traffic(ggml_backend_t backend, cudaStream_t stream, const std::array<ggml_tensor *, layers> & outputs) {
    using namespace cupti::utils;
    CUcontext context = nullptr;
    DRIVER_API_CALL(cuCtxGetCurrent(&context));
    CUPTI_API_CALL(cuptiInitializeProfiler());
    {
        RangeProfiler access_check(context);
        CUPTI_API_CALL(access_check.EnableRangeProfiler());
        CUPTI_API_CALL(access_check.DisableRangeProfiler());
    }
    MetricEvaluator evaluator(context);
    std::vector<std::string> metrics{
        "dram__bytes_read.sum", "dram__bytes_write.sum",
        "lts__t_sectors_op_read.sum", "lts__t_sector_hit_rate.pct",
    };
    json trials = json::array();
    for (const bool rotate : {true, false}) {
        const char * name = rotate ? "after_layer_rotation" : "after_same_layer";
        RangeProfiler profiler(context);
        CUPTI_API_CALL(profiler.EnableRangeProfiler(CUPTI_CACHE_CONTROL_MODE_NONE));
        std::vector<uint8_t> counter_data;
        CUPTI_API_CALL(profiler.SetConfig(CUPTI_UserRange, CUPTI_UserReplay, metrics, counter_data, 1));
        std::vector<uint8_t> partition;
        int passes = 0;
        do {
            for (int layer = 1; layer < layers; ++layer) {
                attention_bench_launch(backend, outputs[rotate ? layer : 0]);
            }
            cuda_check(cudaStreamSynchronize(stream));
            CUPTI_API_CALL(profiler.StartRangeProfiler());
            if (passes == 0) {
                CUPTI_API_CALL(profiler.GetDevicePartitionInfo(partition));
            }
            CUPTI_API_CALL(profiler.PushRange(name));
            attention_bench_launch(backend, outputs.front());
            cuda_check(cudaStreamSynchronize(stream));
            CUPTI_API_CALL(profiler.PopRange());
            CUPTI_API_CALL(profiler.StopRangeProfiler());
            if (++passes > 32) {
                throw std::runtime_error("CUPTI did not finish replaying the attention range.");
            }
        } while (!profiler.GetPassState().isAllPassSubmitted);
        CUPTI_API_CALL(profiler.DecodeCounterData());
        CUPTI_API_CALL(profiler.DisableRangeProfiler());
        evaluator.setDevicePartitionInfo(partition);
        std::vector<MetricEvaluator::RangeInfo> ranges;
        evaluator.evaluateAllRanges(counter_data, metrics, ranges);
        if (ranges.size() != 1 || ranges.front().rangeName != name) {
            throw std::runtime_error("Unexpected CUPTI attention range.");
        }
        json counters = json::object();
        for (const auto & metric : ranges.front().metricAndValues) {
            if (!std::isfinite(metric.value) || metric.value < 0) {
                throw std::runtime_error("Invalid CUPTI metric: " + metric.metricName);
            }
            counters[metric.metricName] = metric.value;
        }
        if (rotate && counters.at("dram__bytes_read.sum").get<double>() <= 0) {
            throw std::runtime_error("CUPTI reported zero DRAM reads for the rotated attention range.");
        }
        trials.push_back({{"lead_in", name}, {"passes", passes}, {"counters", counters},
                          {"l2_read_sector_bytes", 32 * counters.at("lts__t_sectors_op_read.sum").get<double>()}});
    }
    CUPTI_API_CALL(cuptiDeinitializeProfiler());
    return trials;
}

json run(int depth, const std::string & mode, bool counters) {
    cuda_check(cudaSetDevice(0));
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cuda_init(0), ggml_backend_free);
    if (!backend) {
        throw std::runtime_error("CUDA backend initialization failed.");
    }
    const auto stream = attention_bench_stream(backend.get());
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({(6 * layers + 1) * ggml_tensor_overhead() + 16384, nullptr, true}), ggml_free);
    if (!ctx) {
        throw std::runtime_error("Attention tensor context initialization failed.");
    }

    auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, depth, 1, 1, 1);
    std::array<ggml_tensor *, layers> queries{}, keys{}, values{}, outputs{};
    for (int i = 0; i < layers; ++i) {
        queries[i] = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, 1, query_heads, 1);
        keys[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, head_dim * kv_heads, depth);
        values[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, head_dim * kv_heads, depth);
        // llama.cpp stores each token's KV heads together, then presents permuted views to Flash Attention.
        const auto view = [&](ggml_tensor * tensor) {
            return ggml_view_4d(ctx.get(), tensor, head_dim, depth, kv_heads, 1,
                                tensor->nb[1], head_dim * sizeof(ggml_fp16_t), ggml_nbytes(tensor), 0);
        };
        outputs[i] = ggml_flash_attn_ext(ctx.get(), queries[i], view(keys[i]), view(values[i]), mask,
                                       1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
        ggml_prec_set_acc(outputs[i], GGML_PREC_F32);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer || !ggml_backend_supports_op(backend.get(), outputs.front())) {
        throw std::runtime_error("CUDA could not allocate or execute the attention tensors.");
    }
    if (mode != attention_bench_path(outputs.front())) {
        throw std::runtime_error("Attention dispatcher selected an unexpected path.");
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 0.8f);
    std::vector<float> query(query_heads * head_dim);
    std::vector<ggml_fp16_t> key(kv_heads * depth * head_dim), value(key.size()), zeros(depth, 0);
    for (auto & item : query) item = normal(rng);
    for (auto & item : key) item = ggml_fp32_to_fp16(normal(rng));
    for (auto & item : value) item = ggml_fp32_to_fp16(normal(rng));
    for (int i = 0; i < layers; ++i) {
        ggml_backend_tensor_set(queries[i], query.data(), 0, ggml_nbytes(queries[i]));
        ggml_backend_tensor_set(keys[i], key.data(), 0, ggml_nbytes(keys[i]));
        ggml_backend_tensor_set(values[i], value.data(), 0, ggml_nbytes(values[i]));
    }
    ggml_backend_tensor_set(mask, zeros.data(), 0, ggml_nbytes(mask));

    attention_bench_launch(backend.get(), outputs.front());
    cuda_check(cudaStreamSynchronize(stream));
    std::vector<float> actual(query.size());
    ggml_backend_tensor_get(outputs.front(), actual.data(), 0, ggml_nbytes(outputs.front()));
    const auto reference = attention_reference(depth, query, key, value);
    double max_error = 0, error_sq = 0, reference_sq = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            throw std::runtime_error("Attention produced a non-finite output.");
        }
        const double error = actual[i] - reference[i];
        max_error = std::max(max_error, std::abs(error));
        error_sq += error * error;
        reference_sq += double(reference[i]) * reference[i];
    }
    const double relative_rms = std::sqrt(error_sq / reference_sq);
    if (max_error > 0.002 || relative_rms > 0.03) {
        throw std::runtime_error("Attention disagreed with the CPU reference. Max error " +
                                 std::to_string(max_error) + "; relative RMS " + std::to_string(relative_rms));
    }
    json result = {
        {"context", depth}, {"path", mode}, {"logical_kv_bytes", (ggml_nbytes(keys.front()) + ggml_nbytes(values.front()))},
        {"distinct_kv_pairs", layers}, {"max_abs_error", max_error}, {"relative_rms_error", relative_rms},
    };
    if (counters) {
        result["traffic"] = profile_traffic(backend.get(), stream, outputs);
        return result;
    }

    cudaEvent_t begin, end;
    cuda_check(cudaEventCreate(&begin));
    cuda_check(cudaEventCreate(&end));
    constexpr int calls = 64;
    for (const bool rotate : {true, false}) {
        const auto launch = [&] {
            for (int j = 0; j < calls; ++j) {
                attention_bench_launch(backend.get(), outputs[rotate ? j % layers : 0]);
            }
        };
        launch();
        std::vector<double> samples;
        for (int i = 0; i < 12; ++i) {
            cuda_check(cudaEventRecord(begin, stream));
            launch();
            cuda_check(cudaEventRecord(end, stream));
            cuda_check(cudaEventSynchronize(end));
            float elapsed = 0;
            cuda_check(cudaEventElapsedTime(&elapsed, begin, end));
            samples.push_back(1000.0 * elapsed / calls);
        }
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        result[rotate ? "rotated" : "same_layer"] = {
            {"microseconds", samples}, {"median_us", (sorted[5] + sorted[6]) / 2},
        };
    }
    cuda_check(cudaEventDestroy(begin));
    cuda_check(cudaEventDestroy(end));
    return result;
}

int main(int argc, char ** argv) {
    try {
        if (argc < 4 || argc > 5 || (argc == 5 && std::string(argv[4]) != "--counters")) {
            throw std::runtime_error("Usage: attention-bench <output.json> <4096|8192> <vector|grouped_mma> [--counters]");
        }
        const auto depth = std::stoi(argv[2]);
        const std::string mode = argv[3];
        if ((depth != 4096 && depth != 8192) || (mode != "vector" && mode != "grouped_mma")) {
            throw std::runtime_error("Use a tested depth and CUDA path.");
        }
        const auto result = run(depth, mode, argc == 5);
        std::ofstream stream(argv[1]);
        if (!(stream << result.dump(2) << '\n')) {
            throw std::runtime_error("Could not write the result JSON.");
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
