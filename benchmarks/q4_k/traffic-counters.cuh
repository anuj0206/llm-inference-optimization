#pragma once

#include <cupti_range_profiler_util.h>

static json measure_traffic(const std::vector<std::unique_ptr<Weights>> &weights, cudaStream_t stream,
                            int l2_bytes) {
    using cupti::utils::MetricEvaluator;
    using cupti::utils::RangeProfiler;
    using cupti::utils::cuptiDeinitializeProfiler;
    using cupti::utils::cuptiInitializeProfiler;

    constexpr std::size_t target_layer = 17;
    const Weights &target = *weights.at(target_layer);
    const std::size_t packed_bytes = 2 * std::size_t(target.n) * target.k / 256 * sizeof(block_q4_K);
    const std::size_t activation_bytes = target.k / 32 * sizeof(block_q8_1);
    const std::size_t output_bytes = target.n * sizeof(float);
    Device<float> x(target.k), out(target.n);
    Device<block_q8_1> q(target.k / 32);
    const auto input = activation(target.k, 0);
    check(cudaMemcpyAsync(x.p, input.data(), target.k * sizeof(float), cudaMemcpyHostToDevice, stream));
    quantize(x.p, q.p, target.k, stream);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));

    CUcontext context = nullptr;
    DRIVER_API_CALL(cuCtxGetCurrent(&context));
    CUPTI_API_CALL(cuptiInitializeProfiler());
    {
        RangeProfiler access_check(context);
        CUPTI_API_CALL(access_check.EnableRangeProfiler());
        CUPTI_API_CALL(access_check.DisableRangeProfiler());
    }
    MetricEvaluator evaluator(context);
    std::vector<std::string> metrics = {
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sector_hit_rate.pct",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    };
    auto profile = [&](bool rotate, const char *implementation, auto launch) {
        const char *range_name = rotate ? "after_layer_rotation" : "after_same_pair";
        RangeProfiler profiler(context);
        CUPTI_API_CALL(profiler.EnableRangeProfiler(CUPTI_CACHE_CONTROL_MODE_NONE));
        std::vector<uint8_t> counter_data;
        CUPTI_API_CALL(profiler.SetConfig(CUPTI_UserRange, CUPTI_UserReplay, metrics, counter_data, 1));

        std::vector<uint8_t> device_partition;
        int passes = 0;
        do {
            // Recreate the cache lead-in on every counter pass. Only the target launch is counted.
            for (std::size_t offset = 1; offset < weights.size(); ++offset) {
                const Weights &previous = *weights[rotate ? (target_layer + offset) % weights.size() : target_layer];
                launch(previous.gpu_up.p, previous.gpu_gate.p, q.p, out.p, previous.n, previous.k, stream);
            }
            check(cudaGetLastError());
            check(cudaStreamSynchronize(stream));
            CUPTI_API_CALL(profiler.StartRangeProfiler());
            if (passes == 0) {
                CUPTI_API_CALL(profiler.GetDevicePartitionInfo(device_partition));
            }
            CUPTI_API_CALL(profiler.PushRange(range_name));
            launch(target.gpu_up.p, target.gpu_gate.p, q.p, out.p, target.n, target.k, stream);
            check(cudaGetLastError());
            check(cudaStreamSynchronize(stream));
            CUPTI_API_CALL(profiler.PopRange());
            CUPTI_API_CALL(profiler.StopRangeProfiler());
            if (++passes > 32) throw std::runtime_error("CUPTI did not finish submitting counter passes");
        } while (!profiler.GetPassState().isAllPassSubmitted);
        CUPTI_API_CALL(profiler.DecodeCounterData());
        CUPTI_API_CALL(profiler.DisableRangeProfiler());

        evaluator.setDevicePartitionInfo(device_partition);
        std::vector<MetricEvaluator::RangeInfo> ranges;
        evaluator.evaluateAllRanges(counter_data, metrics, ranges);
        if (ranges.size() != 1 || ranges.front().rangeName != range_name) {
            throw std::runtime_error("CUPTI did not return the expected single-launch range");
        }
        json counters = json::object();
        for (const auto &metric : ranges.front().metricAndValues) {
            if (!std::isfinite(metric.value) || metric.value < 0) {
                throw std::runtime_error("Invalid CUPTI metric: " + metric.metricName);
            }
            counters[metric.metricName] = metric.value;
        }
        const double dram_read_bytes = counters.at("dram__bytes_read.sum");
        const double l2_read_bytes = 32 * counters.at("lts__t_sectors_op_read.sum").get<double>();
        if (dram_read_bytes == 0 || l2_read_bytes == 0) {
            throw std::runtime_error("CUPTI reported zero traffic for the measured launch");
        }
        return json{
            {"range", range_name}, {"implementation", implementation}, {"passes", passes},
            {"launches_per_range", 1}, {"counters", counters},
            {"l2_read_sector_bytes", l2_read_bytes},
            {"dram_read_bytes_per_packed_weight_byte", dram_read_bytes / packed_bytes},
            {"l2_read_sector_bytes_per_packed_weight_byte", l2_read_bytes / packed_bytes},
        };
    };
    json samples = json::array();
    for (const bool rotate : {true, false}) {
        samples.push_back(profile(rotate, "stock_fused", &matvec<true>));
        samples.push_back(profile(rotate, "packed_header_fused", &matvec<true,4,1,true>));
    }
    CUPTI_API_CALL(cuptiDeinitializeProfiler());
    return {
        {"layer", target.layer}, {"packed_weight_bytes", packed_bytes},
        {"unique_q8_1_activation_bytes", activation_bytes}, {"output_bytes", output_bytes},
        {"l2_capacity_bytes", l2_bytes}, {"cache_control", "none"},
        {"lead_in_launches", weights.size() - 1}, {"samples", samples},
    };
}
