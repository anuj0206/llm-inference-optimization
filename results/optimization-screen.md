# First optimization screen - November 20, 2026

Model: pinned Qwen3 8B Q4_K_M. Runtime: unmodified llama.cpp `8bf88ba`. Hardware: RTX 4050 Laptop GPU, 6141 MiB VRAM. All new performance runs used AC power, default warmup, five repetitions, 128 generated tokens, eight CPU threads, batch 512, microbatch 128, Flash Attention enabled, and 37/37 reported layers offloaded to CUDA.

## Cache precision comparison

| Populated context | FP16 generation, tokens/sec | Q8_0 generation, tokens/sec | FP16 peak device MiB | Q8_0 peak device MiB |
| --- | ---: | ---: | ---: | ---: |
| 0 | 36.92 | 36.60 | 4671 | 4661 |
| 8192 | 29.51 | 29.11 | 5835 | 5289 |

The 8K values average two five-repetition run means for each cache type. The first long-context pair ran Q8 then FP16; the second ran FP16 then Q8. Individual FP16 run means were 29.487543 and 29.537635; Q8 means were 29.117780 and 29.100377. Short-context values are from one five-repetition run each, FP16 followed by Q8. Raw JSON includes individual samples and within-run standard deviations.

At 8K populated depth, actual allocated context was 8448 tokens. The FP16 KV buffer was 1188 MiB; Q8_0 was 631.12 MiB. Sampled peak device memory dropped by 546 MiB, leaving 852 MiB free rather than 306 MiB. Total device-memory savings differ from the KV-buffer difference because sampling and other runtime/driver allocations contribute. The Q8 throughput reduction was approximately 1.37%; Q8 is a memory-saving tradeoff here, not a measured speed optimization. These are short runs rather than a thermal steady-state study, and more sessions would establish how stable small differences are.

Primary new performance artifacts begin with `context-8b-q4-` and timestamps `20261120T162137`, `162205`, `162233`, `162313`, `162600`, and `162640`. Matching `.metadata.json` and `.gpu.csv` files record commands, model provenance, power state, observed allocation sizes and 200 ms device-wide telemetry. Memory samples may miss brief peaks. Existing user-run forward/reverse sweeps are also preserved.

## Nsight Systems profiles

The diagnostic profiles use FP16 cache, one 128-token generation repetition, and depth 0 or 8192. The complete captures include loading, warmup and prefill. Analysis selects only repeated decode CUDA graphs: three graph components repeated 128 times at depth 0 and 127 times at depth 8192. The first long-context decode step was not in those steady graphs and is excluded. Prefill graph components repeat only 16 times and are excluded. The analysis validates this structure rather than aggregating all captured work.

| Share of summed selected GPU kernel duration | Depth 0 | Depth 8192 |
| --- | ---: | ---: |
| Quantized matrix-vector kernels | 95.46% | 75.77% |
| Attention kernels | 1.65% | 21.96% |
| Other kernels | 2.89% | 2.27% |

Selected kernel intervals cover about 97.4% and 97.7% of their respective generation windows. Remaining time includes transfers and gaps; it does not directly measure CPU overhead. Kernel sums are not end-to-end Amdahl fractions when work overlaps. Profiling overhead means these captures must not replace the unprofiled throughput measurements.

The largest kernel families are existing quantized matrix-vector paths for Q4_K and Q6_K weights, including already-fused variants. Q4_K_M is a mixed-format weight configuration. The profile supports prioritizing matrix-vector execution, but does not establish whether DRAM bandwidth, cache behavior, instruction throughput or another mechanism is limiting it. Hardware counters were not collected; current non-administrator permissions previously blocked Nsight Compute counters. Attention is a significant secondary target at long context.

Full captures and SQLite exports are retained outside OneDrive under `%LOCALAPPDATA%\LLMOptimization\profiles`. `profile-analysis.json` records exact source paths, graph selection, detailed kernel names and timings. The runtime source was not modified to collect profiles.

## Output regression screen

Both cache configurations produced byte-identical response content on all 12 fixed prompts. After separating answer correctness from output formatting, both answered 10/12 correctly. Both gave the same wrong Python slice and C++ loop-count answers. Only 4/12 matched the requested JSON value directly; several correct answers were wrapped in single-value objects, and one retrieval response was a bare JSON key/value fragment. Raw outputs and normalization decisions are retained.

The screen includes nine easy short arithmetic, code, logic and structured-data checks plus three synthetic retrieval prompts of 6870 input tokens each. The long prompts place a code at the start, middle or end. Thinking was disabled, temperature was zero, seed was 42, prefix caching was disabled, and each server used one 8448-token slot. Both cache settings retrieved all three codes after the documented output normalization. Expected answers were fixed before inference.

This is a small regression screen, not a representative accuracy benchmark or proof of equal quality. It cannot establish general coding/reasoning performance, perplexity, difficult long-context behavior, or quality at contexts beyond those tested. Fixtures are in `quality-fixtures.json`; responses and grading are in `quality-20261120T162410Z.json`.

## Next engineering task

Keep FP16 cache as the performance reference and retain Q8 as a useful memory-saving option. Investigate the existing Q4_K matrix-vector path first: measure representative shapes, verify numerical correctness, then compare a narrowly scoped change against stock kernel and end-to-end decode performance. Attention at 8K is the second target. Avoid presenting cache configuration gains as a custom CUDA contribution.
