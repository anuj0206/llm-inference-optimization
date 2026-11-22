# Documentation

These references cover the experimental code, benchmark methods, and results for running Qwen3 on an RTX 4050 Laptop GPU with 6 GB of dedicated VRAM.

## Implementation and experiments

- [Project overview and setup](../README.md) lists selected results, source code, and build commands.
- [Adaptive n-gram speculation](experiment-log.md#2026-11-21-verification-scores-and-acceptance-based-draft-budget) covers draft ranking, acceptance feedback, and output checks. [Policy source](../benchmarks/ngram)
- [Grouped Q8_0 attention](experiment-log.md#2026-11-21-custom-grouped-q8_0-decode-attention) and [grouped Q4_0 attention](experiment-log.md#2026-11-22-grouped-q4_0-decode-attention) describe the decode kernels and their isolated and full-model comparisons. [Kernel source](../benchmarks/attention/grouped-decode.cuh)
- [Time to first token and 8K prefill](experiment-log.md#2026-11-21-time-to-first-token-and-8k-prefill) covers streaming timing and kernel profiles.
- [Q4_0 versus Q8_0 KV cache](experiment-log.md#2026-11-22-q4_0-versus-q8_0-kv-cache-at-8k) compares fresh 8K latency, decode, VRAM, and retrieval.
- [Experiment log](experiment-log.md) records the full sequence of experiments and links their raw artifacts.

## Reference measurements and methods

- [Inference concepts](concepts.md) explains model size, weight precision, cache precision, prefill, and decode.
- [8B baseline](../results/qwen3-8b-baseline.md) compares 4B and 8B speed and memory use.
- [Cache and profiling experiments](../results/optimization-screen.md) report FP16 and Q8 cache results, kernel profiles, and limited output checks.
- [Q4_K kernel investigation](q4-k-investigation.md) identifies the decode path and model tensor shapes.
- [Isolated Q4_K harness](q4-k-benchmark.md) documents correctness references, timing methods, and commands.

## Benchmark entry points

| Measurement | Script |
| --- | --- |
| Stock prompt processing and generation | [`benchmark.ps1`](../scripts/benchmark.ps1) |
| Decode speed and VRAM by context depth | [`context-sweep.ps1`](../scripts/context-sweep.ps1), [`show-context-results.ps1`](../scripts/show-context-results.ps1) |
| Adaptive speculation and grouped attention | [`benchmark-combined.py`](../scripts/benchmark-combined.py) |
| Live time to first token and 8K prefill tracing | [`benchmark-ttft.py`](../scripts/benchmark-ttft.py) |
| Decode CUDA traces and kernel breakdown | [`profile-decode.ps1`](../scripts/profile-decode.ps1), [`analyze-profiles.py`](../scripts/analyze-profiles.py) |

Raw machine-readable measurements are under `results/`. Timestamped files preserve separate runs rather than replacing older evidence. Heavy weights, toolchains, compiled output and full profiler captures live under `%LOCALAPPDATA%\LLMOptimization`, outside OneDrive, and are excluded from Git. Each report identifies its relevant raw artifacts.

## What counts as a contribution

Stock-runtime configuration changes establish useful operating points. Custom CUDA work must be measured against the same stock operation, preserving numerical correctness and then improving actual model inference. Keep weight quantization, KV-cache quantization, and kernel execution improvements distinct in reports.

## Documentation convention

For each experiment, retain commands/settings, source/model identifiers, raw data, interpretation and limitations. Add a short entry to the experiment log. Update a claim when evidence changes, and distinguish measurements from estimates or hypotheses. Record failed correctness runs rather than silently replacing them with passing output.
