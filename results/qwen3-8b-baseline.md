# Qwen3 8B Q4_K_M baseline - November 20, 2026

The official model was downloaded at revision `7c41481f57cb95916b40956ab2f0b139b296d974` and its SHA-256 matched the publisher's value. The file is 5,027,783,488 bytes. Model provenance is in `target-model.json`; the weights and license are under `%LOCALAPPDATA%\LLMOptimization\models`.

Both sizes were benchmarked sequentially on AC power with the unmodified llama.cpp build `8bf88ba`. Each run used default warmup and five repetitions per workload, eight CPU threads, batch 512, microbatch 128, Flash Attention enabled, FP16 K/V cache and 99 requested GPU layers. GPU telemetry was sampled every 200 ms during loading, warmup and measurement. The fresh 4B reference uses the same monitoring as 8B. The earlier user-collected `baseline-4b-q4.json` is preserved separately.

| Metric | Qwen3 4B Q4_K_M | Qwen3 8B Q4_K_M |
| --- | ---: | ---: |
| Process 512 input tokens, tokens/sec | 2551.23 +/- 53.24 | 1423.72 +/- 22.08 |
| Process 2048 input tokens, tokens/sec | 2373.08 +/- 10.78 | 1412.62 +/- 53.70 |
| Generate 128 tokens, tokens/sec | 62.05 +/- 0.57 | 37.01 +/- 0.05 |
| Sampled peak device memory | 2853 MiB (2.79 GiB) | 4939 MiB (4.82 GiB) |
| Maximum sampled GPU temperature | 58 C | 62 C |
| Reported GPU-layer offload | 37/37 | 37/37 |

Throughput variation is the benchmark's standard deviation across five repetitions, not a confidence interval. The 8B run retained about 60% of the 4B generation throughput. Its 128-token generation test took about 3.46 seconds on average.

The 8B run's sampled device memory peak left 1202 MiB (1.17 GiB) below the GPU's reported 6141 MiB capacity. The runtime reported a 4455.34 MiB CUDA model buffer and a 333.84 MiB CPU-mapped model buffer. All reported layers were GPU-offloaded; this does not mean every model/runtime byte resides on the GPU.

Raw artifacts:

- `baseline-4b-q4-20261120T055650291621Z.json`, with matching `.metadata.json` and `.gpu.csv`.
- `baseline-8b-q4-20261120T055722079444Z.json`, with matching `.metadata.json` and `.gpu.csv`.
- Verbose runtime logs are retained locally and ignored by Git.

These are short synthetic performance tests, excluding tokenization and sampling. Prompt processing and generation are separate; generation starts at context depth zero. The tests do not establish time to first token, long-context generation speed, model quality, thermal steady state, or an allocator-exact VRAM peak. The GPU memory sampler can miss sub-200 ms spikes. Runs were ordered 4B then 8B without a controlled cooldown.

Run the selected model with `scripts/run-model.ps1 -ModelSize 8B`. Reproduce the benchmark with `scripts/benchmark.ps1 -ModelSize 8B`. The next experiment should measure decode speed and memory across increasing populated context depths.
