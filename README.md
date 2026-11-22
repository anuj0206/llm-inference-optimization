# Hardware-Constrained LLM Inference Optimization

A CUDA performance project for running **Qwen3 8B Q4_K_M on a 6 GB NVIDIA RTX 4050 Laptop GPU**. Experiments use a pinned llama.cpp build to measure time to first token, generation throughput, and GPU memory use as the key/value (KV) cache grows.

## Selected results

| Experiment | Measured result | Workload |
| --- | --- | --- |
| [Ranked n-gram speculation](docs/experiment-log.md#2026-11-21-verification-scores-and-acceptance-based-draft-budget) | 28.94 → **99.60 tok/s** | Repetitive Python continuation at 8K with FP16 KV |
| [Grouped Q8_0 CUDA attention](docs/experiment-log.md#2026-11-21-custom-grouped-q8_0-decode-attention) | 237.36 → **169.72 µs** isolated attention latency, and 28.54 → **30.05 tok/s** full-model decode | 8K context with Q8_0 KV in both arms |
| [Q8_0 KV cache](docs/experiment-log.md#2026-11-21-time-to-first-token-and-8k-prefill) | 5,785 → **5,257 MiB** sampled peak VRAM | FP16 versus Q8_0 K and V with an 8,448-token context allocation |
| [Prefill microbatch tuning](docs/experiment-log.md#2026-11-21-8k-prefill-microbatch-sweep) | 6.815 → **6.048 s** median time to first token | Fresh 8K prompt, microbatch 128 → 512, batch fixed at 512 |

Speculation gains depend on how much text the draft can reuse. Several prose cases produced different greedy continuations. The [combined benchmark](docs/experiment-log.md#2026-11-21-combined-grouped-q8_0-attention-and-adaptive-n-gram) reports per-workload speed, draft acceptance, and token agreement.

## Source code

| Implementation | Source | Runtime integration |
| --- | --- | --- |
| Adaptive n-gram speculation | [Consensus backoff](benchmarks/ngram/consensus-backoff.inc), [ranked draft selection](benchmarks/ngram/ranked-draft.inc), [acceptance-based draft budget](benchmarks/ngram/acceptance-budget.inc) | [N-gram build script](scripts/build-adaptive-ngram.py) integrates the policies with llama.cpp’s `ngram-simple` path. |
| Grouped Q8_0 decode attention | [CUDA kernel](benchmarks/attention/grouped-q8.cuh), [correctness and timing harness](benchmarks/attention/benchmark.cpp) | [Attention build script](scripts/build-attention-bench.py) adds a dispatch path for single-token decode on Ada with grouped-query attention. |
| Q4_K feed-forward kernels | [Decode microbenchmark](benchmarks/q4_k/benchmark.cu), [fused prefill tile-width harness](benchmarks/prefill-mmq/benchmark.cpp) | [Decode harness guide](docs/q4-k-benchmark.md), [prefill tile-width experiment](docs/experiment-log.md#2026-11-21-fused-q4_k-prefill-tile-width) |

The build scripts use a separate, pinned llama.cpp checkout and generate the patched translation units and runtime DLLs locally. [Pinned llama.cpp revision](https://github.com/ggml-org/llama.cpp/tree/8bf88ba9368e70fe3ba1b0001b34e70ffc713716)

## Build and run on Windows

From this repository’s root in PowerShell, use Git, Python 3.13, `uv`, Visual Studio Build Tools with the x64 C++ toolset and Windows SDK, and a compatible NVIDIA driver. The project build targets native CUDA `sm_89` and was validated with CUDA 13.5 and MSVC 19.51. [Tested environment](results/setup-validation.md)

```powershell
.\scripts\setup-offline.ps1
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe .\scripts\install_cuda.py
git clone https://github.com/ggml-org/llama.cpp.git "$env:LOCALAPPDATA\LLMOptimization\llama.cpp"
git -C "$env:LOCALAPPDATA\LLMOptimization\llama.cpp" checkout 8bf88ba9368e70fe3ba1b0001b34e70ffc713716
.\.venv\Scripts\python.exe .\scripts\download_model.py --size 8B
.\scripts\build-llama.ps1
.\scripts\run-model.ps1 -ModelSize 8B
```

The CUDA components and GGUF weights require several gigabytes of downloads. The scripts verify their SHA-256 hashes and store the toolkit, model, and build under `%LOCALAPPDATA%\LLMOptimization`. The model downloader also supports `--size 4B` for the smaller reference model. Pass `--profilers` to the CUDA installer to include Nsight Systems and Nsight Compute.

### Run the custom runtime

The attention harness requires the CUDA CUPTI component. This sequence builds both variants and runs one prompt through grouped Q8_0 attention with ranked n-gram speculation.

```powershell
.\.venv\Scripts\python.exe .\scripts\install_cuda.py --components cuda_cupti
.\scripts\build-attention-bench.ps1
.\scripts\build-adaptive-ngram.ps1 -Budget
. .\scripts\enter-dev.ps1
python .\scripts\prepare_combined_runtime.py
.\scripts\run-model.ps1 -Context 8448 -GroupedQ8Attention -AdaptiveNgram
```

### Reproduce the comparisons

```powershell
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-combined.py
python .\scripts\benchmark-ttft.py --sweep-microbatch
```

See the [documentation index](docs/README.md) for individual benchmarks and profiling workflows. The [experiment log](docs/experiment-log.md) links the raw results, records the workload and runtime settings, and includes slower variants and output mismatches. [Third-party notices](benchmarks/q4_k/THIRD_PARTY_NOTICES.md)
