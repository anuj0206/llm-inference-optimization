# Hardware-Constrained LLM Inference Optimization

A CUDA performance project for running **Qwen3 8B Q4_K_M on a 6 GB NVIDIA RTX 4050 Laptop GPU**. It combines a custom grouped-query attention kernel, key/value (KV) cache quantization, tuned prefill batching, and adaptive n-gram speculation in a pinned llama.cpp build.

## Selected results

| Metric | Measured result | Benchmark |
| --- | --- | --- |
| Peak device VRAM | 5,785 → **5,146 MiB** (−11%) | Original stock FP16 cache vs. tuned Q4_0 runtime, both with an 8,448-token allocation. [Baseline](docs/experiment-log.md#2026-11-21-time-to-first-token-and-8k-prefill) · [Tuned runtime](docs/experiment-log.md#2026-11-22-grouped-q4_0-with-tuned-batching-and-feedback-n-gram) |
| Fresh 8K time to first token | 6.785 → **5.381 s** (−21%) | Same 8,192-token memo in separate runs. The tuned run uses Q4_0 KV and batch/microbatch 4,096/1,024. [Baseline](docs/experiment-log.md#2026-11-21-time-to-first-token-and-8k-prefill) · [Tuned prefill](docs/experiment-log.md#2026-11-22-q4_0-versus-q8_0-kv-cache-at-8k) |
| Grouped Q4_0 attention at 8K | 237.99 → **115.28 µs** isolated (−52%) · 28.42 → **30.69 tok/s** full-model decode (+8%) | Paired stock vs. grouped attention with Q4_0 KV in both arms and speculation disabled. [Kernel benchmark](docs/experiment-log.md#2026-11-22-grouped-q4_0-decode-attention) |
| Repetitive Python decode at 8K | 28.55 → **83.44 tok/s** (2.92×) | Stock Q4_0 vs. grouped Q4_0 with feedback n-gram at the same batch settings. Greedy token IDs matched. [Combined benchmark](docs/experiment-log.md#2026-11-22-grouped-q4_0-with-tuned-batching-and-feedback-n-gram) |

The experiments keep the model weights fixed at Q4_K_M and vary KV-cache precision independently. Feedback n-gram helped the project memo and slowed the fiction opening relative to grouped attention alone. Both prose continuations changed token IDs. [Per-workload results](docs/experiment-log.md#2026-11-22-grouped-q4_0-with-tuned-batching-and-feedback-n-gram)

## Source code

| Implementation | Source | Runtime integration |
| --- | --- | --- |
| Adaptive n-gram speculation | [Consensus backoff](benchmarks/ngram/consensus-backoff.inc), [ranked draft selection](benchmarks/ngram/ranked-draft.inc), [acceptance-based draft budget](benchmarks/ngram/acceptance-budget.inc) | [N-gram build script](scripts/build-adaptive-ngram.py) integrates the policies with llama.cpp’s `ngram-simple` path. |
| Grouped Q8_0 and Q4_0 decode attention | [CUDA kernel](benchmarks/attention/grouped-decode.cuh), [correctness and timing harness](benchmarks/attention/benchmark.cpp) | [Attention build script](scripts/build-attention-bench.py) adds cache-specific paths for single-token decode on Ada with grouped-query attention. |
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

Build the grouped attention runtime and its harness with the CUDA CUPTI component installed. The launcher selects batch 4,096, microbatch 1,024, and an 8,448-token context. Grouped attention kicks in during single-token decode once the attention operation reaches 4,096 KV rows.

```powershell
.\.venv\Scripts\python.exe .\scripts\install_cuda.py --components cuda_cupti
.\scripts\build-attention-bench.ps1
.\scripts\run-model.ps1 -GroupedQ4Attention
```

To add feedback n-gram, build the draft policy and calibrate its verification costs on your GPU.

```powershell
.\scripts\build-adaptive-ngram.ps1 -Budget
. .\scripts\enter-dev.ps1
python .\scripts\calibrate-ngram.py --grouped-q4
$calibration = Get-ChildItem .\results -Filter 'ngram-cost-grouped-q4-8k-*.json' |
    Sort-Object Name -Descending | Select-Object -First 1
.\scripts\run-model.ps1 -GroupedQ4Attention -NgramFeedback -NgramCalibration $calibration.FullName
```

### Reproduce the comparisons

Reuse the calibration from the runtime setup above.

```powershell
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-ttft.py --compare-kv
python .\scripts\benchmark-attention.py isolated --kv q4_0
python .\scripts\benchmark-attention.py decode --kv q4_0
python .\scripts\benchmark-combined.py --kv q4_0 --calibration $calibration.FullName
```

See the [documentation index](docs/README.md) for individual benchmarks and profiling workflows. The [experiment log](docs/experiment-log.md) links the raw results, records the workload and runtime settings, and includes slower variants and output mismatches. [Third-party notices](benchmarks/q4_k/THIRD_PARTY_NOTICES.md)
