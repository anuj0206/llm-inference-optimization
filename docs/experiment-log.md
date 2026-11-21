# Experiment log

## 2026-11-19: Project and offline environment

Chose hardware-constrained local inference with C++/CUDA and llama.cpp. Audited the RTX 4050 Laptop GPU, approximately 32 GB system RAM and installed MSVC tools. Created an offline Python environment and private GitHub repository; deferred large downloads until home Wi-Fi. See the main README for setup history.

## 2026-11-20: Native CUDA runtime and first models

Installed user-local CUDA 13.5 development components, CMake, Ninja and Nsight tools. Built the pinned unmodified llama.cpp revision. Verified a CUDA smoke kernel, GPU inference and Nsight Systems tracing. Nsight Compute counters require additional administrator access. Downloaded and SHA-256-verified official Qwen3 4B and 8B Q4_K_M models. See [environment validation](../results/setup-validation.md) and [8B baseline](../results/qwen3-8b-baseline.md).

## 2026-11-20: Context depth and ordering

Measured generation and sampled VRAM at context depths 0 through 8192, then repeated in reverse order. Matching forward/reverse speeds were close. The 8K run used about 5835 MiB of the reported 6141 MiB and generated about 29.6 tokens/sec. This established a useful memory-constrained baseline and motivated cache experiments. Raw timestamped context artifacts preserve both sweeps.

## 2026-11-20: Cache precision and kernel profiles

Q8_0 KV cache saved 546 MiB at 8K depth, with approximately 1.4% lower generation throughput in the initial paired/reversed-order tests. Twelve small output checks gave identical responses under both cache settings, including the same errors; this is not a broad quality evaluation. Selected short-context GPU kernel time was about 95% quantized matrix-vector operations; at 8K it was about 76% matrix-vector and 22% attention. See [the full screen](../results/optimization-screen.md).

## 2026-11-20: Q4_K investigation

Traced packed Q4_K weights, temporary Q8_1 activations and existing fusion. Identified the 12288-by-4096 up/gate pair as a concrete target: 36 calls per token, about 308 us per profiled call, about 41% of selected short-context GPU kernel time. The pair contains 54 MiB of packed weights versus 24 MiB GPU L2. Bandwidth limitation remains a hypothesis without hardware counters. See [the investigation](q4-k-investigation.md).

## 2026-11-20: Isolated harness

Added a project-owned harness that compiles exact stock kernel excerpts outside the source checkout, reads real weights directly from verified GGUF offsets, and compares fused outputs with a scalar-decoding CPU reference and an unfused GPU control. The harness gates CUDA-event timing on correctness. Timing covers prepared activations and quantization plus operation, with one-pair reuse and rotation through all 36 layer pairs. See [the harness guide](q4-k-benchmark.md) for design and the latest run interpretation.

Validation passed 36/36 cases with 72/72 checker negative detections. A follow-up run added independent activation-quantization checks, with zero violations. Stock fused execution was about 309 us, while the unfused control was about 315 us; fusion was already present in production, so this is a harness baseline rather than an original speedup. Raw successful runs, metadata and a generated summary are retained in `results/`. No runtime changes were integrated.

## 2026-11-21 Two warps per block

The isolated candidate switched the fused operation from 128 to 64 threads per block while keeping one output row per block. Both timed runs passed 36/36 cases against the CPU reference. The prepared-activation latency changed by -0.27% and +0.61% in the layer rotation. With quantization included the candidate was slower in both runs, by 0.95% and 0.27%. See the [exact diff command, tolerances and raw runs](q4-k-benchmark.md#two-warp-experiment).

## 2026-11-21 One warp and RTX 4050 hardware

Both three-way runs passed 36/36 cases. One warp was about 1% slower in the prepared-activation layer rotation and 0.6% slower with activation quantization included. CUDA reports an occupancy ceiling of 24 resident warps per SM for one warp per block and 36 for the stock kernel. The [RTX 4050 tuning notes](rtx4050-tuning.md) record the hardware, disassembly observations, measurements and candidates for the next experiment.

## 2026-11-21 Two output rows per block

Keeping four warps per block and processing two rows halved the grid size. Both runs passed 36/36 cases and matched stock on all checked outputs. The candidate added 0.22–0.23% latency with prepared activations in the layer rotation and 0.15–0.16% with activation quantization included. The higher register count lowered calculated residency from 36 to 32 warps per SM. The [harness guide](q4-k-benchmark.md#two-rows-per-block) records both raw runs and the exact diff command.

## 2026-11-21 Streaming-read bandwidth

Advanced Threat Defense blocked both Nsight Compute attempts. A CUDA-event benchmark ran without elevation and measured approximately 187.5–187.7 GB/s for 256 and 512 MiB input buffers in both size orders. The stock Q4_K kernel's nominal weight throughput was about 183.4 GB/s. See the [streaming-read results](rtx4050-tuning.md#streaming-read-bandwidth) for the method, raw data and limits of the comparison.

## 2026-11-21 CUPTI memory counters

Added an opt-in, in-process CUPTI build of the isolated stock kernel harness. The elevated run passed the 36 correctness cases and measured 54.010 MiB of DRAM reads and 54.405 MiB of L2 read sectors after the rotation lead-in. The 324 MiB of L1 sector requests and the disassembly point to fewer metadata load instructions as a candidate to test. See the [counter method and results](q4-k-benchmark.md#in-process-memory-counters).

## 2026-11-21 Packed Q4_K header load

Compiled a single 128-bit metadata load per matrix in the fused inner loop. The variant matched stock bitwise across all 36 correctness cases. CUPTI showed 33.3% fewer L1 global-load requests and 11.1% less L1 sector volume, with DRAM reads still about 54 MiB. Prepared-activation latency in the rotating workload was 0.06–0.07% higher across two runs. The [benchmark notes](q4-k-benchmark.md#packed-q4_k-header-load) link to the timings and counter comparison.

## 2026-11-21 N-gram speculative decoding

Enabled llama.cpp's existing `ngram-simple` mode through the opt-in `-NgramSpeculation` launch switch. The test matches six tokens in the history and drafts up to eight. The target model verifies the drafts. We ran two repetitions per mode and workload on AC, with 8,192 input tokens, 128 generated tokens and FP16 KV cache. Each session reported 37/37 layers offloaded. Prompt reuse was disabled.

| Workload | Stock decode | N-gram decode | Accepted drafts per run |
| --- | ---: | ---: | ---: |
| Repetitive Python continuation | 28.96 tokens/s | 60.38 tokens/s | 72 / 138 |
| Prose response to project notes | 28.80 tokens/s | 29.11 tokens/s | 2 / 16 |

Greedy token IDs matched across all repeats for both prompts. The device-wide sampled peak was 5,785 MiB in every session. The Python case continued closely repeated helper functions and reached 2.08× the stock decode rate. The single prose case showed little change. Measure other prompts before generalizing the speedup. [Raw prompts, outputs and measurements](../results/ngram-8k-20261121T220522Z.json)

## 2026-11-21 Adaptive n-gram replay

`scripts/replay-ngram.py` replays the saved greedy token sequences. The policy sees the prompt and previously emitted tokens. The replay uses the next stock tokens to count the accepted prefix and advances by that prefix plus the next token from the verifier. It reproduces the measured draft count, accepted count and verification calls for the fixed six-token lookup and eight-token draft limit on both prompts. Counts start after prefill supplies the first output token, so decode without speculation would take 127 target calls for these traces.

```powershell
.\.venv\Scripts\python.exe scripts\replay-ngram.py results\ngram-8k-20261121T220522Z.json
```

The backoff prototype keeps the existing six-token lookup. When it yields no draft, it tries four-, three- and two-token suffixes, in that order. For each length, it checks the most recent 16 matching continuations. At each draft position, it requires at least three votes and 75% agreement among the remaining continuations. It proposes at most three tokens from this backoff path.

| Workload | Policy | Drafted | Accepted | Verification calls | Target calls |
| --- | --- | ---: | ---: | ---: | ---: |
| Python continuation | Fixed 6/8 | 138 | 72 | 18 | 55 |
| Python continuation | Fixed 3/4 | 124 | 90 | 31 | 37 |
| Python continuation | Consensus backoff | 164 | 90 | 27 | 37 |
| Prose response | Fixed 6/8 | 16 | 2 | 2 | 125 |
| Prose response | Fixed 3/4 | 60 | 12 | 15 | 115 |
| Prose response | Consensus backoff | 15 | 8 | 5 | 119 |

These are replay counts from two traces. They exclude GPU verification latency and CPU lookup time. Fixed 3/4 finds more accepted prose drafts and makes three times as many verification calls as consensus backoff. Compare actual time before selecting a policy.

The runtime version can index suffix occurrences per sequence, update the index with confirmed tokens, and propose drafts through the existing llama.cpp interface. Keep the target verifier and KV rollback. Track acceptance separately by suffix length and draft position. For each candidate draft length, estimate the expected number of tokens emitted by the pass and divide by the measured time for lookup, verification and any rollback. Include a zero-length draft as the ordinary decode option. Time both stock and the adaptive policy with the same model and cache on a held-out mix of prose, summarization and code at 8K. Check greedy token IDs, decode tokens/sec, VRAM and the number of accepted drafts. If within-request repeats are scarce, past responses or a domain corpus could supply additional draft candidates. [SuffixDecoding](https://arxiv.org/abs/2411.04975) · [REST](https://aclanthology.org/2024.naacl-long.88/)
