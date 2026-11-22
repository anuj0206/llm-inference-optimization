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

## 2026-11-21 Adaptive n-gram runtime

Added an opt-in C++ implementation of consensus backoff inside the pinned runtime's `ngram-simple` lookup. It scans the confirmed request history for matching suffixes and keeps up to 16 recent matches per lookup length. It uses the greedy voting thresholds from the replay and submits drafts to the existing target verifier. The variant adapts lookup and draft lengths using agreement in the history.

`scripts/build-adaptive-ngram.ps1` compiles the modified n-gram host object and relinks `llama-common.dll` into a separate build directory using the stock objects and import libraries. The launchers, server implementation and CUDA library in the candidate directory are byte-identical copies of the stock files. The benchmark records the common-library hashes in its output.

The live comparison uses the archived Python and prose token sequences plus a new prose briefing. Each request has 8,192 prompt tokens and generates 128 tokens with greedy sampling and FP16 KV cache. Each mode runs twice in opposite order on AC, with full GPU offload and prompt reuse disabled.

| Workload | Stock decode | Fixed n-gram | Adaptive n-gram | Adaptive drafts accepted per run |
| --- | ---: | ---: | ---: | ---: |
| Repetitive Python continuation | 28.97 tok/s | 60.86 tok/s | 85.18 tok/s | 90 / 164 |
| Prose project review | 29.07 tok/s | 29.18 tok/s | 30.52 tok/s | 8 / 15 |
| Held-out prose briefing | 28.92 tok/s | 28.62 tok/s | 28.73 tok/s | 0 / 8 |

All greedy token IDs matched stock across the 18 measured requests. The replay reproduced the live draft, acceptance and verification counts for both speculative modes on all three prompts. Every session peaked at 5,785 MiB in the 200 ms GPU samples. The adaptive variant improved the project-review prompt by 5.0% over stock and left the briefing within 1% of stock. The brief produced no accepted drafts. The two prose prompts support a workload-dependent speedup. [Raw results](../results/ngram-adaptive-8k-20261121T223052Z.json)

## 2026-11-21 Research on speculative decoding for prose

The briefing's only draft came from the fixed six-token lookup. Its first proposed token was wrong, and the two-, three-, and four-token lookups at that position also led to the same wrong token. Elsewhere in the briefing, fixed three-token lookup with a four-token draft cap found three accepted tokens in four verification calls. It took 124 target calls in replay versus 127 for the current adaptive policy. This is an offline comparison on one saved trace. [Benchmark replay](../scripts/replay-ngram.py) · [Input trace](../results/ngram-adaptive-8k-20261121T223052Z.json)

- **Rank candidate drafts by expected throughput.** SuffixDecoding compares multiple suffix lengths and uses historical continuation frequencies to estimate how many draft tokens the verifier will accept. It reports 1.33× stock speed on Spec-Bench writing with linear suffix drafts, versus 1.23× for prompt lookup, on Llama-3.1-8B-Instruct with an H100. For our 8K Qwen3 run, time lookup, verification and rollback for short drafts as well as the existing eight-token cap. Use those times and estimated acceptance to choose among candidate drafts and ordinary decode. [SuffixDecoding](https://arxiv.org/html/2411.04975v3)
- **Query again after each proposed token.** ANPD predicts the next token from the longest available match and repeats the lookup as the draft grows. A variant of our policy could start with a supported continuation and find a new match for the following token when the original matches diverge. Measure the extra lookup time and stop extending when the estimated benefit falls below the verification cost. ANPD evaluated summarization and code generation. [Adaptive N-gram Parallel Decoding](https://aclanthology.org/2024.naacl-industry.2/)
- **Test more draft sources.** SAM-Decoding combines current-request token matches with an index of prior model responses. Its overall Spec-Bench speedup was 1.84× with the static corpus and 1.64× with only the dynamic index on Vicuna-7B and an RTX A6000. Our pinned llama.cpp already accepts external token statistics in `ngram-cache` through `--lookup-cache-static`. A small local cache of prior Qwen3 responses can test this idea before we build an index. [SAM-Decoding](https://arxiv.org/html/2411.10666) · [llama.cpp n-gram cache](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/docs/speculative.md#n-gram-cache-ngram-cache)

Use fresh prompts for the next live comparison. Include document-grounded summaries and freeform explanations, with the repetitive Python case as a regression check. Compare wall-clock decode tokens per second and confirm greedy token parity at the same 8K prompt depth and FP16 KV precision.

## 2026-11-21 Ranked n-gram drafts and prose cache

The separate adaptive DLL now supports ranked contiguous drafts and a rolling lookup variant. Both use up to 16 prior matches for each suffix length from two through six. The score discounts sparse matches, sums the estimated probabilities that the verifier accepts each successive prefix, and divides expected emitted tokens by the calibrated pass time. A draft must beat ordinary decode by at least 1.5% in that estimate. The rolling variant also tries extending a draft by looking up each newly proposed token in the confirmed history when the best contiguous draft is shorter than four tokens.

Calibration forces draft lengths zero through eight on the archived 8K prose briefing, and uses synthetic rejection to accept zero draft tokens on every pass. This measures the target batch and rejection path with minimal draft lookup. We use the median of two zero-draft measurements and a nondecreasing envelope of the measured times. Actual drafting overhead is included in the live benchmark. [Calibration data](../results/ngram-cost-8k-20261121T225729Z.json)

| Draft tokens | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated ms per pass | 34.37 | 35.29 | 36.49 | 36.95 | 37.50 | 37.50 | 37.58 | 37.61 | 39.42 |

The live benchmark uses 8,192 input tokens, 128 greedy output tokens, FP16 KV and one CUDA-offloaded slot. It runs two document-grounded prose prompts, two new writing prompts after an unrelated context, and the existing Python continuation. It compares stock, consensus backoff, ranked drafts, ranked drafts with rolling lookup, and the built-in `ngram-cache` with and without a static cache. The latter cache contains bigram continuation counts from 12 independently generated Qwen3 answers on related GPU topics. The cache modes each start a fresh process per prompt because the pinned runtime retains n-gram statistics across requests within a server slot. The second pass reverses mode and prompt order.

```powershell
.\scripts\build-adaptive-ngram.ps1
. .\scripts\enter-dev.ps1
python .\scripts\calibrate-ngram.py
$calibration = Get-ChildItem -LiteralPath .\results -Filter 'ngram-cost-8k-*.json' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
python .\scripts\benchmark-prose-ngram.py $calibration.FullName
```

| Workload | Stock | Backoff | Ranked | Ranked + rolling | Cache only | Seeded cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python continuation | 28.96 | 85.07 | 100.15 | 100.51 | 112.55 | 113.34 |
| Project status memo | 28.97 | 37.63 | 42.67 | 42.57 | 40.25 | 41.01 |
| Plain-English rewrite | 29.07 | 31.92 | 35.87 | 35.80 | 34.90 | 35.89 |
| Freeform explanation | 28.88 | 31.06 | 31.18 | 31.05 | 32.14 | 32.10 |
| Fiction opening | 28.94 | 28.90 | 28.15 | 28.12 | 28.33 | 28.29 |

Values are mean decode tokens per second over two runs per mode and prompt. The laptop reported AC power before and after the paired runs. All 60 requests completed with 37/37 layers offloaded and every server session had a sampled peak of 5,785 MiB. The static cache used 35,304 bytes for 1,210 two-token keys from 1,521 generated tokens. [Raw timings, token IDs and corpus](../results/ngram-prose-8k-20261121T230013Z.json)

Ranked drafting improved the memo by 47.3% and the rewrite by 23.4% with exact greedy token parity in both runs. Rolling lookup had the same draft and acceptance counts on all five prompts and stayed within 0.5 tok/s of ranked. Seeding the cache gained 1.9% on the memo and 2.8% on the rewrite over the empty-cache control.

The benchmark ended with `token_mismatch`. All speculative modes diverged on token 21 of the freeform explanation. Ranked, rolling, and both cache variants also diverged on fiction. Backoff and the empty cache diverged on the rewrite. The mismatch positions repeated across both passes. Every mode matched the memo and Python outputs. Treat the timings from mismatching requests as throughput observations with different generated sequences.

The probability probe reproduced the freeform mismatch with the same executable for ordinary decode and ranked drafting. At token 21, ordinary decode ranked ` not` above ` often` by 0.0545 log-probability. Ranked decoding reversed the order by 0.0343. Forced one- and eight-token batches also chose ` often` at the same position with zero accepted draft tokens. This shows the target score order can change under batched execution even when all drafts are rejected. The first operation producing the numerical difference has yet to be isolated. [Probability probe](../results/ngram-parity-8k-20261121T231313Z.json)

```powershell
python .\scripts\diagnose-ngram-parity.py .\results\ngram-prose-8k-20261121T230013Z.json
```

## 2026-11-21 Verification scores and acceptance-based draft budget

The verifier submits the last emitted token followed by the draft to the same Qwen3 model. It requests logits for every position. Causal attention restricts each position to its prefix, so the logits after the last emitted token score the first draft token. Qwen3 applies final RMS normalization and the vocabulary projection to obtain one row of logits per position. The existing sampler picks a token from each row. It accepts the matching draft prefix, emits the target token at the first mismatch, and discards the unused speculative KV entries. [Batch construction](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/tools/server/server-context.cpp#L647) · [Qwen3 output graph](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/src/models/qwen3.cpp#L55) · [Verifier sampler](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/common/sampling.cpp#L698)

The earlier freeform result left open whether small numerical differences accumulated over many decode passes. The matched-prefix probe prefills the same 8,211 tokens in each process and confirms the same next token as the anchor. The first generation pass then scores the same anchor with batch sizes one, two, and nine. Stock chooses ` not` with a 0.0530 log-probability lead. Both batched passes choose ` often`, with leads of 0.00055 and 0.01167, and the synthetic verifier accepts zero drafts. One verification pass is enough to reproduce the score reversal from a common KV prefix. [Matched-prefix probe](../results/ngram-parity-8k-20261121T232921Z.json)

CUDA selects matmul kernels by the number of activation columns as well as tensor type. The quantized matrix-vector path has different templates for single and multiple columns, and its fused variant is available only for the single-column case. These are places where the arithmetic can differ. The probability probe observes the end of the graph. [CUDA dispatch](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/ggml/src/ggml-cuda/ggml-cuda.cu#L2535) · [Quantized matmul templates](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/ggml/src/ggml-cuda/mmvq.cu#L1061)

With `GGML_CUDA_DISABLE_FUSION=1`, all three matched-prefix arms produced the same ten output tokens. The one-token baseline itself now chose ` often` at token 21. This implicates the execution path affected by CUDA fusion and changes the stock reference at the same time. The toggle applies during prefill and decode, so the specific operation responsible is still unresolved. [Fusion-disabled probe](../results/ngram-parity-8k-20261121T233553Z.json)

The opt-in `budget` policy starts each sequence at eight possible draft tokens. After every four verified drafts, it compares the emitted tokens implied by accepted prefixes with the calibrated time for the observed draft lengths. A window that misses the 1.5% advantage threshold cuts the cap in half, or quarters it when all four drafts miss. A profitable window with a fully accepted cap-sized draft can double the cap up to eight. At zero the policy skips eight decode decisions, then tries a single-token draft when a candidate is available. Successful probes resume at two. Misses increase the cooldown up to 32 decisions.

The count comes from llama.cpp's existing verifier callback and uses the length actually offered after request limits. State resets at each sequence start. The budget limits candidate scoring inside the ranked policy. The alternate build recompiles the two host objects and uses the same CUDA DLL and verifier executable. A directed controller test exercised shrinking, cooldown, probe rejection, recovery, and reset. The paired benchmark uses the five archived 8K prompts, compares stock, ranked, and ranked with feedback twice, and reverses both mode and prompt order.

```powershell
.\scripts\build-adaptive-ngram.ps1 -Budget
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-prose-ngram.py .\results\ngram-cost-8k-20261121T225729Z.json `
    --acceptance-budget .\results\ngram-prose-8k-20261121T230013Z.json
python .\scripts\diagnose-ngram-parity.py .\results\ngram-prose-8k-20261121T230013Z.json --same-prefix
python .\scripts\diagnose-ngram-parity.py .\results\ngram-prose-8k-20261121T230013Z.json `
    --same-prefix --disable-cuda-fusion
```

| Workload | Stock tok/s | Ranked tok/s | Ranked + feedback tok/s | Drafted tokens per run, ranked → feedback |
| --- | ---: | ---: | ---: | ---: |
| Python continuation | 28.94 | 99.60 | 100.98 | 232 → 232 |
| Project status memo | 28.98 | 42.45 | 41.66 | 189 → 175 |
| Plain-English rewrite | 29.09 | 35.90 | 34.57 | 231 → 172 |
| Freeform explanation | 29.11 | 31.24 | 32.18 | 204 → 89 |
| Fiction opening | 28.98 | 28.16 | 28.27 | 141 → 52 |

All 30 requests completed with 37/37 layers offloaded and a sampled 5,785 MiB peak in each session. The figures average the two passes. Feedback stayed at the initial cap on Python. Both fiction runs dropped the cap from eight to two to zero, then recovered or paused again based on probe results. The memo and rewrite lost 1.9% and 3.7% versus ranked while matching stock tokens in both runs. Freeform and fiction still diverged from stock. Freeform first changed at token 21 in both speculative policies. Fiction first changed at token 73 in ranked and 44 with feedback. The timings for divergent sequences measure throughput on different outputs. [Paired benchmark and budget decisions](../results/ngram-budget-8k-20261121T233006Z.json)

## 2026-11-21 FP16 long-context attention dispatch

The alternate CUDA DLL changes dispatch for one-query, one-sequence FP16 attention with head dimension 128, GQA ratio four, an eligible mask and at least 4,096 KV rows. It selects the pinned runtime's existing grouped Tensor Core implementation in place of the vector path. Other attention shapes use the pinned dispatcher. Both choices in the comparison run through the same rebuilt DLL.

The isolated harness creates Qwen3-shaped tensors with 32 query heads and eight KV heads. The storage is FP16 for K and V, with the heads interleaved within each token row as in the model cache. Each call computes a dense attention result on a fixed seeded input and scale. The harness checks both paths against a double-precision CPU reference before timing. Eight separate KV pairs model cache turnover across layers. Repeating the same pair measures a more cache-resident case. CUDA events bracket all kernels issued by the attention operation, including any partial reduction. The two timing passes reverse depth and dispatch order.

CUPTI uses the toolkit's Range Profiler in the same executable. Every replay restores one of the two cache states with seven lead-in calls outside the measured range. Counters cover the complete target attention call, including reduction where applicable. The GPU driver denied performance-counter access in the regular-user probe.

```powershell
.\scripts\build-attention-bench.ps1
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-attention.py isolated
python .\scripts\benchmark-attention.py decode
# On this laptop, use the project paths from an elevated PowerShell for counters.
$env:PATH = (Join-Path $env:CUDA_PATH 'lib\x64') + ';' + $env:PATH
python .\scripts\benchmark-attention.py counters
```

| Populated KV depth | Rotated vector µs | Rotated grouped µs | Same-layer vector µs | Same-layer grouped µs | Full-model vector tok/s | Full-model grouped tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 110.72 | 108.65 | 85.08 | 29.67 | 32.16 | 32.39 |
| 8,192 | 225.62 | 215.75 | 206.95 | 191.50 | 28.96 | 29.06 |

The isolated numbers average four session medians per shape and path. The 4K rotated advantage changed sign with order. At 8K every grouped session was faster than every vector session, with a 4.4% reduction in mean latency. Both paths passed all CPU comparisons. The largest grouped absolute error was 0.0000731. [First isolated run](../results/attention-isolated-20261121T235507Z.json) · [Isolated repeat](../results/attention-isolated-20261122T000057Z.json)

The full-model run used the same 8B weights, FP16 KV, CUDA graph settings, 128 generated greedy tokens and no speculative decoding. It ran two archived prompts twice per depth and path on AC, reversing order in the second pass. All sessions offloaded 37/37 layers. The sampled peak matched across paths at each depth, 5,207 MiB at 4K and 5,785 MiB at 8K. Both prompts matched stock token IDs at 8K. At 4K, the memo diverged at token 108 in both grouped runs. The full-model speed increases were 0.71% at 4K and 0.34% at 8K. [Full-model comparison](../results/attention-decode-20261121T235744Z.json)

At 8K, one layer holds 32 MiB of unique FP16 K and V. Streaming that amount once at the measured 188 GB/s streaming-read reference would take about 179 µs, compared with 216 µs for grouped attention. This scenario suggests room for a focused kernel experiment. It assumes a single read at the streaming rate and omits attention arithmetic, so it neither quantifies actual DRAM traffic nor predicts a wall-clock gain. A grouped Q8_0 variant could also test whether smaller reads translate into speed at this workload.

### CUPTI memory counters

The regular-user probe returned the NVIDIA driver permission error `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`. The elevated follow-up completed both forward and reverse passes. Each range required two CUPTI replay passes. The values below average two measurements per path and depth. L2 read volume is the sector count times 32 bytes. [Raw counter report](../results/attention-counters-20261122T002612Z.json)

| KV depth | Distinct FP16 KV MiB | Rotated DRAM reads, vector → grouped MiB | Rotated L2 read sectors, vector → grouped MiB |
| --- | ---: | ---: | ---: |
| 4,096 | 16 | 16.07 → 16.14 | 63.69 → 16.64 |
| 8,192 | 32 | 32.09 → 32.14 | 125.99 → 32.81 |

Grouped attention cut rotated L2 read-sector volume by about 74% at both depths. DRAM read volume was close to the size of one distinct KV pair on both paths. The vector path requested close to four times the unique bytes from L2, consistent with four query heads sharing each KV head. The results indicate that L2 served most of the repeated reads in this isolated workload.

Repeating the same 4K KV pair reduced DRAM reads to 13.35 MiB on vector and 10.90 MiB on grouped. At 8K, the same-layer result stayed near 32 MiB on both paths. The 4K warm-vector L2 hit-rate metric exceeded 100%, so the interpretation uses read sector and DRAM byte totals. For a custom kernel aimed at DRAM savings, grouped Q8_0 is a more direct next experiment. Compare it with stock Q8_0 on the same populated depth, layout, and cache lead-ins, then check end-to-end decode.

## 2026-11-21 Custom grouped Q8_0 decode attention

The custom CUDA path handles batch-one decode on Ada, with 128-dimensional heads and four query heads per KV head. Both K and V use the pinned llama.cpp Q8_0 layout. The dispatcher requires compatible strides and a dense mask, and sends prefill, multi-token verification, and other shapes through the existing CUDA dispatcher. The experiment selects the new path through `LLM_OPT_ATTENTION_Q8=1`. [Kernel source](../benchmarks/attention/grouped-q8.cuh)

Each block processes one KV head and one 128-position split. Four warps cooperate to stage each 32-position packed tile in shared memory, then each warp computes one query head. The QK dot product uses the stock temporary Q8_1 query quantization and packed DP4A instructions. Each warp maintains its own online softmax and accumulates the weighted Q8_0 values in FP32. A second kernel merges split partials using their maxima and normalization sums.

The compiled attention block uses 56 registers per thread and 9,856 bytes of static shared memory per block. The merge uses 40 registers per thread. The resource report shows zero local memory for both kernels.

The harness packs identical seeded KV inputs for the two kernels and checks against a double-precision CPU reference using dequantized Q8_0 KV and the vector path's scaled, quantized query. Both paths pass dense and masked tests at both depths. The masked case hides an entire split and a partial suffix. Timing rotates eight KV pairs and reverses depth and implementation order. The full-model comparison keeps the model, Q8_0 cache format, prompts, and sampling settings fixed.

```powershell
.\scripts\build-attention-bench.ps1
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-attention.py isolated --kv q8_0
python .\scripts\benchmark-attention.py decode --kv q8_0
.\scripts\run-model.ps1 -GroupedQ8Attention -Context 8448
```

The launcher selects Q8_0 for both caches. The kernel becomes eligible when attention reaches at least 4,096 KV rows.

| KV depth | Stock attention | Grouped attention | Attention latency reduction | Stock decode | Grouped decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 126.94 µs | 81.57 µs | 35.7% | 31.46 tok/s | 32.61 tok/s |
| 8K | 237.36 µs | 169.72 µs | 28.5% | 28.54 tok/s | 30.05 tok/s |

The isolated latencies average four rotated session medians per depth and path. Every grouped session was faster than every stock session at each depth. The maximum absolute error against the CPU reference was below 2.2e-8 across both kernels and mask cases. [First isolated run](../results/attention-isolated-q8-20261122T003735Z.json) · [Isolated repeat](../results/attention-isolated-q8-20261122T004109Z.json)

The full-model comparison completed 16 requests on AC. All sessions reported 37/37 layers on CUDA and peak sampled VRAM was equal across paths at each depth, 4,933 MiB at 4K and 5,255 MiB at 8K. The decode gains were 3.6% and 5.3%, respectively. Both 8K fixtures and the 4K Python fixture matched greedy token IDs. At 4K the prose memo diverged at token 118 in both grouped runs. The CLI launcher also confirmed custom dispatch with a 4,629-token prompt and Q8_0 K and V. [Full-model results](../results/attention-decode-q8-20261122T003758Z.json)

## 2026-11-21 Combined grouped Q8_0 attention and adaptive n-gram

The combined runtime pairs the custom CUDA DLL with the adaptive n-gram host DLL. The preparation script checks their source revisions and verifies the hashes of both patched DLLs and their stock counterparts. Each benchmark arm uses the same executable and Qwen3 8B Q4_K_M weights. The experiment varies the attention dispatch and selects ordinary decode, ranked n-gram, or ranked n-gram with acceptance feedback.

Each run uses 8,192 prompt tokens and 128 greedy output tokens, Q8_0 K and V, one 8,448-token slot, and no prompt reuse. The workloads are the archived Python continuation, project memo, and fiction opening. The return pass reverses mode and prompt order. Both n-gram policies use the same saved FP16 verification-cost table so that enabling grouped attention is the only configuration change between each pair of speculative modes.

The grouped kernel applies to single-token passes. Multi-token verification batches use the stock dispatcher. A request's single-token pass count is 127 minus accepted draft tokens minus the number of verifications. The first output token comes from prefill, and the remaining output includes one target token per pass plus any accepted draft prefix.

```powershell
.\scripts\build-attention-bench.ps1
.\scripts\build-adaptive-ngram.ps1 -Budget
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-combined.py
.\scripts\run-model.ps1 -GroupedQ8Attention -AdaptiveNgram -Context 8448
```

The benchmark prepares the combined runtime. To prepare it without running the benchmark, use `python .\scripts\prepare_combined_runtime.py` after building its inputs. The combined launcher selects the ranked policy and Q8_0 cache.

| Workload | Stock | Grouped attention | Ranked | Attention + ranked | Feedback | Attention + feedback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repetitive Python | 27.82 | 28.75 | 81.63 | 80.87 | 81.00 | 82.01 |
| Project memo | 27.78 | 29.23 | 37.71 | 38.98 | 37.89 | 37.58 |
| Fiction opening | 27.65 | 29.16 | 25.76 | 26.56 | 26.04 | 27.18 |

Values are mean decode tokens per second over two runs. All 36 requests completed on AC and all sessions reported 37/37 CUDA layers and a sampled peak of 5,255 MiB. Sampled session temperature peaks ranged from 67 to 79 °C. [Raw tokens, timings, and pass counts](../results/attention-ngram-8k-20261122T031348Z.json)

Python accepted 96 draft tokens per request in every speculative arm. There was only one single-token pass out of 31 target passes. All modes produced identical Python tokens. The memo with grouped attention and ranked speculation had 54 single-token passes out of 80. It ran 3.4% faster than ranked alone and matched the stock output in both repeats. Ranked alone diverged at token 124 and feedback alone at token 120, so those comparisons include small differences in generated sequences.

Grouped attention alone had the highest decode rate on fiction. The combined modes accepted only two drafts each. Grouped attention on fiction diverged from stock at token 36 in both repeats, including the attention-only mode. The overall experiment reports a token mismatch. The CLI smoke test confirmed grouped dispatch, n-gram initialization, and full GPU offload when both launch switches were set.

## 2026-11-21 Time to first token and 8K prefill

The client times the loopback HTTP request through the first streamed token event. The server is already loaded and processes a 512-token warmup before measured requests. The workload is the archived project memo. The shorter prompts retain its first 128 tokens and the tail. A fresh request starts with zero cached tokens. Its immediate follow-up keeps the prefix in the same server slot and replaces the last 128 tokens with the end of the archived freeform prompt. The benchmark checks the server's cached and processed token counts on every request.

The baseline uses stock Qwen3 8B Q4_K_M on the RTX 4050 Laptop GPU, with FP16 or Q8_0 K and V, context capacity 8,448, one slot, Flash Attention enabled, batch 512, and microbatch 128. It alternates the order of prompt lengths and reverses the order of cache formats across sessions. Each result has four samples per cache format, depth, and reuse condition.

The profile traces the same loopback server and fresh 8,192-token fixture with one generated token. `nsys` records individual CUDA graph nodes. The analyzer aligns the client's request and token timestamps with the trace clock, counts attention launches to validate the window, and reports both summed kernel duration by family and the union of GPU kernel and copy intervals. The per-kernel timings come from instrumented runs. Use the unprofiled streaming benchmark for TTFT comparisons.

```powershell
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-ttft.py --microbatch 128
python .\scripts\benchmark-ttft.py --profile --microbatch 128
```

| Prompt length | Q8_0 fresh TTFT | FP16 fresh TTFT | Q8_0 reused prefix | FP16 reused prefix |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 1,603 ms | 1,603 ms | 106 ms | 106 ms |
| 4,096 | 3,436 ms | 3,441 ms | 118 ms | 110 ms |
| 8,192 | 7,181 ms | 6,785 ms | 135 ms | 127 ms |

All 48 measured requests completed on AC with 37/37 layers offloaded. Q8_0 peaked at 5,257 MiB and FP16 at 5,785 MiB. Fresh 8K Q8_0 TTFT ranged from 6.34 to 7.55 s. FP16 ranged from 6.16 to 7.29 s. The overlapping samples make the 8K cache-format timing difference uncertain. The server prompt timer accounts for roughly 7.10 s of the 7.18 s Q8_0 median at 8K. Prefix reuse cuts the model's work to 128 input tokens, with 8,064 tokens supplied by the existing slot. [Streaming timings](../results/ttft-20261122T034508Z.json)

| CUDA kernel family | Q8_0 time | Q8_0 share | FP16 time | FP16 share |
| --- | ---: | ---: | ---: | ---: |
| Quantized matrix multiplication and activation preparation | 5,064 ms | 73.5% | 5,076 ms | 78.1% |
| Flash Attention and its merge | 1,121 ms | 16.3% | 1,100 ms | 16.9% |
| Q8_0 to FP16 conversion | 260 ms | 3.8% | 0 ms | 0.0% |
| Row writes and other kernels | 449 ms | 6.5% | 323 ms | 5.0% |

Each capture contained 2,304 Flash Attention launches before merge. The number matches 36 attention layers across 64 microbatches. Q8_0 ran 4,608 Q8_0-to-FP16 conversions because stock batched attention converts K and V for its FP16 matrix instructions. CUDA kernel and copy intervals covered 6.91 s of the 7.16 s Q8_0 trace window. The rest includes gaps between GPU operations and time at the request boundaries. Both traces confirmed 37/37 layers on the GPU. [Profile summary](../results/prefill-profile-8k-20261122T034837Z.json) · [Stock K/V conversion](https://github.com/ggml-org/llama.cpp/blob/8bf88ba9368e70fe3ba1b0001b34e70ffc713716/ggml/src/ggml-cuda/fattn-common.cuh#L1084)

The largest kernels are `mul_mat_q` for Q4_K weights. Sweep physical microbatch sizes at a fixed batch of 512 first to check whether larger matrix multiplies reduce fresh TTFT within the VRAM budget. Raw traces and SQLite exports are in `%LOCALAPPDATA%\LLMOptimization\profiles`.

## 2026-11-21 8K prefill microbatch sweep

The sweep varies the physical microbatch across 128, 256, and 512 at a fixed logical batch of 512. It uses the stock server, Qwen3 8B Q4_K_M, Q8_0 K and V, one 8,448-token slot, and Flash Attention. Each session warms up on 2,048 tokens, then measures two fresh 8,192-token requests on the archived project memo. The three blocks rotate microbatch order so each setting appears once in each position. The server logs confirm the effective microbatch and full CUDA layer offload. All requests time the first streamed token and verify that the server processed the full prompt with zero cached tokens.

```powershell
. .\scripts\enter-dev.ps1
python .\scripts\benchmark-ttft.py --sweep-microbatch
```

| Microbatch | Median TTFT | TTFT range | Median prefill speed | Sampled peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 6.815 s | 6.413–7.090 s | 1,203 tok/s | 5,257 MiB |
| 256 | 6.407 s | 5.790–6.811 s | 1,280 tok/s | 5,257 MiB |
| 512 | 6.048 s | 5.552–6.421 s | 1,359 tok/s | 5,315 MiB |

Each setting has six measured requests. All 18 returned the same first token and used 37/37 GPU layers. The laptop ran on AC, and sampled session temperature peaks ranged from 62 to 75 °C. Microbatch 512 had the lowest median in every block. Its overall median was 0.77 s lower than the 128 baseline, a reduction of 11.2%, for another 58 MiB of sampled peak VRAM. That leaves 826 MiB of the reported 6,141 MiB on the device. The CUDA compute buffer grew from 45.07 MiB at 128 to 57.13 MiB at 256 and 112.26 MiB at 512. The experiment measures the setting's effect on one fresh 8K prompt, with first-token agreement as the output check. [Raw timings and session telemetry](../results/ttft-microbatch-8k-20261122T035931Z.json)

Batch 512 and microbatch 512 are the baseline for subsequent TTFT measurements and prefill profiles.

## 2026-11-21 cuBLAS prefill comparison

The comparison rebuilds `ggml-cuda.cu`, `mmq.cu`, and `mmq-few.cu` with the pinned upstream `GGML_CUDA_FORCE_CUBLAS` definition and relinks `ggml-cuda.dll` in a separate runtime. The flag disables the standard and small-batch quantized matrix multiplication routes so eligible operations fall through to cuBLAS. The benchmark checks the source revision and binary hashes, and both arms use the same `llama-server.exe`.

Each arm runs the archived fresh 8,192-token memo with Qwen3 8B Q4_K_M, a Q8_0 K/V cache, and batch and microbatch 512. The timing order is stock, cuBLAS, cuBLAS, stock, with one 2K warmup and three measured requests per server session. CUDA traces use the same 8K prompt and settings to identify GEMM and weight conversion costs.

```powershell
. .\scripts\enter-dev.ps1
python .\scripts\build-cublas.py
python .\scripts\benchmark-ttft.py --compare-cublas
python .\scripts\benchmark-ttft.py --profile --kv q8_0
python .\scripts\benchmark-ttft.py --profile --kv q8_0 --cublas
```

| Prefill path | Median TTFT | TTFT range | Median prefill speed | Sampled peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Stock quantized matmul | 5.469 s | 5.318–5.955 s | 1,500 tok/s | 5,315 MiB |
| Forced cuBLAS | 6.990 s | 6.948–7.532 s | 1,173 tok/s | 5,439 MiB |

Forced cuBLAS increased median time to first token by 1.52 s (27.8%) and peak sampled VRAM by 124 MiB. The 12 requests finished on AC with all 37 layers on the GPU. Each session peaked at 70 °C, and every request returned the same first token. The 5.469 s stock figure comes from this paired run. The earlier microbatch sweep measured 6.048 s with the same stock settings. [Raw paired timings](../results/ttft-cublas-8k-20261122T040755Z.json)

One 8K trace per arm confirmed the dispatch. The stock quantized matmul kernels and activation packing took 4.503 s. With cuBLAS, GEMMs took 3.566 s, Q4_K and Q6_K weight unpacking to FP16 took 1.524 s, and the separate FP32-to-FP16 and FP16-to-FP32 activation casts took 0.565 s. A separate gated SiLU kernel took 0.257 s in the cuBLAS trace. The GEMMs finished faster in isolation, while the surrounding work increased total prefill latency. Flash Attention took 0.985 s on stock and 1.047 s with cuBLAS. The experiment selects stock quantized matmul at batch and microbatch 512 as the TTFT reference. [Stock trace summary](../results/prefill-profile-8k-20261122T040949Z.json) · [cuBLAS trace summary](../results/prefill-profile-8k-cublas-20261122T041022Z.json)
