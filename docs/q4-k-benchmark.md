# Isolated Q4_K correctness and performance harness

## Purpose and scope

This harness isolates our first target from Qwen3 8B Q4_K_M: two 12288-row by 4096-column matrix-vector products followed by `up * SiLU(gate)`, for one token. It provides a reproducible stock baseline and a numerical gate for future variants. It does not change llama.cpp or implement an optimized kernel yet.

Read the [concept refresher](concepts.md) and [kernel investigation](q4-k-investigation.md) first. The [latest result summary](../results/q4-microbench-summary.md) links the full machine-readable measurements.

## Reproduce

From PowerShell in the project directory, with AC power and no concurrent model runs:

```powershell
.\scripts\build-q4-bench.ps1
.\scripts\run-q4-bench.ps1
.\.venv\Scripts\python.exe scripts\summarize-q4-bench.py
```

The existing native CUDA environment and stock runtime build are prerequisites. Building prepares fixtures, verifies the entire model SHA-256, checks the pinned source commit and clean checkout, configures CMake, and compiles one harness executable. Running executes correctness before performance and writes fresh timestamped JSON plus run metadata. A failure returns nonzero and retains a JSON failure record; performance is skipped when correctness fails. Local build/run console logs may be redirected to ignored `.log` files in `results/`.

## Files and provenance

| File | Role |
| --- | --- |
| `benchmarks/q4_k/benchmark.cu` | Stock-kernel launch wrappers, scalar CPU reference, cases, event timing and JSON output |
| `benchmarks/q4_k/CMakeLists.txt` | Release CUDA build for native sm_89, existing headers and ggml-base import library |
| `scripts/prepare-q4-bench.py` | Source/model verification, exact kernel excerpts and GGUF offsets |
| `scripts/build-q4-bench.ps1` | Enter development shell and build |
| `scripts/run-q4-bench.ps1` | Check power/processes, run, record hashes and GPU state |
| `scripts/summarize-q4-bench.py` | Turn a selected/latest raw result into a readable summary |

Generated excerpts, executable, fixture and copied upstream license live under `%LOCALAPPDATA%\LLMOptimization\build\q4-microbench`. The original llama.cpp checkout remains untouched at `8bf88ba9368e70fe3ba1b0001b34e70ffc713716`.

The generator takes the exact prefix containing the stock `mul_mat_vec_q` kernel and its parameter helpers, stopping before the dedicated MoE kernel, plus the exact stock `quantize_q8_1` device kernel. It includes the existing CUDA helper headers and links the existing ggml-base host library for common host helpers. It omits unrelated backend dispatch implementations; it does not edit the selected device kernels. Excerpt structure is checked at the pinned revision, and the fixture records source and generated-file byte hashes. See [third-party notices](../benchmarks/q4_k/THIRD_PARTY_NOTICES.md).

No weights are copied into Git or duplicated into a separate fixture file. The fixture contains offsets into the existing verified GGUF model. The runner reads each layer's real up/gate pair, uploads it and releases its host storage after any required checks. All 36 device pairs occupy 1944 MiB, excluding small scratch buffers. The original 5 GB GGUF is verified when preparing a build; rebuild fixtures after any model/source replacement.

## Exact GPU operation

- Weights: unchanged packed Q4_K blocks, 256 weights per 144 bytes.
- Input: deterministic synthetic FP32 activation vector, converted by the stock CUDA quantizer to Q8_1. These activations are not captured from a real request.
- Stock fused specialization: `mul_mat_vec_q<GGML_TYPE_Q4_K,1,true>`, grid `(12288,1,1)`, block `(32,4,1)`.
- Unfused control: two stock non-fused matrix-vector launches, then a separate activation/multiply kernel using the stock SiLU helper.
- Device target: native sm_89; CUDA fast math is enabled as in the existing runtime build. The host reference uses double-precision accumulation.

The Q8_1 activation buffer is independent of Q8_0 KV-cache precision. This isolated operation has no KV cache, tokenizer, attention or server.

## Correctness design

There are 36 cases:

- Three real model layers (0, 17, 35), each tested with six activation patterns: deterministic random, zero, alternating signs/magnitudes, a single impulse, small range and large range.
- Three synthetic packed-weight patterns at N=128, K=512, each with the same six activations: zero superblock factors, maximum packed scale/minimum fields, and nonzero minima with alternating high/low nibbles and varied packed metadata.

The CPU decoder reads scalar Q4_K fields and reconstructs weight values without calling the CUDA packed-dot helper. It reconstructs the effective activation from the actual CUDA-produced Q8_1 bytes, then computes both dot products in double precision and applies a stable double-precision SiLU. This separates kernel arithmetic from the deliberate FP32-to-Q8_1 activation quantization. It does not score original unquantized-model accuracy or activation-quantization quality.

An additional independent scalar check recomputes each activation block's maximum, scale, quantized integer codes and sum from the original FP32 input. Integer codes must agree except for a one-code difference within `1e-3` of a half-integer rounding boundary, where approximate device division can affect rounding. Encoded FP16 scales/sums use a `6e-8 + 0.001*abs(expected)` comparison allowance. The harness records reconstruction error relative to the original activation separately; passing these checks is numerical validation, not a model-quality claim.

For every output row, the harness compares both unfused GPU dot products with CPU references, fused GPU output with the CPU result, and fused output with the unfused GPU control. The CPU path is independent of the device bit-unpacking/dot implementation, although both use the documented block layout and shared struct definitions.

### Bounds fixed before the passing run

For each dot product, let `L1 = sum(abs(decoded_weight * effective_activation))`:

```text
dot tolerance = 2e-5 * L1 + 2e-5
```

This accounts for cancellation without dividing by a near-zero answer. For up/gate dot bounds `tu` and `tg`, the fused output tolerance propagates their effect using a conservative 1.1 bound on the SiLU derivative:

```text
output tolerance = abs(SiLU(gate_ref))*tu
                 + 1.1*(abs(up_ref)+tu)*tg
                 + 5e-5*(1+abs(output_ref))
```

Fused-versus-unfused GPU output uses `5e-5*(1+abs(unfused_output))`. The JSON stores maximum absolute error, RMSE, relative L2 error, maximum error/bound ratio and violating-value counts. These are conservative screening bounds, not proof that every implementation inside them preserves model quality. Report actual errors alongside pass/fail; use the unchanged stock error distribution when assessing future variants.

Each case also checks that the error checker rejects a NaN and an intentionally perturbed output. These 72 negative detections exercise the checker, not the GPU memory subsystem. There is no Compute Sanitizer/memory-safety result claimed here. Future row/block changes need additional boundary shapes and memory-safety checks.

## Timing methodology

After all cases pass, the harness measures each fused variant and the unfused control with both activation paths and both weight access patterns.

- **Prepared activation:** Q8_1 conversion is outside the timer. Measures the fused kernel, or the two dot kernels plus separate SiLU/multiply control.
- **Quantization-inclusive:** each operation converts the same FP32 input before executing the selected operation. Weights remain resident; host/device upload is excluded.
- **One-pair reuse:** repeatedly executes layer 0's 54 MiB pair.
- **36-layer rotation:** round-robins 36 distinct actual pairs, a 1944 MiB weight working set. Both modes use the same fixed activation vector, unlike full inference.

Each configuration has 200 warmup operations. The current harness takes 24 CUDA-event samples of 108 operations each on one stream. The initial stock harness took 20 samples of 100 operations each. Results include the sample batch means, their mean and median, standard deviation and p95. The p95 describes the distribution of batch means. Timing configuration order is shuffled with seed 42 and stored in output order.

CUDA events measure elapsed device-stream time including gaps between submitted launches. They do not measure model loading, serving, tokenization, sampling, per-token inference, or host-wall-clock latency. The stock runtime's CUDA-graph replay is not reproduced by these direct standalone launches. Profile-derived per-kernel latency and event-batched timing have different overheads.

The run wrapper records AC power, UTC timestamps, runner/executable hashes and GPU temperature/clocks/power snapshots before and after execution. These snapshots are not continuous thermal telemetry. Short runs and a deterministic ordering do not establish thermal steady state or eliminate ordering effects; a future candidate should be compared with interleaved/reversed stock runs across sessions.

## Initial validation result

The first successful run is `q4-microbench-20261120T165156Z.json` with matching `.run.json`. All 36 cases passed, all 72 deliberately corrupted outputs were detected, and fused/unfused GPU outputs were identical on the tested cases. Worst per-case fused-versus-CPU relative L2 error was approximately `2.54e-7`. Large-range cases can have a larger absolute error because outputs are large; inspect normalized errors and bounds together.

The subsequent run `q4-microbench-20261120T165630Z.json` added the independent activation-quantizer checks and also passed all 36 cases with zero quantization-check violations. Both successful runs are retained.

Stock fused prepared-activation timing was about 309 us for both access modes, close to the earlier 308 us model-profile observation. The unfused control was about 315 us. The production runtime already uses fusion. See the [baseline run](../results/q4-microbench-20261120T165630Z.json) for all eight timing rows and variability.

## Two-warp experiment

The harness now benchmarks a separately named, fused two-warp variant alongside the stock four-warp kernel. The generator creates the variant from the pinned stock kernel body by changing its symbol name, its launch bound and its compile-time warp count. The harness supplies the matching block dimensions, `(32, 2, 1)`. Both variants process one output row per block with the same weight and activation formats and the same fused operation.

After building, open the exact device-code diff with:

```powershell
$build = Join-Path $env:LOCALAPPDATA 'LLMOptimization\build\q4-microbench'
code --diff (Join-Path $build 'stock_mmvq_kernel.cuh') (Join-Path $build 'two_warp_mmvq.cuh')
```

The original 36 cases now check both fused variants against the same CPU reference and per-row tolerance. The direct cross-check uses twice that tolerance because each variant has its own rounding error. The original, tighter stock-relative comparison is still recorded as a diagnostic. Its initial run flagged three values in the large-activation case for layer 35 even though both versions passed the CPU checks for all 36 cases.

The comparison warms up both kernels, then alternates which one runs first in each of 20 pairs of samples. Both samples in a pair use the same starting layer. Each sample contains 108 calls, exactly three full rotations when using all 36 layers. The unfused control is timed separately.

Both timed runs passed all 36 correctness cases. The rotating-layer comparison gave the following mean microseconds per call.

| Run | Work included | Four warps | Two warps |
| --- | --- | ---: | ---: |
| [First run](../results/q4-microbench-20261121T173118Z.json) | Prepared activation | 311.23 | 310.38 |
| First run | Activation quantization + fused operation | 313.04 | 316.01 |
| [Repeat](../results/q4-microbench-20261121T173140Z.json) | Prepared activation | 312.30 | 314.21 |
| Repeat | Activation quantization + fused operation | 311.53 | 312.37 |

The small gain with prepared activations in the first run reversed on repeat. Including quantization, two warps were slower in both runs. The experiment stays in the isolated harness. The [initial correctness report](../results/q4-microbench-20261121T173006Z.json) preserves the three tighter diagnostic disagreements.

The harness now also builds a one-warp variant. The six possible orders run four times each, for 24 samples per variant. The result records CUDA function resources and theoretical occupancy. See the [RTX 4050 tuning notes](rtx4050-tuning.md) for the three-way measurement. To inspect the one-warp diff, substitute `one_warp_mmvq.cuh` for `two_warp_mmvq.cuh` above.

## Two rows per block

This variant uses four warps per block, matching stock. The generated device code sets `rows_per_cuda_block = 2`, and the host wrapper launches half as many blocks. At the model's width of 12,288 output rows that yields 6,144 blocks. Each block computes both gate and up dot products for each of two rows and can reuse activation loads. The host wrapper requires an even row count because the kernel reads both rows as a pair.

The benchmark checks the candidate against the CPU and stock outputs on all 36 cases before timing. The four fused variants run in every possible order once, for 24 matched samples of 108 calls. For the exact device-code diff, substitute `two_row_mmvq.cuh` in the command above.

Both runs on AC passed 36/36 cases. The two-row candidate had zero measured difference from stock on the checked outputs. Its worst per-case relative L2 error against the CPU was `2.54e-7`.

| 36-layer rotation | Stock (µs) | Two rows (µs) | Latency change |
| --- | ---: | ---: | ---: |
| [First run](../results/q4-microbench-20261121T182537Z.json), prepared activation | 308.665 | 309.341 | +0.22% |
| [Repeat](../results/q4-microbench-20261121T182608Z.json), prepared activation | 308.734 | 309.439 | +0.23% |
| First run, activation quantization included | 311.866 | 312.335 | +0.15% |
| Repeat, activation quantization included | 311.347 | 311.841 | +0.16% |

The extra accumulators raised compiled register use from 56 to 64 per thread and shared memory from 768 to 1,536 bytes per block. CUDA calculated a residency ceiling of eight blocks per SM (32 warps, 66.7% occupancy), down from nine blocks (36 warps, 75%) for stock. This is a launch-resource bound. With slightly higher latency in both runs, the candidate stays in the isolated harness and inference uses the stock kernel.

## In-process memory counters

The optional counter build links NVIDIA CUPTI and uses the toolkit's Range Profiling API. It runs the same correctness cases before profiling the stock fused kernel directly. It measures one launch on layer 17 with an already quantized Q8_1 activation. Each replay pass restores the cache lead-in by running the other 35 layers in order or by running the target pair 35 times. CUPTI cache flushing is disabled so the two lead-ins can produce different cache states. The lead-in launches are outside the measured range.

Build from normal PowerShell. On systems with restricted GPU counters, run the last command from administrator PowerShell under the project user's account. The build requires only the additional `cuda_cupti` component. [NVIDIA counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)

```powershell
.\.venv\Scripts\python.exe scripts\install_cuda.py --components cuda_cupti
.\scripts\build-q4-bench.ps1 -Counters
.\scripts\run-q4-bench.ps1 -Counters
```

The output records DRAM read/write bytes, L2 read sectors and sector hit rate, and L1 global load sectors and requests. The unique packed weight payload is 56,623,104 bytes, plus 4,608 bytes of unique Q8_1 activations. Compare DRAM reads against that payload in the rotating case. Compare L2 sector bytes and L1 sectors per request to investigate repeated cached loads or transaction overhead. These counters aggregate the measured range. An excess over the weight payload can include activation traffic, overlap between requests, and extra sectors from alignment. Identifying the contributing load instructions requires a source or instruction-level check.

The regular-user run passed all 36 correctness cases before CUPTI reported insufficient privileges. The elevated run passed the same checks and collected both ranges in two counter passes each. [Raw counter report](../results/q4-traffic-20261121T192402Z.json)

| Layer 17 cache lead-in | DRAM read | L2 read sectors × 32 B | L1 global-load sectors × 32 B | L1 sectors/request |
| --- | ---: | ---: | ---: | ---: |
| Other 35 layers | 54.010 MiB | 54.405 MiB | 324 MiB | 6 |
| Same pair 35 times | 54.030 MiB | 54.429 MiB | 324 MiB | 6 |

With rotation, DRAM read 10,880 bytes above the 54 MiB weight payload, or 0.019%. L2 read sector volume was 0.75% above the payload. Repeating the same pair gave similar totals. These aggregate measurements leave little evidence of extra DRAM transfers in this launch. The 324 MiB figure counts L1 sector requests, including repeated sector touches. [NVIDIA memory counter definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#l1tex-cache)

Disassembly and the Q4_K dot helper account for all 1,769,472 L1 global-load requests. Each of 12,288 CUDA blocks has four warps, two inner-loop iterations and 18 load instructions per warp per iteration. Six instructions load Q8_1 activation fields shared by both dot products. The other 12 load the up and gate weights. In each iteration, each weight matrix contributes two loads for packed values and four for metadata fields. The three scale/minimum loads and one superblock factor load address the same 16-byte header within each Q4_K block. A candidate can load that header together and unpack the existing fields from registers. Compare L1 requests, register use and elapsed time against stock under the same output checks.
