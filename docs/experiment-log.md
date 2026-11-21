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
