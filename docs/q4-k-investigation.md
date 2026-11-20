# Q4_K decode kernel investigation

Model: Qwen3 8B Q4_K_M, pinned model manifest in `results/target-model.json`. Runtime source remains unmodified at `8bf88ba9368e70fe3ba1b0001b34e70ffc713716`. Investigation uses the existing FP16-cache profiles; it does not change weight or cache precision.

## What actually runs

Q4_K_M is the model-file quantization recipe. The active CUDA matrix-vector specialization is `mul_mat_vec_q<GGML_TYPE_Q4_K, 1, ...>` in `ggml/src/ggml-cuda/mmvq.cu`. The model also uses Q6_K matrices; the M recipe does not mean all tensors are Q4_K.

The decode path reads an FP32 activation vector, temporarily quantizes it to Q8_1, then performs packed Q4_K-by-Q8_1 dot products. The Q8_1 activation format is independent of Q8_0 KV-cache storage. The dot-product helper unpacks weight nibbles and packed scales/minima, uses packed integer dot-product instructions (`ggml_cuda_dp4a`), applies scale/minimum corrections, accumulates in floating point and reduces partial sums. It does not expand the full weight matrix to FP16 first.

Q4_K stores 256 weights in a 144-byte block: 128 bytes of 4-bit values, 12 bytes of packed scales/minima, and two FP16 superblock factors. That is 4.5 stored bits per weight for Q4_K blocks, including their metadata.

## Measured first target

The clearest operation to isolate is a feed-forward up/gate pair followed by SwiGLU:

```text
input activation x: 4096 values
up   = W_up   x       W_up:   12288 output rows x 4096 input columns
gate = W_gate x       W_gate: 12288 output rows x 4096 input columns
output = up * SiLU(gate)
```

The GGUF inventory contains 36 Q4_K up matrices and 36 Q4_K gate matrices of this shape. The Q4_K fusion-enabled profile launch has grid `(12288,1,1)` and appears 36 times per decoded token, matching the layer count. The model builder selects parallel SiLU feed-forward operations, and the CUDA graph optimizer supports their combined up/gate/SwiGLU kernel. This mapping is inferred from tensor shapes, counts and source; the original tensor pointers were not instrumented.

| Observation | Short context | 8192 populated context |
| --- | ---: | ---: |
| Mean profiled kernel latency | about 307.9 us | about 308.6 us |
| Calls per decoded token | 36 | 36 |
| Share of summed selected GPU kernel time | about 41.3% | about 32.8% |

The approximately 59% short-context figure for the largest fusion-enabled Q4_K specialization includes other launch shapes too. It must not be attributed entirely to this feed-forward shape.

## Thread mapping on this RTX 4050

The Ada target uses the generic MMVQ parameter table. For the observed one-token Q4_K path, the launch is `(32,4,1)`: four warps, 128 CUDA threads per block, one output row per block. Nsight records 56 registers per thread for these variants; that is a compiler allocation observation, not a measurement of achieved occupancy.

Each warp contributes partial dot-product sums. The implementation combines warps through shared memory and performs a warp reduction before writing the output. For the gate/up operation, it maintains two sets of accumulators and applies the existing fused activation/multiply step. Software prefetch in this source is enabled for a different architecture (GB10), not this Ada target.

One matrix occupies 27 MiB of packed Q4_K blocks; the pair occupies 54 MiB. The CUDA driver reports 24 MiB of L2 cache, a 96-bit memory bus and an 8001 MHz peak memory-clock attribute, implying a theoretical 192.024 GB/s peak bandwidth. Weight-pair bytes divided by the approximately 308 us profiled latency is about 184 GB/s. That derived ratio is not measured DRAM bandwidth: caching, repeated reads, actual clocks, instruction costs and tracing overhead all matter. These facts suggest limited bandwidth headroom may be important, but hardware counters are required to establish the bottleneck.

## First standalone benchmark specification

1. Preserve the current GGUF weight layout and Q8_1 activation quantization, and benchmark the fused up/gate/SwiGLU operation with N=12288, K=4096 and one token. Include an unfused reference for correctness, not as the sole performance comparison.
2. Compare the stock fused implementation with any proposed variant. Time both kernel-only execution on already-quantized activations and the activation-quantization-plus-kernel path separately.
3. Verify outputs against dequantized/reference calculations, including numerical error statistics and adversarial scales, minima, sign patterns and activation ranges. Establish tolerances from the unchanged implementation before judging a variant.
4. Use CUDA-event timing after warmup, repeated runs and both cache-reuse and streaming weight sets. A repeatedly accessed tiny matrix can give an unrealistically favorable cache result. This weight pair exceeds L2, but its full-model reuse distance is larger still.
5. Make one change at a time. Initial candidates are warps per block (2/4/8), rows per block (1/2), and shared-memory versus shuffle reduction organization. They are hypotheses; fewer threads or fewer synchronizations do not guarantee lower latency. Preserve the existing fusion.
6. Only integrate a correct, repeatable improvement, then rerun short- and 8K-context generation against the preserved stock build. Kernel-only gains are not end-to-end claims.

No new kernel has been implemented, and no speedup is claimed by this investigation. The next concrete coding task is the focused correctness/performance harness around the stock fused operation.

## Source map at the pinned revision

- `src/models/qwen3.cpp`: creates the FFN tensors and builds the SiLU feed-forward graph.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: graph fusion and dispatch.
- `ggml/src/ggml-cuda/mmvq.cu`: architecture parameter table, block/thread mapping, fused matrix-vector kernel and Q8_1 activation preparation.
- `ggml/src/ggml-cuda/vecdotq.cuh`: `vec_dot_q4_K_q8_1` and its packed dot-product helper.
- `ggml/src/ggml-common.h`: `block_q4_K` layout.

Run `.\.venv\Scripts\python.exe scripts\inspect-q4-k.py` to reproduce the read-only tensor inventory, driver attribute query and profile-launch analysis. Detailed observations are saved in `results/q4-k-investigation.json`.
