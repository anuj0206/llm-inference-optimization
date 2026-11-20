# Concepts to refresh

## Three independent model settings

Our primary model is **Qwen3 8B**, with its own **Q4_K_M weight file**, and either **FP16 or Q8_0 KV-cache storage**. The earlier Qwen3 4B file is a different model using the same weight-format recipe. Q4 does not mean four billion parameters.

- **8B**: roughly eight billion learned parameters.
- **Q4_K_M**: a mixed-format weight-quantization recipe. This file contains Q4_K matrices, Q6_K matrices and small FP32 tensors.
- **KV cache**: attention data generated while processing a request. It grows with context capacity independently of the fixed weight file.
- **Q8_1 activations**: a temporary format used by the selected matrix-vector kernel. This is independent of Q8_0 cache storage.

## Prefill, decode and context

Prefill processes input tokens in batches. Decode generates one token at a time, attending to earlier tokens. Our initial `-p 512,2048 -n 128` benchmark ran separate prefill and generation tests; it did not measure generation after both prompt lengths. The later `-p 0 -d N -n 128` experiment populated N context tokens before timing 128 generated tokens.

Allocated context can be larger than populated depth because the runtime reserves room for new tokens and rounds allocations. At depth 8192 our test allocated 8448 tokens. Do not compare cache sizes without recording actual capacity.

## The operation we isolated

For the selected feed-forward layer, a 4096-element activation vector feeds two Q4_K matrices with 12288 output rows. The operation is:

```text
up = W_up x
gate = W_gate x
out = up * SiLU(gate)
```

The stock kernel fuses both matrix-vector products and the final activation/multiply. It keeps weights packed, uses temporary Q8_1 activations and packed integer dot products, and accumulates floating-point sums. A CUDA block is a group of cooperating threads; a warp contains 32 threads on this NVIDIA GPU. The observed kernel uses four warps per block.

## Reading the evidence

- Tokens/sec measures throughput. Milliseconds/token is approximately 1000 divided by tokens/sec.
- Standard deviation across repeats describes observed variation, not uncertainty from all possible thermal/power conditions.
- Sampled device VRAM includes driver/runtime memory and can miss brief peaks. A model file's size is not total VRAM use.
- Profiler kernel-time shares locate expensive operations. They do not establish a hardware bottleneck or predict an end-to-end speedup directly.
- CUDA-event microbenchmarks time device-stream execution. They exclude model loading, tokenization, serving, sampling and many CPU costs.
- Passing numerical tests against a reference checks this operation on tested inputs. It does not establish overall model quality.
- The stock fused implementation is the performance reference. Beating an unfused control alone is not an original optimization result.
