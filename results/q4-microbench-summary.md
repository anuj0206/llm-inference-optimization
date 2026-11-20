# Isolated Q4_K benchmark result

Raw result: `q4-microbench-20261120T165630Z.json`. Run metadata: `q4-microbench-20261120T165630Z.run.json`.

Status: **passed**. Correctness cases: **36/36**.

The operation is the stock fused up/gate/SwiGLU path, N=12288, K=4096, one token. Weights come from the verified Qwen3 8B Q4_K_M file. No custom optimization is being claimed.

## Numerical validation

| Comparison | Violating values | Worst case relative L2 error |
| --- | ---: | ---: |
| up_dot | 0 | 1.61e-07 |
| gate_dot | 0 | 1.61e-07 |
| fused_vs_cpu | 0 | 2.54e-07 |
| fused_vs_unfused | 0 | 0 |

Checker negative controls detected: **72**. These are deliberately corrupted host output values, not GPU memory-safety tests.

Independent activation quantization check violations: **0**. Reconstruction error is recorded separately from kernel error in the raw result.

## CUDA-event timings

Values are microseconds per operation; variability is the standard deviation of 20 batch means, each containing 100 operations after 200 warmup calls.

| Weight access | Activation path | Implementation | Mean us | Median us | Batch-mean SD us |
| --- | --- | --- | ---: | ---: | ---: |
| 36_layer_rotation | prepared_activation | stock_fused | 308.614 | 308.659 | 0.236 |
| 36_layer_rotation | prepared_activation | unfused_control | 315.320 | 315.305 | 0.158 |
| 36_layer_rotation | quantize_plus_operation | stock_fused | 311.467 | 311.439 | 0.186 |
| 36_layer_rotation | quantize_plus_operation | unfused_control | 319.486 | 319.534 | 0.232 |
| one_pair_reuse | prepared_activation | stock_fused | 309.622 | 309.647 | 0.332 |
| one_pair_reuse | prepared_activation | unfused_control | 315.911 | 315.863 | 0.212 |
| one_pair_reuse | quantize_plus_operation | stock_fused | 312.670 | 312.678 | 0.214 |
| one_pair_reuse | quantize_plus_operation | unfused_control | 319.944 | 319.872 | 0.228 |

The unfused implementation is a correctness/control comparison. Stock llama.cpp already performs the fusion. The GPU-stream timer includes launch gaps between queued work; it is not a host-wall-clock or model tokens/sec measurement. Quantization-inclusive and prepared-activation cases are separate runs, so their small difference also contains clock/ordering variability.

One-pair reuse reads a 54 MiB pair; the streaming case rotates 36 distinct real layer pairs totaling 1944 MiB. Neither is the complete inference workload. See [the harness guide](../docs/q4-k-benchmark.md) for correctness bounds, cache methodology and reproduction.
