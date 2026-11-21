# Isolated Q4_K benchmark result

Raw result: `q4-microbench-20261121T174016Z.json`. Run metadata: `q4-microbench-20261121T174016Z.run.json`.

Status: **passed**. Correctness cases: **36/36**.

The operation is the stock fused up/gate/SwiGLU path, N=12288, K=4096, one token. Weights come from the verified Qwen3 8B Q4_K_M file. No custom optimization is being claimed.

## Numerical validation

| Comparison | Violating values | Worst case relative L2 error |
| --- | ---: | ---: |
| up_dot | 0 | 1.61e-07 |
| gate_dot | 0 | 1.61e-07 |
| fused_vs_cpu | 0 | 2.54e-07 |
| fused_vs_unfused | 0 | 0 |
| two_warp_vs_cpu | 0 | 2.54e-07 |
| two_warp_vs_stock | 0 | 1.64e-07 |
| two_warp_vs_stock_tight_diagnostic | 3 | 1.64e-07 |
| one_warp_vs_cpu | 0 | 2.54e-07 |
| one_warp_vs_stock | 0 | 1.74e-07 |
| one_warp_vs_stock_tight_diagnostic | 1 | 1.74e-07 |

Checker negative controls detected: **72**.

Independent activation quantization check violations: **0**. Reconstruction error is recorded separately from kernel error in the raw result.

## CUDA-event timings

Values are microseconds per operation. Variability is the standard deviation of 24 batch means, each containing 108 operations after 200 warmup calls.

| Weight access | Activation path | Implementation | Mean us | Median us | Batch-mean SD us |
| --- | --- | --- | ---: | ---: | ---: |
| 36_layer_rotation | prepared_activation | one_warp_fused | 311.957 | 311.846 | 0.854 |
| 36_layer_rotation | prepared_activation | stock_fused | 308.887 | 308.826 | 0.409 |
| 36_layer_rotation | prepared_activation | two_warp_fused | 309.658 | 309.537 | 0.414 |
| 36_layer_rotation | prepared_activation | unfused_control | 315.095 | 315.084 | 0.108 |
| 36_layer_rotation | quantize_plus_operation | one_warp_fused | 313.350 | 313.221 | 0.626 |
| 36_layer_rotation | quantize_plus_operation | stock_fused | 311.463 | 311.391 | 0.221 |
| 36_layer_rotation | quantize_plus_operation | two_warp_fused | 312.030 | 311.998 | 0.289 |
| 36_layer_rotation | quantize_plus_operation | unfused_control | 318.562 | 318.545 | 0.240 |
| one_pair_reuse | prepared_activation | one_warp_fused | 309.289 | 309.305 | 0.185 |
| one_pair_reuse | prepared_activation | stock_fused | 307.500 | 307.480 | 0.172 |
| one_pair_reuse | prepared_activation | two_warp_fused | 308.195 | 308.153 | 0.194 |
| one_pair_reuse | prepared_activation | unfused_control | 315.554 | 314.690 | 3.695 |
| one_pair_reuse | quantize_plus_operation | one_warp_fused | 313.259 | 313.263 | 0.291 |
| one_pair_reuse | quantize_plus_operation | stock_fused | 311.481 | 311.438 | 0.328 |
| one_pair_reuse | quantize_plus_operation | two_warp_fused | 312.139 | 312.121 | 0.291 |
| one_pair_reuse | quantize_plus_operation | unfused_control | 318.372 | 318.322 | 0.252 |

Stock and candidate samples use the same starting layer within a trial. All six permutations of stock, two warps and one warp, four times each.

| Candidate | Weight access | Activation path | Latency change vs stock | Trials where candidate won |
| --- | --- | --- | ---: | ---: |
| one_warp_fused | 36_layer_rotation | prepared_activation | +0.99% | 0/24 |
| two_warp_fused | 36_layer_rotation | prepared_activation | +0.25% | 0/24 |
| one_warp_fused | 36_layer_rotation | quantize_plus_operation | +0.61% | 0/24 |
| two_warp_fused | 36_layer_rotation | quantize_plus_operation | +0.18% | 0/24 |
| one_warp_fused | one_pair_reuse | prepared_activation | +0.58% | 0/24 |
| two_warp_fused | one_pair_reuse | prepared_activation | +0.23% | 0/24 |
| one_warp_fused | one_pair_reuse | quantize_plus_operation | +0.57% | 0/24 |
| two_warp_fused | one_pair_reuse | quantize_plus_operation | +0.21% | 0/24 |

## CUDA launch resources

Occupancy is calculated from launch limits and compiled resource usage. It is a ceiling for resident warps.

| Warps/block | Registers/thread | Local bytes/thread | Shared bytes/block | Resident warps/SM | Occupancy ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 56 | 0 | 768 | 36 | 75.0% |
| 2 | 56 | 0 | 256 | 36 | 75.0% |
| 1 | 56 | 0 | 0 | 24 | 50.0% |

The unfused implementation is a correctness/control comparison. Stock llama.cpp already performs the fusion. The GPU-stream timer includes launch gaps between queued work; it is not a host-wall-clock or model tokens/sec measurement. Quantization-inclusive and prepared-activation cases are separate runs, so their small difference also contains clock/ordering variability.

One-pair reuse reads a 54 MiB pair; the streaming case rotates 36 distinct real layer pairs totaling 1944 MiB. Neither is the complete inference workload. See [the harness guide](../docs/q4-k-benchmark.md) for correctness bounds, cache methodology and reproduction.
