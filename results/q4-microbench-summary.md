# Isolated Q4_K benchmark result

Raw result: `q4-microbench-20261121T182608Z.json`. Run metadata: `q4-microbench-20261121T182608Z.run.json`.

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
| two_row_vs_cpu | 0 | 2.54e-07 |
| two_row_vs_stock | 0 | 0 |
| two_row_vs_stock_tight_diagnostic | 0 | 0 |

Checker negative controls detected: **72**.

Independent activation quantization check violations: **0**. Reconstruction error is recorded separately from kernel error in the raw result.

## CUDA-event timings

Values are microseconds per operation. Variability is the standard deviation of 24 batch means, each containing 108 operations after 200 warmup calls.

| Weight access | Activation path | Implementation | Mean us | Median us | Batch-mean SD us |
| --- | --- | --- | ---: | ---: | ---: |
| 36_layer_rotation | prepared_activation | one_warp_fused | 311.644 | 311.761 | 1.102 |
| 36_layer_rotation | prepared_activation | stock_fused | 308.734 | 308.850 | 0.608 |
| 36_layer_rotation | prepared_activation | two_row_fused | 309.439 | 309.480 | 0.729 |
| 36_layer_rotation | prepared_activation | two_warp_fused | 309.484 | 309.556 | 0.706 |
| 36_layer_rotation | prepared_activation | unfused_control | 315.083 | 315.060 | 0.155 |
| 36_layer_rotation | quantize_plus_operation | one_warp_fused | 313.357 | 313.268 | 0.314 |
| 36_layer_rotation | quantize_plus_operation | stock_fused | 311.347 | 311.353 | 0.133 |
| 36_layer_rotation | quantize_plus_operation | two_row_fused | 311.841 | 311.780 | 0.224 |
| 36_layer_rotation | quantize_plus_operation | two_warp_fused | 312.083 | 312.017 | 0.398 |
| 36_layer_rotation | quantize_plus_operation | unfused_control | 318.686 | 318.639 | 0.242 |
| one_pair_reuse | prepared_activation | one_warp_fused | 309.272 | 309.262 | 0.181 |
| one_pair_reuse | prepared_activation | stock_fused | 307.456 | 307.428 | 0.144 |
| one_pair_reuse | prepared_activation | two_row_fused | 307.792 | 307.769 | 0.116 |
| one_pair_reuse | prepared_activation | two_warp_fused | 308.162 | 308.139 | 0.137 |
| one_pair_reuse | prepared_activation | unfused_control | 314.874 | 314.757 | 0.418 |
| one_pair_reuse | quantize_plus_operation | one_warp_fused | 312.853 | 312.789 | 0.252 |
| one_pair_reuse | quantize_plus_operation | stock_fused | 311.207 | 311.149 | 0.215 |
| one_pair_reuse | quantize_plus_operation | two_row_fused | 313.042 | 311.628 | 6.896 |
| one_pair_reuse | quantize_plus_operation | two_warp_fused | 311.947 | 311.936 | 0.351 |
| one_pair_reuse | quantize_plus_operation | unfused_control | 318.418 | 318.407 | 0.190 |

Stock and candidate samples use the same starting layer within a trial. All 24 permutations of stock, two warps, one warp and two rows, once each.

| Candidate | Weight access | Activation path | Latency change vs stock | Trials where candidate won |
| --- | --- | --- | ---: | ---: |
| one_warp_fused | 36_layer_rotation | prepared_activation | +0.94% | 0/24 |
| two_warp_fused | 36_layer_rotation | prepared_activation | +0.24% | 0/24 |
| two_row_fused | 36_layer_rotation | prepared_activation | +0.23% | 0/24 |
| one_warp_fused | 36_layer_rotation | quantize_plus_operation | +0.65% | 0/24 |
| two_warp_fused | 36_layer_rotation | quantize_plus_operation | +0.24% | 0/24 |
| two_row_fused | 36_layer_rotation | quantize_plus_operation | +0.16% | 0/24 |
| one_warp_fused | one_pair_reuse | prepared_activation | +0.59% | 0/24 |
| two_warp_fused | one_pair_reuse | prepared_activation | +0.23% | 0/24 |
| two_row_fused | one_pair_reuse | prepared_activation | +0.11% | 2/24 |
| one_warp_fused | one_pair_reuse | quantize_plus_operation | +0.53% | 0/24 |
| two_warp_fused | one_pair_reuse | quantize_plus_operation | +0.24% | 0/24 |
| two_row_fused | one_pair_reuse | quantize_plus_operation | +0.59% | 1/24 |

## CUDA launch resources

Occupancy is calculated from launch limits and compiled resource usage. It is a ceiling for resident warps.

| Rows/block | Warps/block | Registers/thread | Local bytes/thread | Shared bytes/block | Resident blocks/SM | Resident warps/SM | Occupancy ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 56 | 0 | 768 | 9 | 36 | 75.0% |
| 1 | 2 | 56 | 0 | 256 | 18 | 36 | 75.0% |
| 1 | 1 | 56 | 0 | 0 | 24 | 24 | 50.0% |
| 2 | 4 | 64 | 0 | 1536 | 8 | 32 | 66.7% |

The unfused implementation is a correctness/control comparison. Stock llama.cpp already performs the fusion. The GPU-stream timer includes launch gaps between queued work; it is not a host-wall-clock or model tokens/sec measurement. Quantization-inclusive and prepared-activation cases are separate runs, so their small difference also contains clock/ordering variability.

One-pair reuse reads a 54 MiB pair; the streaming case rotates 36 distinct real layer pairs totaling 1944 MiB. Neither is the complete inference workload. See [the harness guide](../docs/q4-k-benchmark.md) for correctness bounds, cache methodology and reproduction.
