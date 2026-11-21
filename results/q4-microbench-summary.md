# Isolated Q4_K benchmark result

Raw result: `q4-microbench-20261121T173140Z.json`. Run metadata: `q4-microbench-20261121T173140Z.run.json`.

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

Checker negative controls detected: **72**.

Independent activation quantization check violations: **0**. Reconstruction error is recorded separately from kernel error in the raw result.

## CUDA-event timings

Values are microseconds per operation. Variability is the standard deviation of 20 batch means, each containing 108 operations after 200 warmup calls.

| Weight access | Activation path | Implementation | Mean us | Median us | Batch-mean SD us |
| --- | --- | --- | ---: | ---: | ---: |
| 36_layer_rotation | prepared_activation | stock_fused | 312.299 | 309.281 | 10.314 |
| 36_layer_rotation | prepared_activation | two_warp_fused | 314.214 | 309.978 | 13.158 |
| 36_layer_rotation | prepared_activation | unfused_control | 315.978 | 315.776 | 0.749 |
| 36_layer_rotation | quantize_plus_operation | stock_fused | 311.532 | 311.443 | 0.281 |
| 36_layer_rotation | quantize_plus_operation | two_warp_fused | 312.371 | 312.353 | 0.377 |
| 36_layer_rotation | quantize_plus_operation | unfused_control | 319.036 | 319.014 | 0.145 |
| one_pair_reuse | prepared_activation | stock_fused | 309.232 | 309.243 | 0.354 |
| one_pair_reuse | prepared_activation | two_warp_fused | 309.854 | 309.722 | 0.532 |
| one_pair_reuse | prepared_activation | unfused_control | 315.827 | 315.781 | 0.470 |
| one_pair_reuse | quantize_plus_operation | stock_fused | 312.374 | 312.339 | 0.265 |
| one_pair_reuse | quantize_plus_operation | two_warp_fused | 312.892 | 312.927 | 0.239 |
| one_pair_reuse | quantize_plus_operation | unfused_control | 319.557 | 319.540 | 0.372 |

Each stock/candidate pair starts at the same layer. The benchmark alternates which variant runs first.

| Weight access | Activation path | Two-warp latency change vs stock | Pairs where two warps won |
| --- | --- | ---: | ---: |
| 36_layer_rotation | prepared_activation | +0.61% | 0/20 |
| 36_layer_rotation | quantize_plus_operation | +0.27% | 0/20 |
| one_pair_reuse | prepared_activation | +0.20% | 0/20 |
| one_pair_reuse | quantize_plus_operation | +0.17% | 1/20 |

The unfused implementation is a correctness/control comparison. Stock llama.cpp already performs the fusion. The GPU-stream timer includes launch gaps between queued work; it is not a host-wall-clock or model tokens/sec measurement. Quantization-inclusive and prepared-activation cases are separate runs, so their small difference also contains clock/ordering variability.

One-pair reuse reads a 54 MiB pair; the streaming case rotates 36 distinct real layer pairs totaling 1944 MiB. Neither is the complete inference workload. See [the harness guide](../docs/q4-k-benchmark.md) for correctness bounds, cache methodology and reproduction.
