# Tuning the Q4_K decode kernel for the RTX 4050 Laptop GPU

## Hardware on this laptop

CUDA reported compute capability 8.9 (Ada), 20 streaming multiprocessors, 24 MiB of L2 and a 96-bit memory interface. The driver reports approximately 6 GiB of VRAM. The build already targets native `sm_89`. The 8,001 MHz peak GDDR memory clock and 96-bit interface imply a theoretical peak of 192 GB/s. During one active sample in the repeat run, `nvidia-smi` reported 8,001 MHz memory, 2,100 MHz SM clock and 53.21 W. An idle snapshot reported a 45 W current power limit and a 90 W maximum. These snapshots describe individual moments. They do not establish sustained clocks or throttling. NVIDIA lists the 4050 Laptop SKU with 2,560 CUDA cores. [NVIDIA laptop specifications](https://www.nvidia.com/en-us/geforce/laptops/compare/) · [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)

## One-warp experiment

The candidate compiles the same fused body with one warp and launches 32 threads to produce one output row. Stock uses four warps and 128 threads. Two warps provide a third point in the same run. Each run gives every variant 200 warmup calls and 24 timed samples of 108 calls. Every ordering of the three variants occurs four times, and the variants start at the same layer within each trial. Both runs passed all 36 cases with the CPU and stock bounds from the earlier experiment.

| Rotating all 36 layer pairs | Four warps | Two warps | One warp |
| --- | ---: | ---: | ---: |
| First run, prepared activation | 308.75 µs | 309.59 µs | 311.86 µs |
| Repeat, prepared activation | 308.89 µs | 309.66 µs | 311.96 µs |
| First run, quantization included | 311.45 µs | 312.37 µs | 313.29 µs |
| Repeat, quantization included | 311.46 µs | 312.03 µs | 313.35 µs |

The one-warp variant added about 1% to the isolated kernel timing and 0.6% to the quantization-inclusive path. One value exceeded the tighter stock-relative diagnostic, while all cases passed the CPU and stock bounds. The inference build continues to use four warps. [First run](../results/q4-microbench-20261121T173949Z.json) · [Repeat and resource data](../results/q4-microbench-20261121T174016Z.json)

All three fused kernels compile with 56 registers per thread and zero local bytes per thread. Four and two warps can each have 36 resident warps per SM by the CUDA occupancy calculation. One warp hits the Ada limit of 24 resident blocks per SM, which allows just 24 resident warps. This lowers the theoretical occupancy ceiling from 75% to 50%. Fewer warps can make it harder to hide memory latency. The occupancy API does not measure achieved occupancy or establish why the kernel slowed down. The one-warp disassembly still contains a block barrier, even though it allocates zero shared memory for partial sums. The experiment tests warp count alone with the existing barrier in place. [Ada occupancy limits](https://docs.nvidia.com/cuda/ada-tuning-guide/#occupancy)

## What to tune for this hardware

1. **Measure the weight loads first.** Each call consumes two packed matrices totaling 54 MiB, more than twice this GPU's L2 capacity. The isolated stock run processed about 183 GB/s of nominal packed weights, obtained by dividing 54 MiB by 309 µs. That ratio is close to the 192 GB/s theoretical peak and measures neither DRAM traffic nor hardware utilization. Use a sustained streaming reference and, where available, Nsight Compute metrics for DRAM read throughput, L2 sectors and stalls. This helps distinguish bandwidth pressure from inefficient memory transactions or unpacking overhead.
2. **Tested two output rows per block at four warps.** The rows share the same activation inputs and each needs its own packed weights and accumulators. The 36-layer rotation was about 0.2% slower in two runs, and calculated residency fell from 36 to 32 warps per SM. See the [two-row timings and code diff command](q4-k-benchmark.md#two-rows-per-block).
3. **Test a weight layout only if memory transactions leave room.** The stock format interleaves 128 bytes of packed values and 16 bytes of metadata per block. A separate, aligned payload layout is a concrete hypothesis for reducing transactions or simplifying loads. Any repack must account for peak VRAM and load cost, especially at 8K context.
4. **Evaluate Tensor Cores for prefill or batched decode.** Ada supports FP8 Tensor Core operations. The inspected batch-one Q4_K kernel compiles to packed integer dot products (`IDP.4A`) and uses Q4_K weights with their group metadata. A Tensor Core path would require a suitable operand layout and scale handling. Prefill and multi-token batches reuse each matrix across more activation columns, giving that work a clearer place to start. [NVIDIA Ada Tensor Core guidance](https://docs.nvidia.com/cuda/ada-tuning-guide/#improved-tensor-core-operations)

To reproduce the comparison, build and run the isolated harness as described in the [harness guide](q4-k-benchmark.md). To inspect the generated instructions, use the toolkit's `cuobjdump --dump-sass` on the harness executable and locate the fused `mul_mat_vec_q` variants.
