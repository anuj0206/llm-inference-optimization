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

## Streaming-read bandwidth

Bitdefender's Advanced Threat Defense stopped both Nsight Compute runs before counter collection. The streaming-read test uses CUDA events and runs as a regular user with protection enabled.

From the project directory on AC power, run `.\scripts\run-stream-read.ps1`. It builds the CUDA program for `sm_89` with the installed toolkit. The kernel issues aligned 16-byte reads per thread through L2 and writes one XOR checksum per warp. The host checks the result against the input before timing each size. Input initialization, host-to-device copy and checksum readback are outside the timed batches. Each size gets 64 warmup launches and 12 timed batches of 64 launches. The test visits 64, 256 and 512 MiB in ascending and then descending order. All three sizes exceed the laptop GPU's 24 MiB L2.

| Input buffer | Forward median | Reverse median |
| --- | ---: | ---: |
| 64 MiB | 182.62 GB/s | 185.82 GB/s |
| 256 MiB | 187.45 GB/s | 187.45 GB/s |
| 512 MiB | 187.69 GB/s | 187.72 GB/s |

The 64 MiB forward samples ranged from 162.19 to 185.91 GB/s. The run recorded memory clocks of 7,001 MHz before the benchmark and 8,001 MHz afterward. Those snapshots do not establish clock behavior during individual samples. The 256 and 512 MiB results were consistent in both size orders. [Raw samples](../results/stream-read-20261121T191057Z.csv) · [Run metadata](../results/stream-read-20261121T191057Z.run.json)

The stock fused Q4_K run processed 54 MiB of packed weights in 308.73 µs, or 183.40 GB/s of nominal weight throughput. That is about 98% of the 512 MiB streaming-read reference. The comparison supports bandwidth pressure as an explanation for the small gains from launch changes. CUDA event timing gives effective throughput from bytes requested and elapsed time. Actual DRAM bytes, cache traffic and stall breakdowns remain unmeasured. [Stock kernel run](../results/q4-microbench-20261121T182608Z.json)

## What to tune for this hardware

1. **Profile the weight loads when counters are available.** The [streaming-read reference](#streaming-read-bandwidth) puts stock nominal weight throughput within about 2% of a coalesced read on this laptop. DRAM read throughput, L2 sectors and stalls would help distinguish bandwidth pressure from inefficient memory transactions or unpacking overhead.
2. **Tested two output rows per block at four warps.** The rows share the same activation inputs and each needs its own packed weights and accumulators. The 36-layer rotation was about 0.2% slower in two runs, and calculated residency fell from 36 to 32 warps per SM. See the [two-row timings and code diff command](q4-k-benchmark.md#two-rows-per-block).
3. **Test a weight layout only if memory transactions leave room.** The stock format interleaves 128 bytes of packed values and 16 bytes of metadata per block. A separate, aligned payload layout is a concrete hypothesis for reducing transactions or simplifying loads. Any repack must account for peak VRAM and load cost, especially at 8K context.
4. **Evaluate Tensor Cores for prefill or batched decode.** Ada supports FP8 Tensor Core operations. The inspected batch-one Q4_K kernel compiles to packed integer dot products (`IDP.4A`) and uses Q4_K weights with their group metadata. A Tensor Core path would require a suitable operand layout and scale handling. Prefill and multi-token batches reuse each matrix across more activation columns, giving that work a clearer place to start. [NVIDIA Ada Tensor Core guidance](https://docs.nvidia.com/cuda/ada-tuning-guide/#improved-tensor-core-operations)

To reproduce the comparison, build and run the isolated harness as described in the [harness guide](q4-k-benchmark.md). To inspect the generated instructions, use the toolkit's `cuobjdump --dump-sass` on the harness executable and locate the fused `mul_mat_vec_q` variants.
