"""Write a readable summary of a completed isolated Q4_K benchmark run."""

import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("result", nargs="?", type=Path)
args = parser.parse_args()
path = args.result or sorted(p for p in (root / "results").glob("q4-microbench-20*.json") if not p.name.endswith(".run.json"))[-1]
data = json.loads(path.read_text(encoding="utf-8-sig"))
checks = data["correctness"]
lines = ["# Isolated Q4_K benchmark result", "", f"Raw result: `{path.name}`. Run metadata: `{path.stem}.run.json`.", "",
         f"Status: **{data['status']}**. Correctness cases: **{sum(c['passed'] for c in checks)}/{len(checks)}**.", "",
         "The operation is the stock fused up/gate/SwiGLU path, N=12288, K=4096, one token. Weights come from the verified Qwen3 8B Q4_K_M file. No custom optimization is being claimed.", "",
         "## Numerical validation", "", "| Comparison | Violating values | Worst case relative L2 error |", "| --- | ---: | ---: |"]
for key in ("up_dot", "gate_dot", "fused_vs_cpu", "fused_vs_unfused", "two_warp_vs_cpu", "two_warp_vs_stock",
            "two_warp_vs_stock_tight_diagnostic", "one_warp_vs_cpu", "one_warp_vs_stock",
            "one_warp_vs_stock_tight_diagnostic", "two_row_vs_cpu", "two_row_vs_stock",
            "two_row_vs_stock_tight_diagnostic"):
    if not all(key in case for case in checks):
        continue
    lines.append(f"| {key} | {sum(c[key]['failures'] for c in checks)} | {max(c[key]['relative_l2_error'] for c in checks):.3g} |")
if not data["timings"]:
    raise SystemExit(f"No timing data in {path}. See the raw result for the failure.")
sample = data["timings"][0]
lines += ["", f"Checker negative controls detected: **{sum(c['negative_controls_detected'] for c in checks)}**.", "",
          "## CUDA-event timings", "", f"Values are microseconds per operation. Variability is the standard deviation of {len(sample['samples_us'])} batch means, each containing {sample['calls_per_sample']} operations after {sample['warmup_calls']} warmup calls.", "",
          "| Weight access | Activation path | Implementation | Mean us | Median us | Batch-mean SD us |", "| --- | --- | --- | ---: | ---: | ---: |"]
if all("activation_quantization" in c for c in checks):
    position = lines.index("## CUDA-event timings")
    lines[position:position] = [f"Independent activation quantization check violations: **{sum(c['activation_quantization']['failures'] for c in checks)}**. Reconstruction error is recorded separately from kernel error in the raw result.", ""]
for row in sorted(data["timings"], key=lambda r: (r["mode"], r["path"], r["implementation"])):
    lines.append(f"| {row['mode']} | {row['path']} | {row['implementation']} | {row['mean_us']:.3f} | {row['median_us']:.3f} | {row['stddev_batch_mean_us']:.3f} |")
if "fused_sample_order" in data:
    lines += ["", "Stock and candidate samples use the same starting layer within a trial. " + data["fused_sample_order"] + ".", "",
              "| Candidate | Weight access | Activation path | Latency change vs stock | Trials where candidate won |",
              "| --- | --- | --- | ---: | ---: |"]
    timings = {(r["mode"], r["path"], r["implementation"]): r for r in data["timings"]}
    for (mode, path_name, implementation), stock in sorted(timings.items()):
        if implementation != "stock_fused":
            continue
        for label in ("one_warp_fused", "two_warp_fused", "two_row_fused"):
            candidate = timings.get((mode, path_name, label))
            if candidate is None:
                continue
            change = 100 * (candidate["mean_us"] / stock["mean_us"] - 1)
            wins = sum(a < b for a, b in zip(candidate["samples_us"], stock["samples_us"], strict=True))
            lines.append(f"| {label} | {mode} | {path_name} | {change:+.2f}% | {wins}/{len(stock['samples_us'])} |")
if "launch_resources" in data:
    lines += ["", "## CUDA launch resources", "", "Occupancy is calculated from launch limits and compiled resource usage. It is a ceiling for resident warps.", "",
              "| Rows/block | Warps/block | Registers/thread | Local bytes/thread | Shared bytes/block | Resident blocks/SM | Resident warps/SM | Occupancy ceiling |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["launch_resources"]:
        lines.append(
            f"| {row.get('rows_per_block', 1)} | {row['warps_per_block']} | {row['registers_per_thread']} | "
            f"{row['local_bytes_per_thread']} | {row['shared_bytes_per_block']} | "
            f"{row['max_resident_blocks_per_sm']} | {row['max_resident_warps_per_sm']} | {row['theoretical_occupancy']:.1%} |"
        )
lines += ["", "The unfused implementation is a correctness/control comparison. Stock llama.cpp already performs the fusion. The GPU-stream timer includes launch gaps between queued work; it is not a host-wall-clock or model tokens/sec measurement. Quantization-inclusive and prepared-activation cases are separate runs, so their small difference also contains clock/ordering variability.", "",
          "One-pair reuse reads a 54 MiB pair; the streaming case rotates 36 distinct real layer pairs totaling 1944 MiB. Neither is the complete inference workload. See [the harness guide](../docs/q4-k-benchmark.md) for correctness bounds, cache methodology and reproduction.", ""]
summary = root / "results/q4-microbench-summary.md"
summary.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
