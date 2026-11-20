"""Summarize steady generation CUDA graphs from the fixed 128-token profiles."""

from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3


def analyze(path):
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    groups = connection.execute("SELECT graphId, COUNT(DISTINCT correlationId) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL GROUP BY graphId").fetchall()
    # The captured runtime splits each decode step into three graphs. Prefill
    # graphs repeat only 16 times at depth 8192; decode graphs repeat 127/128 times.
    selected = [(graph, count) for graph, count in groups if count in (127, 128)]
    if len(selected) != 3 or len({count for _, count in selected}) != 1:
        raise RuntimeError(f"Unexpected graph structure in {path}: {groups}")
    placeholders = ",".join("?" for _ in selected)
    rows = connection.execute(f"SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.graphId IN ({placeholders}) ORDER BY k.start", [graph for graph, _ in selected]).fetchall()
    totals = defaultdict(lambda: {"time_ns": 0, "calls": 0})
    families = defaultdict(int)
    for start, end, name in rows:
        totals[name]["time_ns"] += end - start
        totals[name]["calls"] += 1
        if "mul_mat_vec_q" in name:
            family = "quantized_matrix_vector"
        elif "flash_attn" in name:
            family = "attention"
        else:
            family = "other"
        families[family] += end - start
    total = sum(item["time_ns"] for item in totals.values())
    merged = []
    for start, end, _ in rows:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    span = rows[-1][1] - rows[0][0]
    union = sum(end - start for start, end in merged)
    return {
        "source": str(path), "decode_steps": selected[0][1], "selected_graphs": selected,
        "selection": "Repeated 127/128-step decode graphs only; excludes loading, warmup, prefill and the first captured decode step where needed.",
        "summed_kernel_time_ms": total / 1e6,
        "generation_window_ms": span / 1e6,
        "kernel_interval_coverage_percent": 100 * union / span,
        "family_percent_of_summed_kernel_time": {name: 100 * value / total for name, value in families.items()},
        "top_kernels": [{"name": name, "calls": value["calls"], "time_ms": value["time_ns"] / 1e6,
                         "percent_of_summed_kernel_time": 100 * value["time_ns"] / total}
                        for name, value in sorted(totals.items(), key=lambda item: item[1]["time_ns"], reverse=True)[:12]],
        "limitations": "Diagnostic tracing overhead; summed kernel shares are not end-to-end speedup estimates. No hardware counters collected; kernel names/time alone do not establish a bandwidth bottleneck.",
    }


if __name__ == "__main__":
    root = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization/profiles"
    summaries = {str(depth): analyze(sorted(root.glob(f"decode-8b-f16-d{depth}-*.sqlite"))[-1]) for depth in (0, 8192)}
    target = Path(__file__).resolve().parent.parent / "results/profile-analysis.json"
    target.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    for depth, result in summaries.items():
        print(depth, result["family_percent_of_summed_kernel_time"], "coverage", result["kernel_interval_coverage_percent"])
