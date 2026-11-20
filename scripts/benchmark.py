"""Run the fixed Qwen3 baseline workload and sample device-wide GPU telemetry."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("4B", "8B"), required=True)
    parser.add_argument("--power-state", choices=("AC", "battery", "unknown"), required=True)
    parser.add_argument("--context-depth", type=int, help="Benchmark only generation after prefilling this many tokens")
    parser.add_argument("--cache-type", choices=("f16", "q8_0"), default="f16")
    args = parser.parse_args()
    if args.context_depth is not None and args.context_depth < 0:
        parser.error("--context-depth must be nonnegative")
    root = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization"
    model = root / "models" / f"Qwen3-{args.size}-Q4_K_M.gguf"
    executable = root / "build/llama-cuda/bin/llama-bench.exe"
    if not model.is_file() or not executable.is_file():
        raise SystemExit("The model and CUDA benchmark executable must be installed first.")
    results = Path(__file__).resolve().parent.parent / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"baseline-{args.size.lower()}-q4" if args.context_depth is None else f"context-{args.size.lower()}-q4-d{args.context_depth}"
    if args.cache_type != "f16":
        name += f"-kv{args.cache_type}"
    prefix = results / f"{name}-{stamp}"
    output = prefix.with_suffix(".json")
    log = prefix.with_suffix(".log")
    telemetry = prefix.with_suffix(".gpu.csv")
    workload = ["-p", "512,2048"] if args.context_depth is None else ["-p", "0", "-d", str(args.context_depth)]
    command = [str(executable), "-m", str(model), "-ngl", "99",
               *workload, "-n", "128", "-r", "5", "-t", "8",
               "-b", "512", "-ub", "128", "-fa", "on", "-ctk", args.cache_type,
               "-ctv", args.cache_type, "-o", "json", "-v"]
    monitor_command = ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
                       "--format=csv,nounits", "--loop-ms=200"]
    print(f"Benchmarking {args.size} on {args.power_state}; results: {output}", flush=True)
    monitor = None
    with telemetry.open("w", encoding="utf-8") as metrics, output.open("w", encoding="utf-8") as stdout, log.open("w", encoding="utf-8") as stderr:
        try:
            monitor = subprocess.Popen(monitor_command, stdout=metrics, stderr=stderr)
            run = subprocess.run(command, stdout=stdout, stderr=stderr)
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait(timeout=10)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log_text)
    if run.returncode != 0:
        raise SystemExit(f"Benchmark failed ({run.returncode}); inspect {log}")
    measurements = json.loads(output.read_text(encoding="utf-8"))
    rows = []
    with telemetry.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            clean = {key.strip(): value.strip() for key, value in row.items() if key is not None and value is not None}
            try:
                rows.append({key: float(value) for key, value in clean.items() if key != "timestamp"})
            except ValueError:
                continue
    model_manifest = model.with_name(model.stem + ".manifest.json")
    if not model_manifest.is_file() and args.size == "4B":
        model_manifest = model.with_name("model-manifest.json")
    metadata = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "power_state": args.power_state,
        "command": command,
        "warmup": "llama-bench default enabled",
        "offloaded_layers": offload,
        "context_depth": args.context_depth,
        "cache_type": args.cache_type,
        "allocated_context_tokens": [int(n) for n in re.findall(r"\bn_ctx\s*=\s*(\d+)", log_text)],
        "cuda_kv_buffer_mib": [float(n) for n in re.findall(r"CUDA0 KV buffer size\s*=\s*([\d.]+) MiB", log_text)],
        "model_manifest": json.loads(model_manifest.read_text(encoding="utf-8")),
        "telemetry_sample_period_ms": 200,
        "telemetry_sample_count": len(rows),
        "device_peak_memory_used_mib": max((r["memory.used [MiB]"] for r in rows), default=None),
        "device_max_temperature_c": max((r["temperature.gpu"] for r in rows), default=None),
        "telemetry_note": "Device-wide sampled memory, not an allocator-exact peak; may miss sub-200 ms spikes. Includes model loading and warmup.",
    }
    prefix.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in measurements:
        label = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
        print(f"{label} at depth {row['n_depth']}: {row['avg_ts']:.2f} +/- {row['stddev_ts']:.2f} tokens/sec", flush=True)
    print(f"Offload: {offload}; sampled peak device VRAM: {metadata['device_peak_memory_used_mib']} MiB", flush=True)
