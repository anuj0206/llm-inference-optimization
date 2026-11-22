"""Measure the cost of a target pass with zero through eight rejected draft tokens."""

import argparse
import json
import os
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path
from statistics import median

from benchmark_ngram import NGRAM_ARGS, OUTPUT_TOKENS, TUNED_BATCH, TUNED_MICROBATCH, completion, server_args
from local_llama_server import local_server
from prepare_combined_runtime import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-q4", action="store_true", help="Calibrate the combined Q4_0 runtime with tuned batching.")
    options = parser.parse_args()
    root = Path(os.environ["LLM_OPT_ROOT"])
    project = Path(__file__).resolve().parent.parent
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    candidate = prepare(root) if options.grouped_q4 else root / "build/llama-adaptive-ngram"
    build = json.loads((candidate / "build.json").read_text(encoding="utf-8"))
    if options.grouped_q4 and not build["attention"].get("grouped_q4_sha256"):
        raise RuntimeError("Rebuild attention with grouped Q4_0 before calibration.")
    executable = candidate / "bin/llama-server.exe"
    source = project / "results/ngram-adaptive-8k-20261121T223052Z.json"
    tokens = json.loads(source.read_text(encoding="utf-8"))["prompts"]["prose_briefing"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "ngram-cost-grouped-q4-8k" if options.grouped_q4 else "ngram-cost-8k"
    result = project / "results" / f"{label}-{stamp}.json"
    dataset = {
        "status": "running", "prompt_source": source.name, "case": "prose_briefing",
        "build": build,
        "sessions": [],
    }
    server_settings = {}
    if options.grouped_q4:
        server_settings = {"cache_type": "q4_0", "batch": TUNED_BATCH, "microbatch": TUNED_MICROBATCH}
        dataset.update(grouped_attention=True, **server_settings)
    costs = [[] for _ in range(9)]
    try:
        for index, length in enumerate((0, 1, 8, 2, 7, 3, 6, 4, 5, 0), start=1):
            environment = dict(os.environ)
            for key in ("LLM_OPT_NGRAM_POLICY", "LLM_OPT_NGRAM_PROBE", "LLM_OPT_NGRAM_BACKOFF",
                        "LLM_OPT_ATTENTION_Q4", "LLM_OPT_ATTENTION_Q8", "LLM_OPT_ATTENTION_MMA"):
                environment.pop(key, None)
            if options.grouped_q4:
                environment["LLM_OPT_ATTENTION_Q4"] = "1"
            args = server_args(executable, model, "ngram-simple" if length else "none", **server_settings)
            if length:
                environment["LLM_OPT_NGRAM_PROBE"] = str(length)
                args += NGRAM_ARGS + ["--spec-synth-rates", ",".join(["0"] * 8)]
            log = result.with_name(f"{result.stem}-{index:02}-{length}.log")
            with local_server(args, log, environment=environment) as (base, command):
                row = completion(base, tokens)
            decision_passes = OUTPUT_TOKENS - 1
            expected_drafted = sum(min(length, remaining) for remaining in range(decision_passes))
            if (row["accepted"] or row["drafted"] != expected_drafted
                    or row["verifications"] != (decision_passes - 1 if length else 0)):
                raise RuntimeError(f"Unexpected forced-draft counters at length {length}: {row}")
            server_log = log.read_text(encoding="utf-8", errors="replace")
            if "offloaded 37/37 layers to GPU" not in server_log:
                raise RuntimeError(f"Full GPU offload not confirmed: {log}")
            if options.grouped_q4 and length == 0 and "enabled custom grouped Q4_0 decode attention" not in server_log:
                raise RuntimeError(f"Grouped serial reference did not dispatch: {log}")
            step_us = row["timings"]["predicted_ms"] * 1000 / decision_passes
            costs[length].append(step_us)
            dataset["sessions"].append({
                "draft_length": length, "command": command, "step_us": step_us,
                **{key: row[key] for key in ("timings", "drafted", "accepted", "verifications")},
            })
            print(f"{length} draft tokens: {step_us / 1000:.2f} ms/pass", flush=True)
        dataset["step_us"] = [round(value) for value in accumulate(map(median, costs), max)]
        dataset["status"] = "passed"
    except Exception as error:
        dataset["status"] = "failed"
        dataset["error"] = str(error)
        raise
    finally:
        result.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
