"""Measure the cost of a target pass with zero through eight rejected draft tokens."""

import json
import os
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path
from statistics import median

from benchmark_ngram import NGRAM_ARGS, OUTPUT_TOKENS, completion, server_args
from local_llama_server import local_server


def main():
    root = Path(os.environ["LLM_OPT_ROOT"])
    project = Path(__file__).resolve().parent.parent
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    candidate = root / "build/llama-adaptive-ngram"
    executable = candidate / "bin/llama-server.exe"
    source = project / "results/ngram-adaptive-8k-20261121T223052Z.json"
    tokens = json.loads(source.read_text(encoding="utf-8"))["prompts"]["prose_briefing"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = project / "results" / f"ngram-cost-8k-{stamp}.json"
    dataset = {
        "status": "running", "prompt_source": source.name, "case": "prose_briefing",
        "build": json.loads((candidate / "build.json").read_text(encoding="utf-8")),
        "sessions": [],
    }
    costs = [[] for _ in range(9)]
    try:
        for index, length in enumerate((0, 1, 8, 2, 7, 3, 6, 4, 5, 0), start=1):
            environment = dict(os.environ)
            for key in ("LLM_OPT_NGRAM_POLICY", "LLM_OPT_NGRAM_PROBE", "LLM_OPT_NGRAM_BACKOFF"):
                environment.pop(key, None)
            args = server_args(executable, model, "ngram-simple" if length else "none")
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
            if "offloaded 37/37 layers to GPU" not in log.read_text(encoding="utf-8", errors="replace"):
                raise RuntimeError(f"Full GPU offload not confirmed: {log}")
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
