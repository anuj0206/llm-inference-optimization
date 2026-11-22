"""Compare grouped Q8_0 attention and adaptive n-gram on matched 8K prompts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from benchmark_ngram import (
    CONTEXT_CAPACITY, NGRAM_ARGS, OUTPUT_TOKENS, PROMPT_TOKENS, completion,
    measured_server, server_args, token_comparisons,
)
from prepare_combined_runtime import prepare

MODES = (
    ("none", False, "none"),
    ("attention", True, "none"),
    ("ranked", False, "ranked"),
    ("attention-ranked", True, "ranked"),
    ("budget", False, "budget"),
    ("attention-budget", True, "budget"),
)
GROUPED_MARKER = "enabled custom grouped Q8_0 decode attention"
ENV_FLAGS = ("LLM_OPT_ATTENTION_Q8", "LLM_OPT_ATTENTION_MMA", "LLM_OPT_NGRAM_POLICY",
             "LLM_OPT_NGRAM_STEP_US", "LLM_OPT_NGRAM_PROBE", "LLM_OPT_NGRAM_BACKOFF")


def main():
    project = Path(__file__).resolve().parent.parent
    root = Path(os.environ["LLM_OPT_ROOT"])
    runtime = prepare(root)
    build = json.loads((runtime / "build.json").read_text(encoding="utf-8"))
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    corpus = project / "results/ngram-prose-8k-20261121T230013Z.json"
    cost_file = project / "results/ngram-cost-8k-20261121T225729Z.json"
    archive = json.loads(corpus.read_text(encoding="utf-8"))["prompts"]
    prompts = {case: archive[case] for case in ("code_continuation", "prose_summary", "prose_fiction")}
    if any(len(tokens) != PROMPT_TOKENS for tokens in prompts.values()):
        raise RuntimeError("The archived prompts must each contain 8,192 tokens.")
    calibration = json.loads(cost_file.read_text(encoding="utf-8"))
    if (calibration["status"] != "passed" or
            calibration["build"]["source_commit"] != build["ngram"]["source_commit"]):
        raise RuntimeError("Use completed calibration from the pinned source revision.")
    step_us = calibration["step_us"]
    costs = " ".join(map(str, step_us))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = project / "results" / f"attention-ngram-8k-{stamp}.json"
    dataset = {
        "status": "running", "kv_type": "q8_0", "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS, "context_capacity": CONTEXT_CAPACITY,
        "prompt_source": corpus.name, "calibration": cost_file.name, "step_us": step_us,
        "prompts": prompts, "build": build,
        "model_manifest": json.loads(model.with_name(model.stem + ".manifest.json").read_text(encoding="utf-8")),
        "sessions": [],
    }

    def save():
        partial = result.with_suffix(".partial")
        partial.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")
        partial.replace(result)

    try:
        for index, (mode, grouped, policy) in enumerate(MODES + MODES[::-1], start=1):
            environment = {key: value for key, value in os.environ.items() if key not in ENV_FLAGS}
            speculative = policy != "none"
            if grouped:
                environment["LLM_OPT_ATTENTION_Q8"] = "1"
            if speculative:
                environment.update(LLM_OPT_NGRAM_POLICY=policy, LLM_OPT_NGRAM_STEP_US=costs)
            args = server_args(runtime / "bin/llama-server.exe", model,
                               "ngram-simple" if speculative else "none", cache_type="q8_0")
            if speculative:
                args += NGRAM_ARGS
            log = result.with_name(f"{result.stem}-{index:02}-{mode}.log")
            session = {"mode": mode, "grouped_attention": grouped, "policy": policy, "measurements": []}
            dataset["sessions"].append(session)
            print(f"Session {index}: {mode}", flush=True)
            cases = list(prompts) if index <= len(MODES) else list(reversed(prompts))
            with measured_server(args, log, environment, session) as base:
                for case in cases:
                    row = {"case": case, **completion(base, prompts[case])}
                    # Prefill emits the first token. Every decode pass emits one target token
                    # plus any verified draft prefix.
                    row["target_passes"] = OUTPUT_TOKENS - 1 - row["accepted"]
                    row["single_token_passes"] = row["target_passes"] - row["verifications"]
                    if (row["single_token_passes"] < 0 or (not speculative and row["drafted"] != 0)
                            or (speculative and case == "code_continuation" and row["accepted"] == 0)):
                        raise RuntimeError(f"Unexpected speculative counters for {mode}/{case}: {row}")
                    session["measurements"].append(row)
                    print(f"  {case}: {row['timings']['predicted_per_second']:.2f} tok/s, "
                          f"accepted {row['accepted']}/{row['drafted']}", flush=True)
                    save()
            text = log.read_text(encoding="utf-8", errors="replace")
            if (GROUPED_MARKER in text) != grouped:
                raise RuntimeError(f"Unexpected CUDA attention dispatch: {log}")
            if (policy == "budget") != ("ngram-budget begin seq=0" in text):
                raise RuntimeError(f"Unexpected n-gram budget dispatch: {log}")
            save()
        comparisons = token_comparisons(prompts, dataset["sessions"])
        dataset["token_comparisons"] = comparisons
        dataset["status"] = "passed" if all(row["all_runs_match"] for row in comparisons.values()) else "token_mismatch"
    except Exception as error:
        dataset.update(status="failed", error=str(error))
        raise
    finally:
        save()
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
