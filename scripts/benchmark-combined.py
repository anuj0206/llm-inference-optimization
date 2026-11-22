"""Compare grouped quantized attention and adaptive n-gram on matched 8K prompts."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

from benchmark_ngram import (
    CONTEXT_CAPACITY, NGRAM_ARGS, OUTPUT_TOKENS, PROMPT_TOKENS, TUNED_BATCH, TUNED_MICROBATCH, completion,
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
ENV_FLAGS = ("LLM_OPT_ATTENTION_Q4", "LLM_OPT_ATTENTION_Q8", "LLM_OPT_ATTENTION_MMA", "LLM_OPT_NGRAM_POLICY",
             "LLM_OPT_NGRAM_STEP_US", "LLM_OPT_NGRAM_PROBE", "LLM_OPT_NGRAM_BACKOFF")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv", choices=("q8_0", "q4_0"), default="q8_0")
    parser.add_argument("--calibration", type=Path, help="Cost report from calibrate-ngram.py.")
    options = parser.parse_args()
    tuned_q4 = options.kv == "q4_0"
    if tuned_q4 and options.calibration is None:
        parser.error("--kv q4_0 requires a --grouped-q4 calibration file.")
    project = Path(__file__).resolve().parent.parent
    root = Path(os.environ["LLM_OPT_ROOT"])
    runtime = prepare(root)
    build = json.loads((runtime / "build.json").read_text(encoding="utf-8"))
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    corpus = project / "results/ngram-prose-8k-20261121T230013Z.json"
    cost_file = options.calibration or project / "results/ngram-cost-8k-20261121T225729Z.json"
    archive = json.loads(corpus.read_text(encoding="utf-8"))["prompts"]
    prompts = {case: archive[case] for case in ("code_continuation", "prose_summary", "prose_fiction")}
    if any(len(tokens) != PROMPT_TOKENS for tokens in prompts.values()):
        raise RuntimeError("The archived prompts must each contain 8,192 tokens.")
    calibration = json.loads(cost_file.read_text(encoding="utf-8"))
    if calibration["status"] != "passed":
        raise RuntimeError("Use a completed calibration.")
    batch, microbatch = (TUNED_BATCH, TUNED_MICROBATCH) if tuned_q4 else (512, 128)
    if tuned_q4:
        if (calibration["build"] != build or not build["attention"].get("grouped_q4_sha256")
                or not calibration.get("grouped_attention") or calibration.get("cache_type") != options.kv
                or (calibration.get("batch"), calibration.get("microbatch")) != (batch, microbatch)):
            raise RuntimeError("Use the grouped Q4_0 cost report from this combined build and batch settings.")
    elif calibration["build"].get("source_commit") != build["ngram"]["source_commit"]:
        raise RuntimeError("Use completed calibration from the pinned source revision.")
    step_us = calibration["step_us"]
    costs = " ".join(map(str, step_us))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "attention-ngram-q4-8k" if tuned_q4 else "attention-ngram-8k"
    result = project / "results" / f"{label}-{stamp}.json"
    grouped_flag = "LLM_OPT_ATTENTION_" + options.kv[:2].upper()
    grouped_marker = f"enabled custom grouped {options.kv.upper()} decode attention"
    dataset = {
        "status": "running", "kv_type": options.kv, "prompt_tokens": PROMPT_TOKENS,
        "batch": batch, "microbatch": microbatch,
        "output_tokens": OUTPUT_TOKENS, "context_capacity": CONTEXT_CAPACITY,
        "prompt_source": corpus.name, "calibration": cost_file.name, "step_us": step_us,
        "prompts": prompts, "build": build,
        "model_manifest": json.loads(model.with_name(model.stem + ".manifest.json").read_text(encoding="utf-8")),
        "sessions": [],
    }
    if tuned_q4:
        dataset["power_status"] = subprocess.check_output([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "(Get-CimInstance Win32_Battery).BatteryStatus",
        ], text=True).strip()

    def save():
        partial = result.with_suffix(".partial")
        partial.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")
        partial.replace(result)

    modes = tuple(mode for mode in MODES if mode[2] != "ranked") if tuned_q4 else MODES
    try:
        for index, (mode, grouped, policy) in enumerate(modes + modes[::-1], start=1):
            environment = {key: value for key, value in os.environ.items() if key not in ENV_FLAGS}
            speculative = policy != "none"
            if grouped:
                environment[grouped_flag] = "1"
            if speculative:
                environment.update(LLM_OPT_NGRAM_POLICY=policy, LLM_OPT_NGRAM_STEP_US=costs)
            args = server_args(runtime / "bin/llama-server.exe", model,
                               "ngram-simple" if speculative else "none", cache_type=options.kv,
                               batch=batch, microbatch=microbatch)
            if speculative:
                args += NGRAM_ARGS
            log = result.with_name(f"{result.stem}-{index:02}-{mode}.log")
            session = {"mode": mode, "grouped_attention": grouped, "policy": policy, "measurements": []}
            dataset["sessions"].append(session)
            print(f"Session {index}: {mode}", flush=True)
            cases = list(prompts) if index <= len(modes) else list(reversed(prompts))
            with measured_server(args, log, environment, session) as base:
                for case in cases:
                    row = {"case": case, **completion(base, prompts[case])}
                    if row["timings"]["cache_n"] != 0:
                        raise RuntimeError(f"Expected a fresh prompt for {mode}/{case}.")
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
            if (grouped_marker in text) != grouped:
                raise RuntimeError(f"Unexpected CUDA attention dispatch: {log}")
            if (policy == "budget") != ("ngram-budget begin seq=0" in text):
                raise RuntimeError(f"Unexpected n-gram budget dispatch: {log}")
            if tuned_q4 and (not re.search(rf"\bn_batch\s*=\s*{batch}\b", text)
                             or not re.search(rf"\bn_ubatch\s*=\s*{microbatch}\b", text)
                             or "K (q4_0):" not in text or "V (q4_0):" not in text):
                raise RuntimeError(f"Effective Q4_0 cache and batches unconfirmed: {log}")
            save()
        comparisons = token_comparisons(prompts, dataset["sessions"])
        dataset["token_comparisons"] = comparisons
        if tuned_q4:
            dataset["feedback_token_comparisons"] = {
                reference: token_comparisons(
                    prompts, [session for session in dataset["sessions"] if session["mode"] in (reference, feedback)],
                    reference_mode=reference,
                )
                for reference, feedback in (("none", "budget"), ("attention", "attention-budget"))
            }
        dataset["status"] = "passed" if all(row["all_runs_match"] for row in comparisons.values()) else "token_mismatch"
    except Exception as error:
        dataset.update(status="failed", error=str(error))
        raise
    finally:
        save()
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
