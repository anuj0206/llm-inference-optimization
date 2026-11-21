"""Compare top target probabilities at the first greedy mismatch in a prose run."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from benchmark_ngram import NGRAM_ARGS, OUTPUT_TOKENS, draft_counters, server_args
from local_llama_server import local_server, request


def first_difference(reference, candidate):
    return next((i for i, (a, b) in enumerate(zip(reference, candidate, strict=True)) if a != b), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    data = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if data["status"] not in ("passed", "token_mismatch"):
        raise SystemExit("Use a completed prose benchmark.")
    reference = next(s for s in data["sessions"] if s["mode"] == "none")
    ranked = next(s for s in data["sessions"] if s["mode"] == "ranked")
    mismatch = next(((a, b) for a, b in zip(reference["measurements"], ranked["measurements"], strict=True)
                     if a["tokens"] != b["tokens"]), None)
    if mismatch is None:
        raise SystemExit("No ranked mismatch to inspect in the forward pass.")
    stock_row, ranked_row = mismatch
    case = stock_row["case"]
    tokens = data["prompts"][case]
    root = Path(os.environ["LLM_OPT_ROOT"])
    candidate = root / "build/llama-adaptive-ngram"
    if json.loads((candidate / "build.json").read_text(encoding="utf-8")) != data["build"]:
        raise RuntimeError("Use the same adaptive build as the benchmark.")
    executable = candidate / "bin/llama-server.exe"
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = args.benchmark.with_name(f"ngram-parity-8k-{stamp}.json")
    findings = {"benchmark": args.benchmark.name, "case": case, "sessions": []}
    for mode, probe in (("stock", 0), ("ranked", 0), ("reject-one", 1), ("reject-eight", 8)):
        environment = dict(os.environ)
        for key in ("LLM_OPT_NGRAM_POLICY", "LLM_OPT_NGRAM_PROBE", "LLM_OPT_NGRAM_BACKOFF"):
            environment.pop(key, None)
        command = server_args(executable, model, "none" if mode == "stock" else "ngram-simple")
        if mode != "stock":
            command += NGRAM_ARGS
        if mode == "ranked":
            environment["LLM_OPT_NGRAM_POLICY"] = "ranked"
            environment["LLM_OPT_NGRAM_STEP_US"] = " ".join(map(str, data["step_us"]))
        if probe:
            environment["LLM_OPT_NGRAM_PROBE"] = str(probe)
            command += ["--spec-synth-rates", ",".join(["0"] * 8)]
        with local_server(command, result.with_name(f"{result.stem}-{mode}.log"), environment=environment) as (base, actual):
            response = request(base, "/completion", {
                "prompt": tokens, "n_predict": OUTPUT_TOKENS, "temperature": 0, "seed": 42,
                "cache_prompt": False, "ignore_eos": True, "stream": False, "return_tokens": True,
                "n_probs": 4, "post_sampling_probs": False,
            })
            counts = draft_counters(base)
        if probe and (counts["accepted"] != 0 or counts["drafted"] == 0):
            raise RuntimeError(f"Synthetic rejection failed for {mode}: {counts}")
        probs = response["completion_probabilities"]
        observed = response["tokens"]
        if len(observed) != OUTPUT_TOKENS or len(probs) != OUTPUT_TOKENS:
            raise RuntimeError(f"Unexpected completion length for {mode}.")
        if mode == "stock":
            stock_tokens, stock_probs = observed, probs
        index = first_difference(stock_tokens, observed)
        expected = {"stock": stock_row["tokens"], "ranked": ranked_row["tokens"]}.get(mode)
        findings["sessions"].append({
            "mode": mode, "command": actual, "first_difference_zero_based": index,
            "matches_benchmark": observed == expected if expected is not None else None,
            "stock_probabilities": stock_probs[index] if index is not None else None,
            "candidate_probabilities": probs[index] if index is not None else None,
            "stock_token": stock_tokens[index] if index is not None else None,
            "candidate_token": observed[index] if index is not None else None,
            **counts,
        })
        print(f"{mode}: first different token index {index} ({counts['accepted']} accepted drafts)", flush=True)
    result.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
