"""Compare ranked, rolling, and corpus-based drafts on fresh greedy prose prompts."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct

from benchmark_ngram import (
    CONTEXT_CAPACITY, NGRAM_ARGS, OUTPUT_TOKENS, PROMPT_TOKENS,
    completion, measured_server, prompts, server_args, token_comparisons,
)
from local_llama_server import request

CORPUS_PROMPTS = (
    "Explain the difference between GPU memory capacity and memory bandwidth to a new developer.",
    "Write a short note about why improving an L1 cache metric might leave application performance unchanged.",
    "Summarize the tradeoffs of reducing the precision of an inference KV cache.",
    "Explain to a developer why processing a prompt and generating one token use the GPU differently.",
    "Describe how you would test whether a matrix-vector kernel is limited by memory traffic.",
    "Write a short onboarding memo that explains greedy decoding and how to check for output regressions.",
    "Describe how a long conversation can change the memory usage and speed of a local language model.",
    "Explain why a microbenchmark can improve while application throughput remains flat.",
    "Explain in plain English how to tell whether a throughput improvement is within measurement noise.",
    "Summarize an experiment that compares two inference settings while holding model weights fixed.",
    "Explain the difference between CUDA throughput and the time a user waits for the first token.",
    "Write a short note on why a benchmark should log VRAM, GPU offload and wall-clock time.",
)


def build_static_cache(base, path):
    corpus = []
    counts = defaultdict(Counter)
    for instruction in CORPUS_PROMPTS:
        prompt = request(base, "/apply-template", {
            "messages": [{"role": "user", "content": instruction}],
            "chat_template_kwargs": {"enable_thinking": False},
        })["prompt"]
        response = request(base, "/completion", {
            "prompt": prompt, "n_predict": OUTPUT_TOKENS, "temperature": 0, "seed": 42,
            "cache_prompt": False, "stream": False, "return_tokens": True,
        })
        tokens = response["tokens"]
        if len(tokens) < 3:
            raise RuntimeError(f"Corpus response too short for {instruction!r}")
        corpus.append({"instruction": instruction, "tokens": tokens})
        for a, b, token in zip(tokens, tokens[1:], tokens[2:]):
            counts[a, b][token] += 1
    # Pinned common_ngram_cache_save layout on Windows x64. Static entries use two tokens.
    with path.open("xb") as stream:
        for (a, b), votes in sorted(counts.items()):
            stream.write(struct.pack("<5i", a, b, -1, -1, len(votes)))
            for token, count in sorted(votes.items()):
                stream.write(struct.pack("<2i", token, count))
    return {
        "bytes": path.stat().st_size, "keys": len(counts),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "responses": corpus,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    options = parser.parse_args()
    root = Path(os.environ["LLM_OPT_ROOT"])
    project = Path(__file__).resolve().parent.parent
    stock = root / "build/llama-cuda/bin/llama-server.exe"
    candidate = root / "build/llama-adaptive-ngram"
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    build = json.loads((candidate / "build.json").read_text(encoding="utf-8"))
    calibration = json.loads(options.calibration.read_text(encoding="utf-8"))
    if calibration["status"] != "passed" or calibration["build"] != build:
        raise RuntimeError("Use completed calibration from the current adaptive build.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = project / "results" / f"ngram-prose-8k-{stamp}.json"
    cache = candidate / f"prose-cache-{stamp}.bin"
    dataset = {
        "status": "running", "prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS,
        "context_capacity": CONTEXT_CAPACITY, "kv_cache": "f16",
        "model_manifest": json.loads(model.with_name(model.stem + ".manifest.json").read_text(encoding="utf-8")),
        "build": build, "calibration": options.calibration.name, "step_us": calibration["step_us"],
        "sessions": [],
    }
    modes = ["none", "backoff", "ranked", "rolling", "cache-empty", "cache-seeded"]

    def save():
        partial = result.with_suffix(".partial")
        partial.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")
        partial.replace(result)

    try:
        for index, mode in enumerate(modes + modes[::-1], start=1):
            is_cache = mode.startswith("cache-")
            environment = dict(os.environ)
            for key in ("LLM_OPT_NGRAM_POLICY", "LLM_OPT_NGRAM_PROBE", "LLM_OPT_NGRAM_BACKOFF"):
                environment.pop(key, None)
            if mode in ("ranked", "rolling"):
                environment["LLM_OPT_NGRAM_POLICY"] = mode
                environment["LLM_OPT_NGRAM_STEP_US"] = " ".join(map(str, dataset["step_us"]))
            if mode == "backoff":
                environment["LLM_OPT_NGRAM_BACKOFF"] = "1"
            if mode == "none" or is_cache:
                args = server_args(stock, model, "ngram-cache" if is_cache else "none")
                if mode == "cache-seeded":
                    args += ["--lookup-cache-static", str(cache)]
            else:
                args = server_args(candidate / "bin/llama-server.exe", model, "ngram-simple") + NGRAM_ARGS
            cases = list(dataset.get("prompts", {}))
            if index > len(modes):
                cases.reverse()
            # The pinned cache retains context statistics across requests in a server slot.
            groups = [[case] for case in cases] if is_cache else [cases]
            for group in groups:
                log_suffix = f"-{group[0]}" if is_cache else ""
                log = result.with_name(f"{result.stem}-{index:02}-{mode}{log_suffix}.log")
                session = {"mode": mode, "pass": index, "measurements": []}
                dataset["sessions"].append(session)
                print(f"Session {index}: {mode}{log_suffix}", flush=True)
                with measured_server(args, log, environment, session) as base:
                    if "prompts" not in dataset:
                        dataset["prompts"] = prompts(base, project, fresh=True)
                        group = list(dataset["prompts"])
                    for case in group:
                        row = {"case": case, **completion(base, dataset["prompts"][case])}
                        if mode == "none" and row["drafted"] != 0:
                            raise RuntimeError("Stock decode unexpectedly drafted tokens.")
                        session["measurements"].append(row)
                        print(f"  {case}: {row['timings']['predicted_per_second']:.2f} tokens/s, "
                              f"accepted {row['accepted']}/{row['drafted']}", flush=True)
                        save()
                    if "cache_corpus" not in dataset:
                        print("Building the static cache from separate Qwen3 responses.", flush=True)
                        dataset["cache_corpus"] = build_static_cache(base, cache)
                save()

        comparisons = token_comparisons(dataset["prompts"], dataset["sessions"])
        dataset["token_comparisons"] = comparisons
        dataset["status"] = "passed" if all(row["all_runs_match"] for row in comparisons.values()) else "token_mismatch"
    except Exception as error:
        dataset["status"] = "failed"
        dataset["error"] = str(error)
        raise
    finally:
        save()
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
