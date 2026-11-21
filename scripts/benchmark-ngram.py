"""Compare unchanged Qwen3 8B with and without n-gram speculative decoding."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

from benchmark_ngram import (
    CONTEXT_CAPACITY, NGRAM_ARGS, OUTPUT_TOKENS, PROMPT_TOKENS,
    completion, measured_server, prompts, server_args, token_comparisons,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", action="store_true", help="Also compare the built adaptive runtime")
    options = parser.parse_args()
    root = Path(os.environ["LLM_OPT_ROOT"])
    project = Path(__file__).resolve().parent.parent
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    exe = root / "build/llama-cuda/bin/llama-server.exe"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "ngram-adaptive" if options.adaptive else "ngram"
    result = project / "results" / f"{label}-8k-{stamp}.json"
    dataset = {
        "status": "running", "prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS,
        "context_capacity": CONTEXT_CAPACITY, "kv_cache": "f16",
        "model_manifest": json.loads(model.with_name(model.stem + ".manifest.json").read_text()),
        "runtime_commit": subprocess.check_output(
            ["git", "-C", str(root / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip(),
        "sessions": [],
    }
    modes = ["none", "ngram-simple"]
    if options.adaptive:
        modes.append("ngram-backoff")
        candidate = root / "build/llama-adaptive-ngram"
        dataset["adaptive_build"] = json.loads((candidate / "build.json").read_text(encoding="utf-8"))
        archived = project / "results/ngram-8k-20261121T220522Z.json"
        dataset["archived_prompt_source"] = archived.name
        archived_prompts = json.loads(archived.read_text(encoding="utf-8"))["prompts"]

    def save():
        partial = result.with_suffix(".partial")
        partial.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")
        partial.replace(result)

    try:
        for index, mode in enumerate(modes + modes[::-1]):
            adaptive = mode == "ngram-backoff"
            executable = candidate / "bin/llama-server.exe" if adaptive else exe
            environment = {**os.environ, "LLM_OPT_NGRAM_BACKOFF": "1" if adaptive else "0"}
            args = server_args(executable, model, "ngram-simple" if adaptive else mode)
            if mode != "none":
                args += NGRAM_ARGS
            log_path = result.with_name(f"{result.stem}-{index + 1}-{mode}.log")
            session = {"mode": mode, "measurements": []}
            dataset["sessions"].append(session)
            print(f"Starting session {index + 1}: {mode}", flush=True)
            with measured_server(args, log_path, environment, session) as base:
                if "prompts" not in dataset:
                    dataset["prompts"] = ({**archived_prompts, **prompts(base, project, held_out=True)}
                                          if options.adaptive else prompts(base, project))
                cases = list(dataset["prompts"].items())
                if index >= len(modes):
                    cases.reverse()
                for name, tokens in cases:
                    row = {"case": name, **completion(base, tokens)}
                    if mode == "none" and row["drafted"] != 0:
                        raise RuntimeError("Stock decode unexpectedly drafted tokens")
                    session["measurements"].append(row)
                    print(f"{name}: {row['timings']['predicted_per_second']:.2f} tokens/s, "
                          f"accepted {row['accepted']}/{row['drafted']}", flush=True)
                    save()

        dataset["token_comparisons"] = token_comparisons(dataset["prompts"], dataset["sessions"])
        dataset["status"] = "passed"
    except Exception as error:
        dataset["status"] = "failed"
        dataset["error"] = str(error)
        raise
    finally:
        save()
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
