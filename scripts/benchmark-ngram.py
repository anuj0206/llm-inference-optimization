"""Compare unchanged Qwen3 8B with and without n-gram speculative decoding."""

import csv
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

from local_llama_server import local_server, request

PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 128
CONTEXT_CAPACITY = 8448
NGRAM_ARGS = ["--spec-ngram-simple-size-n", "6", "--spec-ngram-simple-size-m", "8"]
DRAFT_METRICS = {
    "drafted": "llamacpp:spec_decode_num_draft_tokens_total",
    "accepted": "llamacpp:spec_decode_num_accepted_tokens_total",
    "verifications": "llamacpp:spec_decode_num_drafts_total",
}


def prompts(base, project):
    def tokenize(content):
        return request(base, "/tokenize", {"content": content, "add_special": False})["tokens"]

    def fit(before, body, after, *, take_tail=False):
        prefix, middle, suffix = map(tokenize, (before, body, after))
        room = PROMPT_TOKENS - len(prefix) - len(suffix)
        if room <= 0 or len(middle) < room:
            raise RuntimeError("Prompt fixture cannot fill the required context")
        return prefix + (middle[-room:] if take_tail else middle[:room]) + suffix

    code = "".join(
        f'def get_item_{i:03}(client):\n    """Fetch one item by ID."""\n'
        f'    return client.get("/items/{i:03}")\n\n'
        for i in range(500)
    )
    chat_context = "\n\n".join((project / path).read_text(encoding="utf-8") for path in (
        "README.md", "docs/concepts.md", "docs/q4-k-benchmark.md", "docs/rtx4050-tuning.md",
    ))
    marker = "NG_BENCHMARK_CONTEXT_PLACEHOLDER"
    template = request(base, "/apply-template", {
        "messages": [{"role": "user", "content": f"Read these project notes.\n\n{marker}\n\n"
            "Write a practical design review for a colleague. In four paragraphs explain how the tests "
            "were set up, what the results mean, and what the next experiment should measure."}],
        "chat_template_kwargs": {"enable_thinking": False},
    })["prompt"]
    before, separator, after = template.partition(marker)
    if not separator:
        raise RuntimeError("Chat template did not preserve the context placeholder")
    return {
        "code_continuation": fit("# Python API wrappers\n", code, "def get_item_500", take_tail=True),
        "project_chat": fit(before, chat_context, after),
    }


def draft_counters(base):
    metrics = request(base, "/metrics", decode_json=False)
    samples = dict(line.split()[:2] for line in metrics.splitlines() if line and not line.startswith("#"))
    return {name: int(float(samples[key])) for name, key in DRAFT_METRICS.items()}


def completion(base, tokens):
    before = draft_counters(base)
    response = request(base, "/completion", {
        "prompt": tokens, "n_predict": OUTPUT_TOKENS, "temperature": 0, "seed": 42,
        "cache_prompt": False, "ignore_eos": True, "stream": False, "return_tokens": True,
    })
    timing = response["timings"]
    if (timing["prompt_n"] != PROMPT_TOKENS or timing["predicted_n"] != OUTPUT_TOKENS
            or len(response["tokens"]) != OUTPUT_TOKENS or response.get("truncated")):
        raise RuntimeError(f"Unexpected workload lengths: {timing}")
    after = draft_counters(base)
    return {
        "timings": timing, "tokens": response["tokens"], "text": response["content"],
        **{name: after[name] - before[name] for name in before},
    }


def main():
    root = Path(os.environ["LLM_OPT_ROOT"])
    project = Path(__file__).resolve().parent.parent
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    exe = root / "build/llama-cuda/bin/llama-server.exe"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = project / "results" / f"ngram-8k-{stamp}.json"
    dataset = {
        "status": "running", "prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS,
        "context_capacity": CONTEXT_CAPACITY, "kv_cache": "f16",
        "model_manifest": json.loads(model.with_name(model.stem + ".manifest.json").read_text()),
        "runtime_commit": subprocess.check_output(
            ["git", "-C", str(root / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip(),
        "sessions": [],
    }

    def save():
        partial = result.with_suffix(".partial")
        partial.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")
        partial.replace(result)

    try:
        for index, mode in enumerate(("none", "ngram-simple", "ngram-simple", "none")):
            args = [str(exe), "-m", str(model), "-ngl", "99", "-c", str(CONTEXT_CAPACITY),
                    "-np", "1", "-b", "512", "-ub", "128", "-t", "8", "-fa", "on",
                    "-ctk", "f16", "-ctv", "f16", "--fit", "off", "--metrics", "-lv", "4",
                    "--spec-type", mode]
            if mode == "ngram-simple":
                args += NGRAM_ARGS
            log_path = result.with_name(f"{result.stem}-{index + 1}-{mode}.log")
            session = {"mode": mode, "measurements": []}
            dataset["sessions"].append(session)
            print(f"Starting session {index + 1}: {mode}", flush=True)
            gpu_log = log_path.with_suffix(".gpu.log")
            with gpu_log.open("w", encoding="utf-8") as gpu_stream:
                monitor = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits", "--loop-ms=200",
                ], stdout=gpu_stream, stderr=subprocess.STDOUT)
                try:
                    with local_server(args, log_path) as (base, command):
                        session["command"] = command
                        if "prompts" not in dataset:
                            dataset["prompts"] = prompts(base, project)
                        cases = list(dataset["prompts"].items())
                        if index >= 2:
                            cases.reverse()
                        for name, tokens in cases:
                            row = {"case": name, **completion(base, tokens)}
                            if mode == "none" and row["drafted"] != 0:
                                raise RuntimeError("Stock decode unexpectedly drafted tokens")
                            session["measurements"].append(row)
                            print(f"{name}: {row['timings']['predicted_per_second']:.2f} tokens/s, "
                                  f"accepted {row['accepted']}/{row['drafted']}", flush=True)
                            save()
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=10)
            with gpu_log.open(encoding="utf-8") as samples:
                telemetry = [(int(memory), int(temp)) for memory, temp in csv.reader(samples)
                             if memory.strip().isdigit() and temp.strip().isdigit()]
            if not telemetry:
                raise RuntimeError(f"No usable GPU telemetry in {gpu_log}")
            session["sampled_peak_vram_mib"] = max(memory for memory, _ in telemetry)
            session["sampled_max_gpu_temperature_c"] = max(temp for _, temp in telemetry)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            session["offloaded_layers"] = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
            if not session["offloaded_layers"] or any(a != b for a, b in session["offloaded_layers"]):
                raise RuntimeError(f"Full GPU offload not confirmed: inspect {log_path}")

        dataset["token_comparisons"] = {}
        for name in dataset["prompts"]:
            outputs = [row["tokens"] for session in dataset["sessions"]
                       for row in session["measurements"] if row["case"] == name]
            dataset["token_comparisons"][name] = {
                "all_runs_match": all(tokens == outputs[0] for tokens in outputs),
                "first_mismatch_indices": [
                    next((i for i, (a, b) in enumerate(zip(outputs[0], tokens, strict=True)) if a != b), None)
                    for tokens in outputs
                ],
            }
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
