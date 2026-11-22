"""Compare stock and grouped CUDA attention at 4K and 8K."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import time

from benchmark_ngram import completion, measured_server, server_args

GROUPED = {
    "f16": ("grouped_mma", "LLM_OPT_ATTENTION_MMA", "enabled grouped FP16 Tensor Core decode attention"),
    "q8_0": ("grouped_q8", "LLM_OPT_ATTENTION_Q8", "enabled custom grouped Q8_0 decode attention"),
}


@contextmanager
def exclusive_gpu(path):
    # The elevated counter worker can start while a normal timing run is in progress.
    with path.open("a+b") as stream:
        stream.seek(0)
        deadline = time.monotonic() + 600
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Another attention benchmark still holds the GPU lock.")
                time.sleep(1.0)
        try:
            yield
        finally:
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def benchmark(options, candidate):
    project = Path(__file__).resolve().parent.parent
    root = Path(os.environ["LLM_OPT_ROOT"])
    bin_dir = candidate / "bin"
    build = json.loads((candidate / "build.json").read_text(encoding="utf-8"))
    if hashlib.sha256((bin_dir / "ggml-cuda.dll").read_bytes()).hexdigest() != build["experiment_cuda_sha256"]:
        raise RuntimeError("The experimental CUDA DLL changed after the build.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"attention-{options.measurement}" + ("-q8" if options.kv == "q8_0" else "")
    result = project / "results" / f"{label}-{stamp}.json"
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    dataset = {"status": "running", "measurement": options.measurement, "kv_type": options.kv,
               "build": build, "sessions": []}
    if options.measurement == "decode":
        fixture = project / "results/ngram-prose-8k-20261121T230013Z.json"
        archived = json.loads(fixture.read_text(encoding="utf-8"))["prompts"]
        prompts = {case: archived[case] for case in ("code_continuation", "prose_summary")}
        dataset["prompt_source"] = fixture.name
        dataset["short_prompt_method"] = "Keep the first 128 and last 3968 tokens."
        dataset["model_manifest"] = json.loads(model.with_name(model.stem + ".manifest.json").read_text(encoding="utf-8"))
    else:
        dataset["harness_sha256"] = hashlib.sha256((bin_dir / "attention-bench.exe").read_bytes()).hexdigest()

    grouped, enable_grouped, marker = GROUPED[options.kv]
    schedule = [(depth, mode) for depth in (4096, 8192) for mode in ("vector", grouped)]
    try:
        for index, (depth, mode) in enumerate(schedule + schedule[::-1], start=1):
            environment = dict(os.environ)
            for _, flag, _ in GROUPED.values():
                environment.pop(flag, None)
            if mode == grouped:
                environment[enable_grouped] = "1"
            log = result.with_name(f"{result.stem}-{index:02}-{depth}-{mode}.log")
            row = {"depth": depth, "path": mode}
            dataset["sessions"].append(row)
            print(f"{index}: {depth} {mode}", flush=True)
            if options.measurement == "decode":
                args = server_args(bin_dir / "llama-server.exe", model, "none",
                                   context_capacity=depth + 256, cache_type=options.kv)
                row["measurements"] = []
                with measured_server(args, log, environment, row) as base:
                    cases = list(prompts) if index <= len(schedule) else list(reversed(prompts))
                    for case in cases:
                        tokens = prompts[case]
                        if depth != len(tokens):
                            tokens = tokens[:128] + tokens[-(depth - 128):]
                        measurement = completion(base, tokens, prompt_length=depth)
                        row["measurements"].append({"case": case, **measurement})
                        print(f"  {case}: {measurement['timings']['predicted_per_second']:.2f} tok/s", flush=True)
                if (marker in log.read_text(encoding="utf-8", errors="replace")) != (mode == grouped):
                    raise RuntimeError(f"Unexpected grouped-dispatch marker: {log}")
            else:
                scratch = candidate / f"{result.stem}-{index:02}.json"
                args = [str(bin_dir / "attention-bench.exe"), str(scratch), str(depth), mode, options.kv]
                if options.measurement == "counters":
                    args.append("--counters")
                with log.open("w", encoding="utf-8") as stream:
                    run = subprocess.run(args, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
                if run.returncode:
                    log_text = log.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(f"Attention harness failed ({run.returncode}): {log_text[-1500:]}")
                row.update(json.loads(scratch.read_text(encoding="utf-8")))
                if options.measurement == "isolated":
                    print(f"  rotated: {row['rotated']['median_us']:.2f} us", flush=True)
                else:
                    reads = row["traffic"][0]["counters"]["dram__bytes_read.sum"]
                    print(f"  rotated: {reads / 2**20:.2f} MiB DRAM reads", flush=True)

        if options.measurement == "decode":
            parity = {}
            for depth in (4096, 8192):
                sessions = [row for row in dataset["sessions"] if row["depth"] == depth]
                for case in prompts:
                    outputs = [m["tokens"] for row in sessions for m in row["measurements"] if m["case"] == case]
                    parity[f"{depth}:{case}"] = all(tokens == outputs[0] for tokens in outputs)
            dataset["token_parity"] = parity
        dataset["status"] = "passed"
    except Exception as error:
        dataset.update(status="failed", error=str(error))
        raise
    finally:
        result.write_text(json.dumps(dataset, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"Saved {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", choices=("isolated", "counters", "decode"))
    parser.add_argument("--kv", choices=tuple(GROUPED), default="f16")
    options = parser.parse_args()
    candidate = Path(os.environ["LLM_OPT_ROOT"]) / "build/llama-attention"
    with exclusive_gpu(candidate / "benchmark.lock"):
        benchmark(options, candidate)


if __name__ == "__main__":
    main()
