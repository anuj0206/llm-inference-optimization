"""Measure time to the first streamed token on an already loaded Qwen3 8B."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
import urllib.request

from analyze_prefill import analyze
from benchmark_ngram import CONTEXT_CAPACITY, measured_server, server_args
from local_llama_server import local_server


PROJECT = Path(__file__).resolve().parent.parent
FIXTURE = PROJECT / "results/ngram-prose-8k-20261121T230013Z.json"
DEPTHS = (2048, 4096, 8192)
KV_TYPES = ("q8_0", "f16")
NEW_TOKENS = 128


def first_token(base, tokens, *, cached_tokens=0):
    payload = json.dumps({
        "prompt": tokens, "n_predict": 1, "temperature": 0, "seed": 42,
        "cache_prompt": bool(cached_tokens), "ignore_eos": True,
        "stream": True, "return_tokens": True,
    }).encode()
    request = urllib.request.Request(base + "/completion", data=payload, headers={
        "Content-Type": "application/json", "Accept": "text/event-stream",
    })
    first = None
    final = None
    started_utc = time.time_ns()
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=120) as response:
        for line in response:
            received = time.perf_counter_ns()
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            if first is None and event.get("tokens") and not event.get("stop"):
                first = (received - started, event["tokens"][0], event["content"])
            if event.get("stop"):
                final = event

    if first is None or final is None:
        raise RuntimeError("The stream did not contain a token event and final timings.")
    timings = final["timings"]
    if (final["truncated"] or final["tokens_evaluated"] != len(tokens)
            or timings["prompt_n"] != len(tokens) - cached_tokens
            or timings["cache_n"] != cached_tokens or timings["predicted_n"] != 1):
        raise RuntimeError(f"Unexpected request lengths: {timings}")
    elapsed, token, content = first
    return {
        "ttft_ms": elapsed / 1e6, "server_timings": timings, "first_token": token,
        "first_content": content, "request_started_utc_ns": started_utc,
        "first_token_utc_ns": started_utc + elapsed,
    }


def prompt_pair(archived, depth):
    original = archived["prose_summary"]
    base = original[:128] + original[-(depth - 128):]
    suffix = archived["prose_freeform"][-NEW_TOKENS:]
    if suffix[0] == base[-NEW_TOKENS]:
        raise RuntimeError("The follow-up must diverge at the first suffix token.")
    return base, base[:-NEW_TOKENS] + suffix


def timing_summary(sessions, depths, cache_types):
    result = []
    for kv in cache_types:
        for depth in depths:
            for mode in ("fresh", "reused_prefix"):
                rows = [row for session in sessions if session["kv_type"] == kv
                        for row in session["measurements"] if (row["depth"], row["mode"]) == (depth, mode)]
                values = [row["ttft_ms"] for row in rows]
                result.append({
                    "kv_type": kv, "depth": depth, "mode": mode, "repetitions": len(rows),
                    "median_ttft_ms": median(values), "min_ttft_ms": min(values), "max_ttft_ms": max(values),
                    "median_server_prompt_ms": median(row["server_timings"]["prompt_ms"] for row in rows),
                    "median_client_minus_server_prompt_ms": median(
                        row["ttft_ms"] - row["server_timings"]["prompt_ms"] for row in rows),
                })
    return result


def benchmark(dataset, result, root, archived, options):
    cache_types = (options.kv,) if options.kv else KV_TYPES
    depths = (options.depth,) if options.depth else DEPTHS
    schedule = [(kv, depths) for kv in cache_types]
    schedule += [(kv, depths[::-1]) for kv in reversed(cache_types)]
    executable = root / "build/llama-cuda/bin/llama-server.exe"
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    for index, (kv, order) in enumerate(schedule, start=1):
        args = server_args(executable, model, "none", cache_type=kv)
        session = {"kv_type": kv, "measurements": []}
        dataset["sessions"].append(session)
        log = result.with_name(f"{result.stem}-{index:02}-{kv}.log")
        with measured_server(args, log, None, session) as base:
            first_token(base, archived["prose_summary"][:512])
            for order_now in (order, order[::-1]):
                for depth in order_now:
                    original, follow_up = prompt_pair(archived, depth)
                    fresh = first_token(base, original)
                    reused = first_token(base, follow_up, cached_tokens=depth - NEW_TOKENS)
                    session["measurements"].extend((
                        {"depth": depth, "mode": "fresh", **fresh},
                        {"depth": depth, "mode": "reused_prefix", **reused},
                    ))
                    print(f"{kv:4} {depth:4} fresh {fresh['ttft_ms']:.1f} ms, "
                          f"reused {reused['ttft_ms']:.1f} ms", flush=True)
    dataset["summary"] = timing_summary(dataset["sessions"], depths, cache_types)


def profile(dataset, result, root, archived, options):
    executable = root / "build/llama-cuda/bin/llama-server.exe"
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    trace_root = root / "profiles"
    trace_root.mkdir(exist_ok=True)
    dataset["profiler"] = subprocess.check_output(["nsys", "--version"], text=True).strip()
    tokens, _ = prompt_pair(archived, 8192)
    for kv in (options.kv,) if options.kv else KV_TYPES:
        prefix = trace_root / f"{result.stem}-{kv}"
        log = result.with_name(f"{prefix.name}.log")
        server_log = result.with_name(f"{prefix.name}-server.log")
        args = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                "--cuda-graph-trace=node", "--duration=25", "--kill=true", "--output", str(prefix),
                *server_args(executable, model, "none", cache_type=kv), "--log-file", str(server_log)]
        session = {"kv_type": kv, "trace": f"{prefix.name}.nsys-rep", "depth": len(tokens)}
        dataset["sessions"].append(session)
        print(f"Tracing fresh {len(tokens)}-token prefill with {kv} cache", flush=True)
        with local_server(args, log, exit_timeout=45) as (base, command):
            session["command"] = command
            first_token(base, tokens[:512])
            session["request"] = first_token(base, tokens)
        if "offloaded 37/37 layers to GPU" not in server_log.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"Full GPU offload not confirmed: inspect {server_log}")
        database = prefix.with_suffix(".sqlite")
        with log.open("a", encoding="utf-8") as output:
            subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(database),
                            str(prefix.with_suffix(".nsys-rep"))], stdout=output, stderr=output, check=True)
        session["cuda"] = analyze(database, session["request"])
        print(f"  captured {session['cuda']['kernel_calls']} kernels", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv", choices=KV_TYPES)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--depth", type=int, choices=DEPTHS)
    group.add_argument("--profile", action="store_true", help="Trace a fresh 8K request in Nsight Systems.")
    options = parser.parse_args()
    root = Path(os.environ["LLM_OPT_ROOT"])
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    executable = root / "build/llama-cuda/bin/llama-server.exe"
    archived = json.loads(FIXTURE.read_text(encoding="utf-8"))["prompts"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "prefill-profile-8k" if options.profile else "ttft"
    result = PROJECT / "results" / f"{label}-{stamp}.json"
    source_revision = subprocess.check_output(["git", "-C", str(root / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip()
    dataset = {
        "status": "running", "fixture": FIXTURE.name, "context_capacity": CONTEXT_CAPACITY,
        "prompt": "Archived project memo, shortened by keeping the first 128 tokens and the tail.",
        "reused_prefix": "Same slot immediately after the fresh request, replacing its final 128 tokens with the freeform suffix.",
        "ttft_definition": "Elapsed wall time from opening the loopback HTTP request to the first streamed token event.",
        "model_manifest": json.loads(model.with_name(model.stem + ".manifest.json").read_text(encoding="utf-8")),
        "source_revision": source_revision,
        "server_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cuda_dll_sha256": hashlib.sha256(executable.with_name("ggml-cuda.dll").read_bytes()).hexdigest(),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True).strip(),
        "sessions": [],
    }
    try:
        (profile if options.profile else benchmark)(dataset, result, root, archived, options)
        dataset["status"] = "passed"
    except Exception as error:
        dataset.update(status="failed", error=str(error))
        raise
    finally:
        result.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
