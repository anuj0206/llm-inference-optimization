"""Measure time to the first streamed token on an already loaded Qwen3 8B."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
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
BASELINE_MICROBATCH = 512
MICROBATCH_SIZES = (128, 256, 512)
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
    dataset["microbatch"] = options.microbatch
    for index, (kv, order) in enumerate(schedule, start=1):
        args = server_args(executable, model, "none", cache_type=kv, microbatch=options.microbatch)
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


def sweep_microbatch(dataset, result, root, archived, options):
    executable = root / "build/llama-cuda/bin/llama-server.exe"
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    tokens = archived["prose_summary"]
    kv = options.kv or "q8_0"
    if options.compare_microbatches:
        batch, first, second = options.compare_microbatches
        configurations = ((batch, first), (batch, second))
    elif options.sweep_large_microbatch:
        # A common multiple of all three widths avoids short GPU chunks at logical-batch boundaries.
        configurations = ((512, 512), (3072, 512), (3072, 768), (3072, 1024))
    else:
        configurations = tuple((512, microbatch) for microbatch in MICROBATCH_SIZES)

    if options.compare_microbatches or options.sweep_large_microbatch:
        orders = (configurations, configurations[::-1])
        dataset["batch_configurations"] = [{"batch": batch, "microbatch": microbatch}
                                            for batch, microbatch in configurations]
    else:
        orders = tuple(configurations[i:] + configurations[:i] for i in range(len(configurations)))
        dataset.update(batch=512, microbatches=MICROBATCH_SIZES)
    repeats = 3 if options.compare_microbatches else 2
    dataset.update(prompt_tokens=len(tokens), warmup_prompt_tokens=2048, measured_requests_per_session=repeats)
    dataset["prompt"] = "Archived 8,192-token project memo; each request processes the entire prompt."
    dataset.pop("reused_prefix")

    for block, order in enumerate(orders, start=1):
        for batch, microbatch in order:
            args = server_args(executable, model, "none", cache_type=kv, batch=batch, microbatch=microbatch)
            session = {"kv_type": kv, "block": block, "batch": batch, "microbatch": microbatch, "measurements": []}
            dataset["sessions"].append(session)
            log = result.with_name(f"{result.stem}-{block}-{batch}-{microbatch}.log")
            with measured_server(args, log, None, session) as base:
                first_token(base, tokens[:dataset["warmup_prompt_tokens"]])
                for _ in range(dataset["measured_requests_per_session"]):
                    session["measurements"].append(first_token(base, tokens))

            log_text = log.read_text(encoding="utf-8", errors="replace")
            if (not re.search(rf"\bn_batch\s*=\s*{batch}\b", log_text)
                    or not re.search(rf"\bn_ubatch\s*=\s*{microbatch}\b", log_text)):
                raise RuntimeError(f"Effective batch/microbatch {batch}/{microbatch} not confirmed: inspect {log}")
            compute_buffer = re.search(r"\bCUDA0 compute buffer size\s*=\s*([\d.]+) MiB", log_text)
            if compute_buffer:
                session["cuda_compute_buffer_mib"] = float(compute_buffer[1])
            timings = [row["ttft_ms"] for row in session["measurements"]]
            print(f"Block {block}, {kv} batch/microbatch {batch}/{microbatch}: "
                  f"median {median(timings):.1f} ms, peak {session['sampled_peak_vram_mib']} MiB", flush=True)

    summary = []
    for batch, microbatch in configurations:
        sessions = [session for session in dataset["sessions"]
                    if (session["batch"], session["microbatch"]) == (batch, microbatch)]
        rows = [row for session in sessions for row in session["measurements"]]
        ttft = [row["ttft_ms"] for row in rows]
        summary.append({
            "batch": batch, "microbatch": microbatch, "repetitions": len(rows), "median_ttft_ms": median(ttft),
            "min_ttft_ms": min(ttft), "max_ttft_ms": max(ttft),
            "median_server_prompt_ms": median(row["server_timings"]["prompt_ms"] for row in rows),
            "median_server_prompt_tokens_per_second": median(
                row["server_timings"]["prompt_per_second"] for row in rows),
            "sampled_peak_vram_mib": max(session["sampled_peak_vram_mib"] for session in sessions),
            "first_token_ids": sorted({row["first_token"] for row in rows}),
        })
    dataset["summary"] = summary


def experimental_runtime(root, dataset, backend):
    runtime = root / {"cublas": "build/llama-cublas", "fp8": "build/llama-prefill-fp8"}[backend]
    manifest = json.loads((runtime / "build.json").read_text(encoding="utf-8"))
    executable = runtime / "bin/llama-server.exe"
    valid_build = (manifest.get("compile_flag") == "GGML_CUDA_FORCE_CUBLAS" if backend == "cublas"
                   else bool(manifest.get("fp8_source_sha256")))
    if (not valid_build
            or manifest["source_commit"] != dataset["source_revision"]
            or manifest["stock_cuda_sha256"] != dataset["cuda_dll_sha256"]
            or manifest["server_sha256"] != dataset["server_sha256"]
            or hashlib.sha256(executable.read_bytes()).hexdigest() != dataset["server_sha256"]
            or hashlib.sha256(executable.with_name("ggml-cuda.dll").read_bytes()).hexdigest()
            != manifest["experiment_cuda_sha256"]):
        raise RuntimeError(f"The {backend} runtime must match the pinned stock build.")
    dataset[f"{backend}_build"] = manifest
    return executable


def compare_prefill_backends(dataset, result, root, archived, options):
    experimental = "fp8" if options.compare_fp8 else "cublas"
    batch, microbatch = (4096, 1024) if options.compare_fp8 else (512, options.microbatch)
    executables = {
        "stock": root / "build/llama-cuda/bin/llama-server.exe",
        experimental: experimental_runtime(root, dataset, experimental),
    }
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    tokens = archived["prose_summary"]
    kv = options.kv or "q8_0"
    dataset.update(prompt_tokens=len(tokens), batch=batch, microbatch=microbatch,
                   warmup_prompt_tokens=2048, measured_requests_per_session=3)
    dataset["prompt"] = "Archived 8,192-token project memo; each request processes the entire prompt."
    dataset.pop("reused_prefix")
    marker = {"cublas": "FORCE_CUBLAS = 1", "fp8": "native E4M3 FP8 prefill gate/up active"}[experimental]
    for index, backend in enumerate(("stock", experimental, experimental, "stock"), start=1):
        args = server_args(executables[backend], model, "none", cache_type=kv, batch=batch, microbatch=microbatch)
        session = {"backend": backend, "kv_type": kv, "measurements": []}
        dataset["sessions"].append(session)
        log = result.with_name(f"{result.stem}-{index:02}-{backend}.log")
        environment = dict(os.environ)
        environment.pop("LLM_OPT_PREFILL_FP8", None)
        if backend == "fp8":
            environment["LLM_OPT_PREFILL_FP8"] = "1"
        with measured_server(args, log, environment, session) as base:
            first_token(base, tokens[:dataset["warmup_prompt_tokens"]])
            for _ in range(dataset["measured_requests_per_session"]):
                session["measurements"].append(first_token(base, tokens))

        log_text = log.read_text(encoding="utf-8", errors="replace")
        if (not re.search(rf"\bn_batch\s*=\s*{batch}\b", log_text)
                or not re.search(rf"\bn_ubatch\s*=\s*{microbatch}\b", log_text)):
            raise RuntimeError(f"Effective batch sizes not confirmed: inspect {log}")
        if (marker in log_text) != (backend == experimental):
            raise RuntimeError(f"Unexpected {experimental} dispatch: inspect {log}")
        session["experimental_dispatch_confirmed"] = marker in log_text
        timings = [row["ttft_ms"] for row in session["measurements"]]
        print(f"{backend}: TTFT median {median(timings):.1f} ms, "
              f"peak {session['sampled_peak_vram_mib']} MiB", flush=True)

    dataset["summary"] = {}
    for backend in executables:
        sessions = [s for s in dataset["sessions"] if s["backend"] == backend]
        rows = [row for session in sessions for row in session["measurements"]]
        timings = [row["ttft_ms"] for row in rows]
        dataset["summary"][backend] = {
            "repetitions": len(rows), "median_ttft_ms": median(timings),
            "min_ttft_ms": min(timings), "max_ttft_ms": max(timings),
            "median_server_prompt_tokens_per_second": median(
                row["server_timings"]["prompt_per_second"] for row in rows),
            "sampled_peak_vram_mib": max(s["sampled_peak_vram_mib"] for s in sessions),
            "first_token_ids": sorted({row["first_token"] for row in rows}),
        }


def profile(dataset, result, root, archived, options):
    batch, microbatch = (4096, 1024) if options.profile_fp8 else (512, options.microbatch)
    backends = ("stock", "fp8") if options.profile_fp8 else ("cublas" if options.cublas else "stock",)
    cache_types = (options.kv,) if options.kv else (("q8_0",) if options.profile_fp8 else KV_TYPES)
    warmup = 2048 if options.profile_fp8 else 512
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    trace_root = root / "profiles"
    trace_root.mkdir(exist_ok=True)
    dataset["profiler"] = subprocess.check_output(["nsys", "--version"], text=True).strip()
    dataset.update(batch=batch, microbatch=microbatch, warmup_prompt_tokens=warmup)
    if len(backends) == 1:
        dataset["backend"] = backends[0]
    tokens, _ = prompt_pair(archived, 8192)
    executables = {backend: (root / "build/llama-cuda/bin/llama-server.exe" if backend == "stock"
                             else experimental_runtime(root, dataset, backend)) for backend in backends}
    for backend, kv in ((backend, kv) for backend in backends for kv in cache_types):
        suffix = f"{backend}-{kv}" if options.profile_fp8 else kv
        prefix = trace_root / f"{result.stem}-{suffix}"
        log = result.with_name(f"{prefix.name}.log")
        server_log = result.with_name(f"{prefix.name}-server.log")
        args = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                "--cuda-graph-trace=node", "--duration=25", "--kill=true", "--output", str(prefix),
                *server_args(executables[backend], model, "none", cache_type=kv, batch=batch, microbatch=microbatch),
                "--log-file", str(server_log)]
        session = {"backend": backend, "kv_type": kv, "trace": f"{prefix.name}.nsys-rep", "depth": len(tokens)}
        dataset["sessions"].append(session)
        print(f"Tracing fresh {len(tokens)}-token {backend} prefill with {kv} cache", flush=True)
        environment = dict(os.environ)
        environment.pop("LLM_OPT_PREFILL_FP8", None)
        if backend == "fp8":
            environment["LLM_OPT_PREFILL_FP8"] = "1"
        with local_server(args, log, environment=environment, exit_timeout=45) as (base, command):
            session["command"] = command
            first_token(base, tokens[:warmup])
            session["request"] = first_token(base, tokens)
        log_text = server_log.read_text(encoding="utf-8", errors="replace")
        if "offloaded 37/37 layers to GPU" not in log_text:
            raise RuntimeError(f"Full GPU offload not confirmed: inspect {server_log}")
        if ("FORCE_CUBLAS = 1" in log_text) != (backend == "cublas"):
            raise RuntimeError(f"Unexpected cuBLAS build flag: inspect {server_log}")
        if ("native E4M3 FP8 prefill gate/up active" in log_text) != (backend == "fp8"):
            raise RuntimeError(f"Unexpected FP8 dispatch: inspect {server_log}")
        database = prefix.with_suffix(".sqlite")
        with log.open("a", encoding="utf-8") as output:
            subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(database),
                            str(prefix.with_suffix(".nsys-rep"))], stdout=output, stderr=output, check=True)
        session["cuda"] = analyze(database, session["request"], microbatch=microbatch)
        print(f"  captured {session['cuda']['kernel_calls']} kernels", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv", choices=KV_TYPES)
    parser.add_argument("--microbatch", type=int, choices=MICROBATCH_SIZES, default=BASELINE_MICROBATCH)
    parser.add_argument("--cublas", action="store_true", help="Use the cuBLAS build with --profile.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--depth", type=int, choices=DEPTHS)
    group.add_argument("--profile", action="store_true", help="Trace a fresh 8K request in Nsight Systems.")
    group.add_argument("--sweep-microbatch", action="store_true", help="Sweep 128, 256, and 512 at fresh 8K.")
    group.add_argument("--sweep-large-microbatch", action="store_true",
                       help="Sweep 512, 768, and 1024 at fresh 8K with a matched control.")
    group.add_argument("--compare-microbatches", nargs=3, type=int, metavar=("BATCH", "FIRST", "SECOND"),
                       help="Compare two microbatch sizes at the same logical batch on fresh 8K.")
    group.add_argument("--compare-cublas", action="store_true", help="Compare stock and forced cuBLAS at fresh 8K.")
    group.add_argument("--compare-fp8", action="store_true", help="Compare FP8 gate/up and stock integer prefill at fresh 8K.")
    group.add_argument("--profile-fp8", action="store_true", help="Trace stock and FP8 prefill at fresh 8K.")
    options = parser.parse_args()
    if options.cublas and not options.profile:
        parser.error("--cublas requires --profile. Use --compare-cublas for unprofiled timing.")
    if options.compare_microbatches:
        batch, first, second = options.compare_microbatches
        if not (0 < min(first, second) < max(first, second) <= batch <= CONTEXT_CAPACITY):
            parser.error("Use two distinct positive microbatch sizes up to the logical batch and context capacity.")
    root = Path(os.environ["LLM_OPT_ROOT"])
    model = root / "models/Qwen3-8B-Q4_K_M.gguf"
    executable = root / "build/llama-cuda/bin/llama-server.exe"
    archived = json.loads(FIXTURE.read_text(encoding="utf-8"))["prompts"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label, run = "ttft", benchmark
    if options.profile:
        label = "prefill-profile-8k-cublas" if options.cublas else "prefill-profile-8k"
        run = profile
    elif options.profile_fp8:
        label, run = "prefill-profile-8k-fp8", profile
    elif options.sweep_microbatch:
        label, run = "ttft-microbatch-8k", sweep_microbatch
    elif options.sweep_large_microbatch:
        label, run = "ttft-large-microbatch-8k", sweep_microbatch
    elif options.compare_microbatches:
        label, run = "ttft-microbatch-compare-8k", sweep_microbatch
    elif options.compare_cublas:
        label, run = "ttft-cublas-8k", compare_prefill_backends
    elif options.compare_fp8:
        label, run = "ttft-fp8-8k", compare_prefill_backends
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
        run(dataset, result, root, archived, options)
        dataset["status"] = "passed"
    except Exception as error:
        dataset.update(status="failed", error=str(error))
        raise
    finally:
        result.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
