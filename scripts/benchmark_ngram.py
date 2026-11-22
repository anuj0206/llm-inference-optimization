"""Shared settings and requests for the n-gram benchmarks."""

from contextlib import contextmanager
import csv
import re
import subprocess

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


def prompts(base, project, *, held_out=False, fresh=False):
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
    def chat(instruction, context="Read these project notes."):
        template = request(base, "/apply-template", {
            "messages": [{"role": "user", "content": f"{context}\n\n{marker}\n\n{instruction}"}],
            "chat_template_kwargs": {"enable_thinking": False},
        })["prompt"]
        before, separator, after = template.partition(marker)
        if not separator:
            raise RuntimeError("Chat template did not preserve the context placeholder")
        return fit(before, chat_context, after)

    if fresh:
        return {
            "code_continuation": fit("# Python API wrappers\n", code, "def get_item_500", take_tail=True),
            "prose_summary": chat(
                "Write a short status memo for an engineer joining the project. Summarize the evidence "
                "for the memory bottleneck and give one limitation of the measurements."
            ),
            "prose_rewrite": chat(
                "Turn the notes into a plain-English explanation for a project manager. Explain the role "
                "of the 8K context and distinguish the effect on memory from the effect on token speed."
            ),
            "prose_freeform": chat(
                "For a new task, explain to a friend why a busy coffee shop can have idle baristas "
                "while the drink line still gets longer. Use three short paragraphs and one concrete example.",
                "These are notes from an earlier task. Set them aside for the next request.",
            ),
            "prose_fiction": chat(
                "For a new task, write the opening of a short story in which a ferry captain discovers "
                "an old postcard addressed to tomorrow. Begin with an action and keep the tone understated.",
                "These are notes from an earlier task. Set them aside for the next request.",
            ),
        }
    if held_out:
        return {"prose_briefing": chat(
            "Write a letter to a student who has never programmed a GPU. In four paragraphs, explain "
            "why changing how a GPU reads data can fail to make the model faster. Use an everyday analogy "
            "and explain what measurement would change your mind."
        )}
    return {
        "code_continuation": fit("# Python API wrappers\n", code, "def get_item_500", take_tail=True),
        "project_chat": chat(
            "Write a practical design review for a colleague. In four paragraphs explain how the tests "
            "were set up, what the results mean, and what the next experiment should measure."
        ),
    }


def draft_counters(base):
    metrics = request(base, "/metrics", decode_json=False)
    samples = dict(line.split()[:2] for line in metrics.splitlines() if line and not line.startswith("#"))
    return {name: int(float(samples[key])) for name, key in DRAFT_METRICS.items()}


def completion(base, tokens, *, prompt_length=PROMPT_TOKENS):
    before = draft_counters(base)
    response = request(base, "/completion", {
        "prompt": tokens, "n_predict": OUTPUT_TOKENS, "temperature": 0, "seed": 42,
        "cache_prompt": False, "ignore_eos": True, "stream": False, "return_tokens": True,
    })
    timing = response["timings"]
    if (timing["prompt_n"] != prompt_length or timing["predicted_n"] != OUTPUT_TOKENS
            or len(response["tokens"]) != OUTPUT_TOKENS or response.get("truncated")):
        raise RuntimeError(f"Unexpected workload lengths: {timing}")
    after = draft_counters(base)
    return {
        "timings": timing, "tokens": response["tokens"], "text": response["content"],
        **{name: after[name] - before[name] for name in before},
    }


def server_args(executable, model, mode, *, context_capacity=CONTEXT_CAPACITY, cache_type="f16", batch=512, microbatch=128):
    return [str(executable), "-m", str(model), "-ngl", "99", "-c", str(context_capacity),
            "-np", "1", "-b", str(batch), "-ub", str(microbatch), "-t", "8", "-fa", "on",
            "-ctk", cache_type, "-ctv", cache_type, "--fit", "off", "--metrics", "-lv", "4", "--spec-type", mode]


@contextmanager
def measured_server(args, log_path, environment, session):
    gpu_log = log_path.with_suffix(".gpu.log")
    with gpu_log.open("w", encoding="utf-8") as gpu_stream:
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=memory.used,temperature.gpu",
            "--format=csv,noheader,nounits", "--loop-ms=200",
        ], stdout=gpu_stream, stderr=subprocess.STDOUT)
        try:
            with local_server(args, log_path, environment=environment) as (base, command):
                session["command"] = command
                yield base
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
    log = log_path.read_text(encoding="utf-8", errors="replace")
    session["offloaded_layers"] = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    if not session["offloaded_layers"] or any(a != b for a, b in session["offloaded_layers"]):
        raise RuntimeError(f"Full GPU offload not confirmed: inspect {log_path}")


def token_comparisons(fixture, sessions):
    comparisons = {}
    for case in fixture:
        rows = [(session["mode"], row["tokens"]) for session in sessions
                for row in session["measurements"] if row["case"] == case]
        reference = next(tokens for mode, tokens in rows if mode == "none")
        comparisons[case] = {
            "all_runs_match": all(tokens == reference for _, tokens in rows),
            "first_mismatch_indices": [
                next((i for i, (a, b) in enumerate(zip(reference, tokens, strict=True)) if a != b), None)
                for _, tokens in rows
            ],
        }
    return comparisons
