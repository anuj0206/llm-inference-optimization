"""Run a small, fixed JSON-answer regression screen with FP16 and Q8 KV caches."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from quality_score import score_response


def request(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def fixtures():
    cases = [
        ("multiply", "What is 137 times 29? Return a JSON number.", 3973),
        ("percent", "What is 18 percent of 250? Return a JSON number.", 45),
        ("alias", "Python: x=[1,2]; y=x; y.append(3). What is len(x)? Return a JSON number.", 3),
        ("slice", "Python: what is [0,1,2,3,4,5][::-2]? Return a JSON array.", [5, 3, 1]),
        ("loop", "C++: int count=0; for(int i=0;i<7;i+=2){count++;} What is count? Return a JSON number.", 4),
        ("substring", "Python: what is 'Mississippi'.count('ss')? Return a JSON number.", 2),
        ("sort", "Sort and deduplicate [5,1,3,1,2,5]. Return a JSON array.", [1, 2, 3, 5]),
        ("logic", "All glips are blops. All blops are trens. Must every glip be a tren? Return a JSON boolean.", True),
        ("aggregate", 'For values [7,11,13], return a JSON object with keys "total" (sum) and "count" (number of values).', {"total": 31, "count": 3}),
    ]
    filler = [f"Record {i:04d}: the item is ordinary and its status is unchanged.\n" for i in range(400)]
    for position, index, code in (("start", 0, "V7Q4M2"), ("middle", 200, "P8R3T6"), ("end", 400, "X2B9K5")):
        lines = filler.copy()
        lines.insert(index, f"Special record: the access_code is {code}.\n")
        prompt = "Read this ledger carefully.\n" + "".join(lines) + "\nWhat is the access_code in the special record? Return only its value as a JSON string."
        cases.append((f"retrieval_{position}", prompt, code))
    return [{"id": name, "prompt": prompt, "expected": expected} for name, prompt, expected in cases]


if __name__ == "__main__":
    root = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization"
    results = Path(__file__).resolve().parent.parent / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cases = fixtures()
    fixture_path = results / "quality-fixtures.json"
    fixture_path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    records = []
    server_commands = {}
    for cache in ("f16", "q8_0"):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [str(root / "build/llama-cuda/bin/llama-server.exe"), "-m", str(root / "models/Qwen3-8B-Q4_K_M.gguf"),
                   "-ngl", "99", "-c", "8448", "-np", "1", "-b", "512", "-ub", "128", "-t", "8", "-fa", "on",
                   "-ctk", cache, "-ctv", cache, "--fit", "off", "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
        server_commands[cache] = command
        print(f"Starting quality screen with {cache} cache", flush=True)
        with (results / f"quality-{cache}-{stamp}.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 120
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited: inspect quality-{cache}-{stamp}.log")
                    try:
                        if request(base, "/health").get("status") == "ok":
                            break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError("Server startup timed out")
                    time.sleep(0.5)
                for case in cases:
                    messages = [{"role": "system", "content": "Answer accurately. Return only the requested JSON value, without markdown or explanation."},
                                {"role": "user", "content": case["prompt"]}]
                    response = request(base, "/v1/chat/completions", {"messages": messages, "temperature": 0, "seed": 42,
                                       "max_tokens": 128, "stream": False, "cache_prompt": False,
                                       "chat_template_kwargs": {"enable_thinking": False}})
                    choice = response["choices"][0]
                    content = choice["message"]["content"] or ""
                    scores = score_response(content, case["expected"])
                    record = {"cache_type": cache, "case": case["id"], "expected": case["expected"],
                              "content": content, **scores, "usage": response.get("usage"), "finish_reason": choice["finish_reason"]}
                    records.append(record)
                    print(cache, case["id"], "PASS" if scores["correct"] else "FAIL", response.get("usage"), flush=True)
            finally:
                server.terminate()
                server.wait(timeout=20)
                target = results / f"quality-{stamp}.json"
                target.write_text(json.dumps({"sampling": {"temperature": 0, "seed": 42, "thinking": False},
                    "context_capacity": 8448, "fixture_file": fixture_path.name, "records": records,
                    "server_commands": server_commands,
                    "model_manifest": json.loads((root / "models/Qwen3-8B-Q4_K_M.manifest.json").read_text()),
                    "scoring": "Answer correctness allows single-value object wrappers and bare JSON key/value fragments; requested_value_match and strict_json_valid separately record formatting. Raw outputs retained.",
                    "limitation": "12 easy exact-answer checks, including three synthetic long-prefix retrieval cases. A regression screen, not a representative accuracy benchmark or proof of quality parity."}, indent=2) + "\n", encoding="utf-8")
    for cache in ("f16", "q8_0"):
        subset = [row for row in records if row["cache_type"] == cache]
        print(cache, sum(row["correct"] for row in subset), "/", len(subset), flush=True)
