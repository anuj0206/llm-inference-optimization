"""Assemble the pinned host and CUDA optimization DLLs in a separate runtime."""

import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root):
    build_root = root / "build"
    attention = build_root / "llama-attention"
    adaptive = build_root / "llama-budget-ngram"
    if not (attention / "build.json").is_file() or not (adaptive / "build.json").is_file():
        raise RuntimeError("Build attention and the adaptive budget runtime before combining them.")
    cuda = json.loads((attention / "build.json").read_text(encoding="utf-8"))
    host = json.loads((adaptive / "build.json").read_text(encoding="utf-8"))
    if cuda["source_commit"] != host["source_commit"] or not cuda.get("grouped_q8_sha256") or not host.get("budget_policy_sha256"):
        raise RuntimeError("Use compatible grouped Q8_0 and adaptive budget builds.")

    files = {
        "ggml-cuda.dll": (attention / "bin/ggml-cuda.dll", cuda["experiment_cuda_sha256"]),
        "llama-common.dll": (adaptive / "bin/llama-common.dll", host["adaptive_common_sha256"]),
    }
    for path, expected in files.values():
        if sha256(path) != expected:
            raise RuntimeError(f"Build hash changed for {path}")
    stock = build_root / "llama-cuda/bin"
    if (sha256(stock / "ggml-cuda.dll") != cuda["stock_cuda_sha256"]
            or sha256(stock / "llama-common.dll") != host["stock_common_sha256"]):
        raise RuntimeError("The stock build changed after the optimization DLLs were built.")

    target = build_root / "llama-combined"
    binary = target / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    manifest = target / "build.json"
    manifest.unlink(missing_ok=True)
    inputs = {dll.name: dll for dll in stock.glob("*.dll")}
    inputs.update({name: stock / name for name in ("llama-server.exe", "llama-cli.exe")})
    inputs.update({name: path for name, (path, _) in files.items()})
    for name, path in inputs.items():
        shutil.copy2(path, binary / name)
    manifest.write_text(json.dumps({"attention": cuda, "ngram": host}, indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(f"Combined runtime ready at {prepare(Path(os.environ['LLM_OPT_ROOT'])) / 'bin'}")
