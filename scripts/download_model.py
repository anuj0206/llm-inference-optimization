"""Download and hash-check the official Qwen3 4B development GGUF."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

REPO = "Qwen/Qwen3-4B-GGUF"
REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
FILENAME = "Qwen3-4B-Q4_K_M.gguf"
ROOT = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization" / "models"


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    entry = next(item for item in metadata["siblings"] if item["rfilename"] == FILENAME)
    expected = entry["lfs"]["sha256"]
    target = ROOT / FILENAME
    if not target.exists():
        partial = target.with_suffix(".gguf.partial")
        print(f"Downloading {REPO}@{revision}: {entry['size'] / 1024**3:.2f} GiB", flush=True)
        subprocess.run([
            "curl.exe", "--fail", "--location", "--retry", "4", "--continue-at", "-",
            "--output", str(partial), f"https://huggingface.co/{REPO}/resolve/{revision}/{FILENAME}",
        ], check=True)
        candidate = partial
    else:
        candidate = target
    with candidate.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise RuntimeError(f"Model hash mismatch: {digest} != {expected}")
    if candidate != target:
        candidate.replace(target)
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{revision}/LICENSE", timeout=60) as response, (ROOT / "Qwen3-LICENSE").open("wb") as stream:
        shutil.copyfileobj(response, stream)
    manifest = {"repository": REPO, "revision": revision, "filename": FILENAME,
                "sha256": digest, "size_bytes": target.stat().st_size, "license": "Apache-2.0"}
    (ROOT / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified model at {target}", flush=True)
