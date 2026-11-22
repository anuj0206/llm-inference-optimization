"""Build a cuBLAS comparison DLL from the pinned stock CUDA build."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from prepare_combined_runtime import sha256


PIN = "8bf88ba9368e70fe3ba1b0001b34e70ffc713716"
FLAG = "GGML_CUDA_FORCE_CUBLAS"
SOURCES = ("ggml-cuda.cu", "mmq.cu", "mmq-few.cu")


def main():
    root = Path(os.environ["LLM_OPT_ROOT"])
    source = root / "llama.cpp"
    stock = root / "build/llama-cuda"
    target = root / "build/llama-cublas"
    bin_dir = target / "bin"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if revision != PIN or dirty:
        raise RuntimeError("Use the clean, pinned llama.cpp checkout.")

    bin_dir.mkdir(parents=True, exist_ok=True)
    manifest = target / "build.json"
    manifest.unlink(missing_ok=True)
    rules = json.loads(subprocess.check_output([
        "ninja", "-t", "compdb", "-x", "CUDA_COMPILER__ggml-cuda_unscanned_Release",
        "CUDA_SHARED_LIBRARY_LINKER__ggml-cuda_Release",
    ], cwd=stock))
    object_paths = {}
    for name in SOURCES:
        rule = next(row for row in rules if row["output"].endswith(f"/{name}.obj"))
        flags, separator, _ = rule["command"].partition(" -MD -MT ")
        if not separator or FLAG in flags:
            raise RuntimeError(f"Expected an unmodified stock CUDA compile command for {name}.")
        obj = target / f"{name}.obj"
        command = f'{flags} -D{FLAG} -x cu -c "{rule["file"]}" -o "{obj}"'
        print(f"Compiling {name} with {FLAG}", flush=True)
        subprocess.run(command, cwd=stock, check=True)
        object_paths[str(Path(rule["output"]))] = obj

    link_rule = next(row for row in rules if row["output"] == "bin/ggml-cuda.dll")
    match = re.search(r' -- "([^"]+\\link\.exe)" (.+) && cd \."', link_rule["command"])
    if not match:
        raise RuntimeError("Could not locate the stock CUDA linker invocation.")
    linker, arguments = match.groups()
    replacements = {
        **object_paths,
        "bin\\ggml-cuda.dll": bin_dir / "ggml-cuda.dll",
        "ggml\\src\\ggml-cuda\\ggml-cuda.lib": target / "ggml-cuda.lib",
        "bin\\ggml-cuda.pdb": target / "ggml-cuda.pdb",
    }
    for before, after in replacements.items():
        if arguments.count(before) != 1:
            raise RuntimeError(f"Unexpected stock link command around {before}.")
        arguments = arguments.replace(before, f'"{after}"')
    response = target / "link.rsp"
    response.write_text(arguments + "\n", encoding="utf-16")
    subprocess.run([linker, f"@{response}"], cwd=stock, check=True)

    stock_bin = stock / "bin"
    for path in [*stock_bin.glob("*.dll"), stock_bin / "llama-server.exe"]:
        if path.name != "ggml-cuda.dll":
            shutil.copy2(path, bin_dir / path.name)
    manifest.write_text(json.dumps({
        "source_commit": revision,
        "compile_flag": FLAG,
        "rebuilt_sources": SOURCES,
        "stock_cuda_sha256": sha256(stock_bin / "ggml-cuda.dll"),
        "experiment_cuda_sha256": sha256(bin_dir / "ggml-cuda.dll"),
        "server_sha256": sha256(bin_dir / "llama-server.exe"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"cuBLAS runtime ready at {bin_dir}")


if __name__ == "__main__":
    main()
