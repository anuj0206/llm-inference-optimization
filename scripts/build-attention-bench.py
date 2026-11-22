"""Compile a pinned CUDA dispatcher variant without rebuilding llama.cpp."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PIN = "8bf88ba9368e70fe3ba1b0001b34e70ffc713716"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Unexpected pinned layout around {before!r}")
    return text.replace(before, after)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(os.environ["LLM_OPT_ROOT"])
    source, stock, target = (root / path for path in ("llama.cpp", "build/llama-cuda", "build/llama-attention"))
    bin_dir = target / "bin"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if revision != PIN or dirty:
        raise RuntimeError("Use the clean pinned llama.cpp checkout.")

    manifest = target / "build.json"
    bin_dir.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)
    cuda = source / "ggml/src/ggml-cuda"
    text = (cuda / "fattn.cu").read_text(encoding="utf-8")
    marker = """!(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;"""
    text = replace_once(text, marker, """!(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    static const bool try_grouped = std::getenv("LLM_OPT_ATTENTION_MMA") != nullptr;
                    if (try_grouped && gqa_opt_applies && gqa_ratio == 4 && Q->ne[0] == 128 &&
                            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 && K->ne[1] >= 4096) {
                        static const bool announced = [] {
                            GGML_LOG_INFO("LLMOptimization: enabled grouped FP16 Tensor Core decode attention\\n");
                            return true;
                        }();
                        (void) announced;
                        return BEST_FATTN_KERNEL_MMA_F16;
                    }
                    return BEST_FATTN_KERNEL_VEC;""")
    text = "#include <cstdlib>\n#include \"ggml-backend-impl.h\"\n" + text + """

extern "C" __declspec(dllexport) cudaStream_t attention_bench_stream(ggml_backend_t backend) {
    return static_cast<ggml_backend_cuda_context *>(backend->context)->stream();
}

extern "C" __declspec(dllexport) const char * attention_bench_path(ggml_tensor * output) {
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), output)) {
        case BEST_FATTN_KERNEL_VEC: return "vector";
        case BEST_FATTN_KERNEL_MMA_F16: return "grouped_mma";
        default: return "other";
    }
}

extern "C" __declspec(dllexport) void attention_bench_launch(ggml_backend_t backend, ggml_tensor * output) {
    ggml_cuda_flash_attn_ext(*static_cast<ggml_backend_cuda_context *>(backend->context), output);
}
"""
    candidate = target / "fattn.cu"
    candidate.write_text(text, encoding="utf-8")
    rules = json.loads(subprocess.check_output([
        "ninja", "-t", "compdb", "-x", "CUDA_COMPILER__ggml-cuda_unscanned_Release",
        "CUDA_SHARED_LIBRARY_LINKER__ggml-cuda_Release",
    ], cwd=stock))
    compile_rule = next(row for row in rules if row["output"].endswith("/fattn.cu.obj"))
    flags, separator, _ = compile_rule["command"].partition(" -MD -MT ")
    if not separator:
        raise RuntimeError("CUDA compile command is missing the dependency arguments.")
    obj = target / "fattn.cu.obj"
    command = f'{flags} -I"{cuda}" -x cu -c "{candidate}" -o "{obj}"'
    subprocess.run(command, cwd=stock, check=True)

    link_rule = next(row for row in rules if row["output"] == "bin/ggml-cuda.dll")
    match = re.search(r' -- "([^"]+\\link\.exe)" (.+) && cd \."', link_rule["command"])
    if not match:
        raise RuntimeError("Could not locate the stock CUDA linker invocation.")
    link, arguments = match.groups()
    for original, replacement in (
        (str(Path(compile_rule["output"])), obj),
        ("bin\\ggml-cuda.dll", bin_dir / "ggml-cuda.dll"),
        ("ggml\\src\\ggml-cuda\\ggml-cuda.lib", target / "ggml-cuda.lib"),
        ("bin\\ggml-cuda.pdb", target / "ggml-cuda.pdb"),
    ):
        arguments = replace_once(arguments, original, f'"{replacement}"')
    response = target / "link.rsp"
    response.write_text(arguments + "\n", encoding="utf-16")
    subprocess.run([link, f"@{response}"], cwd=stock, check=True)

    stock_bin = stock / "bin"
    for dll in stock_bin.glob("*.dll"):
        if dll.name != "ggml-cuda.dll":
            shutil.copy2(dll, bin_dir / dll.name)
    shutil.copy2(stock_bin / "llama-server.exe", bin_dir / "llama-server.exe")
    manifest.write_text(json.dumps({
        "source_commit": revision,
        "dispatcher_sha256": sha256(candidate),
        "stock_cuda_sha256": sha256(stock_bin / "ggml-cuda.dll"),
        "experiment_cuda_sha256": sha256(bin_dir / "ggml-cuda.dll"),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Attention experiment DLL ready at {bin_dir}")


if __name__ == "__main__":
    main()
