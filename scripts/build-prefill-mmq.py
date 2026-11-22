"""Build a prefill MMQ experiment against the pinned stock CUDA objects."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from prepare_combined_runtime import sha256


PIN = "8bf88ba9368e70fe3ba1b0001b34e70ffc713716"
BENCHMARK = Path(__file__).resolve().parent.parent / "benchmarks/prefill-mmq"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8", action="store_true", help="Build the FP8 gate/up runtime.")
    options = parser.parse_args()
    root = Path(os.environ["LLM_OPT_ROOT"])
    source = root / "llama.cpp"
    stock = root / "build/llama-cuda"
    target = root / ("build/llama-prefill-fp8" if options.fp8 else "build/llama-prefill-mmq")
    binary = target / "bin"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise RuntimeError("Use the clean, pinned llama.cpp checkout.")
    binary.mkdir(parents=True, exist_ok=True)
    manifest = target / "build.json"
    manifest.unlink(missing_ok=True)

    cuda = source / "ggml/src/ggml-cuda"
    original = (cuda / "mmq.cu").read_text(encoding="utf-8")
    marker = "    GGML_ASSERT(J_best > 0);"
    if original.count(marker) != 1:
        raise RuntimeError("The pinned MMQ tile-selection code changed.")
    candidate = target / "mmq.cu"
    if options.fp8:
        source_text = original.replace(marker, '''    static const bool use_prefill_fp8 = [] {
        const char * value = std::getenv("LLM_OPT_PREFILL_FP8");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (use_prefill_fp8 && !ids && gate && src0->type == GGML_TYPE_Q4_K &&
            fusion->glu_op == GGML_GLU_OP_SWIGLU && !fusion->x_scale && !fusion->gate_scale &&
            cc == GGML_CUDA_CC_ADA_LOVELACE && ne00 == 4096 && ne01 == 12288 && ne11 == 1024 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(gate) && ggml_is_contiguous(dst)) {
        prefill_fp8_gate_up(ctx, src0, gate, src1, dst);
        static const bool announced = [] {
            GGML_LOG_INFO("LLMOptimization: native E4M3 FP8 prefill gate/up active\\n");
            return true;
        }();
        (void) announced;
        return;
    }
''' + marker).replace('#include "mmid.cuh"', '#include "mmid.cuh"\n#include "fp8.inc"', 1)
    else:
        source_text = original.replace(marker, '''    if (!ids && gate && src0->type == GGML_TYPE_Q4_K && prec_src1 == GGML_PREC_Q8 &&
            cc == GGML_CUDA_CC_ADA_LOVELACE && ne00 == input_features &&
            ne01 == output_features && ne11 == token_columns && ne12 == 1 && ne13 == 1) {
        if (const int width = prefill_q4_tile_override()) {
            J_best = width;
            nthreads_best = ggml_cuda_mmq_get_config(src0->type, width, fallback, cc, prec_src1).nthreads;
            static const bool announced = [width] {
                GGML_LOG_INFO("LLMOptimization: fused Q4_K prefill tile width = %d\\n", width);
                return true;
            }();
            (void) announced;
        }
    }
''' + marker).replace('#include "mmq.cuh"', '#include "mmq.cuh"\n#include "dispatch.inc"', 1)
        source_text += '\n#include "launch.inc"\n'
    candidate.write_text('#include <cstdlib>\n#include <cstring>\n' + source_text, encoding="utf-8")

    rules = json.loads(subprocess.check_output([
        "ninja", "-t", "compdb", "-x", "CUDA_COMPILER__ggml-cuda_unscanned_Release",
        "CUDA_SHARED_LIBRARY_LINKER__ggml-cuda_Release",
    ], cwd=stock))
    compile_rule = next(row for row in rules if row["output"].endswith("/mmq.cu.obj"))
    flags, separator, _ = compile_rule["command"].partition(" -MD -MT ")
    if not separator or "GGML_CUDA_FORCE_CUBLAS" in flags:
        raise RuntimeError("Expected the stock MMQ compile flags.")
    obj = target / "mmq.cu.obj"
    command = f'{flags} -I"{cuda}" -I"{BENCHMARK}" -x cu -c "{candidate}" -o "{obj}"'
    subprocess.run(command, cwd=stock, check=True)

    link_rule = next(row for row in rules if row["output"] == "bin/ggml-cuda.dll")
    match = re.search(r' -- "([^"]+\\link\.exe)" (.+) && cd \."', link_rule["command"])
    if not match:
        raise RuntimeError("Could not locate the stock CUDA linker invocation.")
    linker, arguments = match.groups()
    for before, after in (
        (str(Path(compile_rule["output"])), obj),
        ("bin\\ggml-cuda.dll", binary / "ggml-cuda.dll"),
        ("ggml\\src\\ggml-cuda\\ggml-cuda.lib", target / "ggml-cuda.lib"),
        ("bin\\ggml-cuda.pdb", target / "ggml-cuda.pdb"),
    ):
        if arguments.count(before) != 1:
            raise RuntimeError(f"Unexpected stock link command around {before}.")
        arguments = arguments.replace(before, f'"{after}"')
    response = target / "link.rsp"
    response.write_text(arguments + "\n", encoding="utf-16")
    subprocess.run([linker, f"@{response}"], cwd=stock, check=True)

    stock_bin = stock / "bin"
    for path in [*stock_bin.glob("*.dll"), stock_bin / "llama-server.exe"]:
        if path.name != "ggml-cuda.dll":
            shutil.copy2(path, binary / path.name)
    build = {
        "source_commit": revision,
        "dispatcher_sha256": sha256(candidate),
        "stock_cuda_sha256": sha256(stock_bin / "ggml-cuda.dll"),
        "experiment_cuda_sha256": sha256(binary / "ggml-cuda.dll"),
        "server_sha256": sha256(binary / "llama-server.exe"),
    }
    if options.fp8:
        build["fp8_source_sha256"] = sha256(BENCHMARK / "fp8.inc")
    else:
        harness = target / "harness"
        subprocess.run(["cmake", "-S", str(BENCHMARK), "-B", str(harness), "-G", "Ninja",
                        "-DCMAKE_BUILD_TYPE=Release", f"-DSTOCK_SOURCE={source}",
                        f"-DSTOCK_RUNTIME_BUILD={stock}", f"-DEXPERIMENT_BUILD={target}"], check=True)
        subprocess.run(["cmake", "--build", str(harness)], check=True)
        build["benchmark_sha256"] = sha256(binary / "prefill-mmq-bench.exe")
    manifest.write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    print(f"Prefill MMQ runtime ready at {binary}")


if __name__ == "__main__":
    main()
