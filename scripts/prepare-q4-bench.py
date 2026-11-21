"""Prepare pinned stock kernel excerpts and offsets into the verified GGUF file."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

PIN = "8bf88ba9368e70fe3ba1b0001b34e70ffc713716"
root = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization"
source = root / "llama.cpp"
build = root / "build/q4-microbench"
build.mkdir(parents=True, exist_ok=True)
if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != PIN:
    raise SystemExit("The harness requires the pinned stock llama.cpp checkout")
if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
    raise SystemExit("The stock checkout must be clean")
mmvq = (source / "ggml/src/ggml-cuda/mmvq.cu").read_text()
quant = (source / "ggml/src/ggml-cuda/quantize.cu").read_text()
prefix = mmvq.split("// Dedicated MoE multi-token kernel.")[0]
if "static __global__ void mul_mat_vec_q(" not in prefix or len(prefix) >= len(mmvq):
    raise SystemExit("Unexpected pinned kernel structure")
start = quant.index("__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)")
end = quant.index("__device__ __forceinline__ uint8_t compute_e8m0_scale", start)
quant_prefix = quant[start:end]
(build / "stock_mmvq.cuh").write_text(prefix)
kernel_start = prefix.index("template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>")
stock_kernel = prefix[kernel_start:]
(build / "stock_mmvq_kernel.cuh").write_text(stock_kernel)
for label, suffix, warps in (("one_warp", "one_warp", 1), ("two_warp", "two_warps", 2)):
    kernel = stock_kernel
    for old, new in (
        ("calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*", f"{warps}*"),
        ("static __global__ void mul_mat_vec_q(", f"static __global__ void mul_mat_vec_q_{suffix}("),
        ("constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);", f"constexpr int nwarps = {warps};"),
    ):
        if kernel.count(old) != 1:
            raise SystemExit(f"Expected one kernel substitution for {old!r}")
        kernel = kernel.replace(old, new)
    (build / f"{label}_mmvq.cuh").write_text(kernel)
row_expression = "calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps)"
if stock_kernel.count(row_expression) != 1:
    raise SystemExit("Unexpected stock rows-per-block expression")
two_row_kernel = stock_kernel.replace(row_expression, "2").replace(
    "static __global__ void mul_mat_vec_q(", "static __global__ void mul_mat_vec_q_two_rows(")
(build / "two_row_mmvq.cuh").write_text(two_row_kernel)
(build / "stock_quantize.cuh").write_text(quant_prefix)
(build / "llama.cpp-LICENSE").write_bytes((source / "LICENSE").read_bytes())
spec = importlib.util.spec_from_file_location("inventory", Path(__file__).with_name("inspect-q4-k.py"))
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)
model = root / "models/Qwen3-8B-Q4_K_M.gguf"
manifest = json.loads(model.with_name(model.stem + ".manifest.json").read_text())
with model.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["sha256"]:
        raise SystemExit("GGUF checksum failed")
metadata, tensors, header_end = inventory.inspect_model(model)
alignment = metadata.get("general.alignment", 32)
data_start = (header_end + alignment - 1) // alignment * alignment
lookup = {t["name"]: t for t in tensors}
pairs = []
for layer in range(36):
    pair = {"layer": layer}
    for role in ("up", "gate"):
        tensor = lookup[f"blk.{layer}.ffn_{role}.weight"]
        if tensor["ggml_type"] != 12 or tensor["ggml_dimensions"] != [4096, 12288]:
            raise SystemExit(f"Unexpected FFN tensor: {tensor}")
        pair[role + "_offset"] = data_start + tensor["data_offset"]
    pairs.append(pair)
fixture = {"model_path": str(model), "model_manifest": manifest, "source_commit": PIN,
           "source_mmvq_sha256": hashlib.sha256((source / "ggml/src/ggml-cuda/mmvq.cu").read_bytes()).hexdigest(),
           "source_quantize_sha256": hashlib.sha256((source / "ggml/src/ggml-cuda/quantize.cu").read_bytes()).hexdigest(),
           "generated_mmvq_sha256": hashlib.sha256((build / "stock_mmvq.cuh").read_bytes()).hexdigest(),
           "generated_two_warp_sha256": hashlib.sha256((build / "two_warp_mmvq.cuh").read_bytes()).hexdigest(),
           "generated_one_warp_sha256": hashlib.sha256((build / "one_warp_mmvq.cuh").read_bytes()).hexdigest(),
           "generated_two_row_sha256": hashlib.sha256((build / "two_row_mmvq.cuh").read_bytes()).hexdigest(),
           "generated_quantize_sha256": hashlib.sha256((build / "stock_quantize.cuh").read_bytes()).hexdigest(),
           "n": 12288, "k": 4096, "pairs": pairs}
(build / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
print(f"Prepared {len(pairs)} weight-pair offsets and exact stock excerpts in {build}")
