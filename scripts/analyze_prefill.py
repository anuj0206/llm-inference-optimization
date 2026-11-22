"""Break down CUDA activity in the streamed 8K prefill request window."""

from collections import defaultdict
from contextlib import closing
import sqlite3


def kernel_family(name):
    if "flash_attn" in name:
        return "flash_attention"
    if name.startswith(("mul_mat_q", "mul_mat_vec_q", "quantize_mmq")):
        return "quantized_matmul_and_activation_prep"
    if "gemm" in name.lower():
        return "gemm"
    if name.startswith(("q4_k_fp8_bound", "q4_k_pack_fp8")):
        return "fp8_weight_preparation"
    if name.startswith(("fp8_amax", "quantize_fp8_e4m3")):
        return "fp8_activation_preparation"
    if name == "prefill_fp8_swiglu":
        return "fp8_swiglu"
    if name.startswith(("dequantize_block_q4_K", "dequantize_block_q6_K")):
        return "q4_k_and_q6_k_weight_dequantization"
    if name.startswith(("dequantize_", "convert_unary", "cpy_")):
        return "type_conversion_and_tensor_copy"
    if "set_rows" in name:
        return "set_rows"
    return "other"


def interval_union(intervals):
    covered = 0
    end = 0
    for left, right in sorted(intervals):
        covered += max(0, right - max(end, left))
        end = max(end, right)
    return covered


def analyze(path, request, *, microbatch):
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        epoch, = connection.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
        start = request["request_started_utc_ns"] - epoch
        end = request["first_token_utc_ns"] - epoch
        if start <= 0 or not request["server_timings"]["prompt_n"] == 8192:
            raise RuntimeError("Trace timestamps do not match an 8K request.")
        bounds = (start, end)
        kernels = connection.execute("""
            SELECT k.start, k.end, s.value, d.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.shortName JOIN StringIds d ON d.id=k.demangledName
            WHERE k.start >= ? AND k.end <= ? ORDER BY k.start
        """, bounds).fetchall()
        copies = connection.execute("""
            SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE start >= ? AND end <= ?
        """, bounds).fetchall()

    flash_calls = sum("flash_attn" in short_name and "fixup" not in short_name
                      for _, _, short_name, _ in kernels)
    attention_steps = 8192 // microbatch * 36
    if flash_calls not in (attention_steps, 2 * attention_steps):
        raise RuntimeError(f"Unexpected attention call count ({flash_calls}); inspect the captured window.")

    families = defaultdict(int)
    by_name = defaultdict(lambda: {"calls": 0, "time_ns": 0})
    for left, right, short_name, name in kernels:
        duration = right - left
        families[kernel_family(short_name)] += duration
        by_name[name]["calls"] += 1
        by_name[name]["time_ns"] += duration

    kernel_intervals = [(left, right) for left, right, *_ in kernels]
    all_intervals = kernel_intervals + [(left, right) for left, right, *_ in copies]
    busy = interval_union(all_intervals)
    summed = sum(families.values())
    return {
        "request_window_ms": (end - start) / 1e6,
        "time_to_first_kernel_ms": (kernels[0][0] - start) / 1e6,
        "last_kernel_to_first_token_ms": (end - max(right for _, right, *_ in kernels)) / 1e6,
        "summed_cuda_kernel_ms": summed / 1e6,
        "kernel_interval_union_ms": interval_union(kernel_intervals) / 1e6,
        "kernel_or_copy_union_ms": busy / 1e6,
        "without_observed_kernels_or_copies_ms": (end - start - busy) / 1e6,
        "summed_memcpy_ms": sum(right - left for left, right, *_ in copies) / 1e6,
        "memcpy_events": len(copies),
        "flash_attention_calls_excluding_fixup": flash_calls,
        "kernel_calls": len(kernels),
        "families": {family: {"ms": ns / 1e6, "pct_summed_kernel_time": 100 * ns / summed}
                     for family, ns in sorted(families.items(), key=lambda x: x[1], reverse=True)},
        "top_kernels": [{"name": name, "calls": stats["calls"], "ms": stats["time_ns"] / 1e6}
                        for name, stats in sorted(by_name.items(), key=lambda x: x[1]["time_ns"], reverse=True)[:16]],
    }
