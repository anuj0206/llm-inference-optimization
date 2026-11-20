"""Inspect GGUF tensor metadata and existing Q4_K CUDA profile launches, read-only."""

from collections import Counter, defaultdict
import ctypes
import json
import math
import os
from pathlib import Path
import sqlite3
import struct


def read_number(stream, fmt):
    return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))[0]


def read_string(stream):
    return stream.read(read_number(stream, "Q")).decode("utf-8")


def read_value(stream, kind):
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    if kind in formats:
        return read_number(stream, formats[kind])
    if kind == 8:
        return read_string(stream)
    if kind == 9:
        subtype, count = read_number(stream, "I"), read_number(stream, "Q")
        for _ in range(count):
            read_value(stream, subtype)
        return {"array_length": count, "element_type": subtype}
    raise ValueError(f"Unknown GGUF value type {kind}")


def inspect_model(model):
    with model.open("rb") as stream:
        if stream.read(4) != b"GGUF" or read_number(stream, "I") != 3:
            raise ValueError("Expected a GGUF version 3 file")
        count, metadata_count = read_number(stream, "Q"), read_number(stream, "Q")
        metadata = {}
        for _ in range(metadata_count):
            name = read_string(stream)
            metadata[name] = read_value(stream, read_number(stream, "I"))
        tensors = []
        for _ in range(count):
            name = read_string(stream)
            dimensions = [read_number(stream, "Q") for _ in range(read_number(stream, "I"))]
            kind, offset = read_number(stream, "I"), read_number(stream, "Q")
            tensors.append({"name": name, "ggml_dimensions": dimensions, "ggml_type": kind, "data_offset": offset})
        return metadata, tensors, stream.tell()


if __name__ == "__main__":
    root = Path(os.environ["LOCALAPPDATA"]) / "LLMOptimization"
    metadata, tensors, header_bytes = inspect_model(root / "models/Qwen3-8B-Q4_K_M.gguf")
    grouped = defaultdict(list)
    for tensor in tensors:
        grouped[(tensor["ggml_type"], tuple(tensor["ggml_dimensions"]))].append(tensor["name"])
    inventory = [{"ggml_type": kind, "ggml_dimensions": list(shape), "count": len(names), "examples": names[:4]}
                 for (kind, shape), names in sorted(grouped.items())]
    profile = json.loads((Path(__file__).resolve().parent.parent / "results/profile-analysis.json").read_text())
    launches = {}
    for depth, summary in profile.items():
        path = Path(summary["source"])
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        ids = [item[0] for item in summary["selected_graphs"]]
        query = f"""SELECT s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                    k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,
                    count(*),sum(k.end-k.start),min(k.end-k.start),max(k.end-k.start)
                    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
                    WHERE k.graphId IN ({','.join('?' for _ in ids)}) AND s.value LIKE '%mul_mat_vec_q<(ggml_type)12,%'
                    GROUP BY s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
                    k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory
                    ORDER BY sum(k.end-k.start) DESC"""
        launches[depth] = [{"kernel": row[0], "grid": list(row[1:4]), "block": list(row[4:7]),
                           "registers_per_thread": row[7], "static_shared_bytes": row[8], "dynamic_shared_bytes": row[9],
                           "calls": row[10], "total_ms": row[11]/1e6, "mean_us": row[11]/row[10]/1e3,
                           "min_us": row[12]/1e3, "max_us": row[13]/1e3} for row in connection.execute(query, ids)]
        connection.close()
    q4_params = sum(math.prod(t["ggml_dimensions"]) for t in tensors if t["ggml_type"] == 12)
    driver = ctypes.WinDLL("nvcuda.dll")
    if driver.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    hardware = {}
    for name, attribute in (("peak_memory_clock_khz", 36), ("memory_bus_width_bits", 37), ("l2_cache_bytes", 38), ("multiprocessors", 16)):
        value = ctypes.c_int()
        if driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, 0) != 0:
            raise RuntimeError(f"CUDA device attribute failed: {name}")
        hardware[name] = value.value
    hardware["theoretical_peak_memory_bandwidth_gb_s"] = 2 * hardware["peak_memory_clock_khz"] * 1000 * hardware["memory_bus_width_bits"] / 8 / 1e9
    pair_bytes = 2 * 12288 * 4096 // 256 * 144
    candidate = {"operation": "FFN up/gate matrix-vector products plus SwiGLU", "input_columns": 4096,
                 "output_rows": 12288, "weight_matrix_count": 2, "packed_weight_bytes_per_call": pair_bytes,
                 "mapping_evidence": "Unique Q4_K tensor shape with 12288 output rows is the FFN up/gate pair. CUDA graph fusion source supports gate/up/SwiGLU fusion. Profile grid 12288 appears exactly 36 times per decode step, matching layer count. Mapping is inferred from these facts; tensor pointers were not traced.",
                 "observations": {}}
    for depth, rows in launches.items():
        row = next(row for row in rows if row["grid"] == [12288, 1, 1] and "(bool)1," in row["kernel"])
        candidate["observations"][depth] = {"mean_profiled_kernel_us": row["mean_us"],
            "percent_of_summed_kernel_time": 100 * row["total_ms"] / profile[depth]["summed_kernel_time_ms"],
            "weight_bytes_over_profiled_time_gb_s": pair_bytes / (row["mean_us"] * 1000),
            "calls_per_decode_step": row["calls"] / profile[depth]["decode_steps"]}
    output = {"source_commit": "8bf88ba9368e70fe3ba1b0001b34e70ffc713716",
              "model_metadata": {k: v for k, v in metadata.items() if k.startswith("qwen3.") or k in ("general.architecture", "general.file_type")},
              "header_bytes_read": header_bytes, "tensor_count": len(tensors),
              "tensor_type_counts": dict(Counter(str(t["ggml_type"]) for t in tensors)),
              "q4_k_weight_parameters": q4_params, "q4_k_storage_bytes": q4_params//256*144,
              "hardware_driver_attributes": hardware, "first_microbenchmark_candidate": candidate,
              "tensor_groups": inventory, "q4_profile_launches": launches,
              "limitations": "GGUF dimensions list input dimension first. Launch geometry alone cannot uniquely identify a tensor, fusion operation, gate pointer, or memory bottleneck. Profile times include diagnostic overhead."}
    target = Path(__file__).resolve().parent.parent / "results/q4-k-investigation.json"
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Tensor count", len(tensors), "header bytes", header_bytes, "Q4_K stored bytes", output["q4_k_storage_bytes"])
    for group in inventory:
        if group["ggml_type"] == 12:
            print(group)
    for depth, rows in launches.items():
        for row in rows:
            print(depth, "fusion=" + row["kernel"].split('(bool)')[1].split(',')[0], row["grid"], row["block"], row["calls"], round(row["mean_us"], 2), row["registers_per_thread"])
