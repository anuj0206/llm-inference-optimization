# Hardware-Constrained LLM Inference Optimization

Optimize local LLM inference on an RTX 4050 Laptop GPU with a 6 GB VRAM budget. Use llama.cpp, C++ and CUDA to investigate quantized kernels and KV-cache memory efficiency.

## Scope

- Development model: Qwen3 4B.
- Target model: Qwen3 8B at Q4 or Q3 precision, subject to measured memory use and quality.
- Establish stock llama.cpp baselines before changing kernels.
- Compare the same quantized model on stock and modified runtimes to isolate kernel improvements.
- Compare precision levels separately; FP16 baselines may require a smaller model or CPU offload. Record offload explicitly.

Track prefill/decode tokens per second, time to first token, peak dedicated VRAM, kernel latency, and quality on a fixed evaluation set. Record model hash, runtime commit, build flags, context length, cache precision, GPU offload, repetitions, and laptop power/thermal conditions with results. System RAM fallback does not count as fitting within dedicated VRAM.

## Offline setup

From PowerShell in this directory:

```powershell
.\scripts\setup-offline.ps1
.\.venv\Scripts\python.exe .\scripts\check_environment.py
```

The setup uses the installed Python, creates an isolated environment without pip or third-party packages, and saves an ignored local machine report. It does not download software or change global settings. Activation is optional; use the environment's Python path directly.

## Initial machine audit — November 19, 2026

| Component | Status |
| --- | --- |
| GPU | NVIDIA RTX 4050 Laptop GPU; 6141 MiB reported; compute capability 8.9 |
| NVIDIA driver | 582.05; GPU visible through nvidia-smi |
| CPU / RAM | Intel Core Ultra 9 185H / approximately 32 GB |
| Git / Python / uv | Installed; Python 3.13 used for offline scripts |
| C++ | Visual Studio Build Tools 2026, MSVC 14.51; Windows SDK 10.0.26100.0 |
| CMake / Ninja | Not found in PATH or the usual Visual Studio CMake locations |
| CUDA Toolkit / profilers | nvcc, Nsight Systems and Nsight Compute not found in checked locations/PATH |
| llama.cpp / model weights | Deferred |

An NVIDIA driver is present, but that does not supply the CUDA compiler. CUDA Toolkit and MSVC/driver compatibility still need verification before choosing installer versions.

## Continue on home Wi-Fi

1. Choose a CUDA Toolkit compatible with the installed NVIDIA driver and a supported MSVC host compiler. Add a supported compiler toolset if needed. Avoid forcing unsupported compiler versions.
2. Install CMake and Ninja (or the Visual Studio CMake tools); install CUDA development tools. Add Nsight profilers when needed.
3. Clone llama.cpp into `external/llama.cpp`, record its commit, and preserve a stock baseline before modifications.
4. In a compatible Visual Studio developer shell, configure and build CUDA inference using the upstream instructions. Typical commands, to verify against the selected revision:

   ```powershell
   cmake -S external/llama.cpp -B build/llama-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build/llama-cuda --parallel 4
   ```

5. Download only the first development model: Qwen3 4B GGUF Q4_K_M, with its source revision, license and SHA-256 recorded. Expect multiple GB. Add 8B and other precisions after the baseline works.
6. Verify CUDA backend loading and actual GPU offload, then collect repeated stock benchmarks after warmup. Start with a short context and increase it while measuring memory.

Use a directory outside OneDrive for large models, build trees, and profiler traces if possible; pass their absolute paths to tools. No PyTorch, Conda, Docker image or separate Linux distribution is required for this initial native Windows plan. Revisit the platform choice if profiler/toolchain constraints warrant it.

Upstream references: [llama.cpp build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), [CUDA Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

## Layout

- `scripts/`: offline setup and environment inspection.
- `results/`: benchmark artifacts; local environment reports are ignored.
- `external/`, `build/`, `models/`: ignored locations for future source, build output and weights.

No inference baseline has been run yet.
