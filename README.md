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

## Native Windows development environment

The online setup uses user-local files under `%LOCALAPPDATA%\LLMOptimization`, outside OneDrive. The installed NVIDIA display driver is retained. CUDA 13.5 supports the existing MSVC 19.51 compiler; this build targets native `sm_89` code for the RTX 4050, allowing execution under CUDA 13.x minor-version compatibility rather than relying on newer PTX JIT support. New toolkit features or profiling may still require a newer driver.

To open a development shell and build:

```powershell
. .\scripts\enter-dev.ps1
.\scripts\verify-cuda.ps1
.\scripts\build-llama.ps1
.\scripts\run-model.ps1
```

The shell setup applies to the current PowerShell process. It supplies MSVC, CUDA, CMake, Ninja and installed profiler paths without modifying global PATH.

To reproduce the downloads on another Windows machine with compatible Visual Studio Build Tools and an NVIDIA driver:

```powershell
.\scripts\setup-offline.ps1
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\install_cuda.py --profilers
git clone https://github.com/ggml-org/llama.cpp.git "$env:LOCALAPPDATA\LLMOptimization\llama.cpp"
git -C "$env:LOCALAPPDATA\LLMOptimization\llama.cpp" checkout 8bf88ba9368e70fe3ba1b0001b34e70ffc713716
.\.venv\Scripts\python.exe scripts\download_model.py
.\scripts\build-llama.ps1
```

CUDA archives and model weights are checked against publisher SHA-256 values. The model's source revision, size, hash and license are saved beside the weights. The CUDA manifest and component licenses are retained in the toolkit directory. The build disables HTTPS support; use the download script for remote models and local paths for inference.

This is an environment validation, not a controlled performance baseline. Final benchmark comparisons need fixed workload settings, warmup, repeated runs, power/thermal controls and a quality evaluation.

See [the validation record](results/setup-validation.md) for verified tools and remaining profiler permissions.

## Qwen3 8B target and repeatable benchmarks

Qwen3 8B Q4_K_M is the primary optimization target. Keep Qwen3 4B Q4_K_M as the smaller reference model. Download the pinned 8B release and run it with:

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py --size 8B
.\scripts\run-model.ps1 -ModelSize 8B
```

`run-model.ps1` now defaults to 8B; pass `-ModelSize 4B` for the development model.

With the laptop plugged in and other GPU applications closed, benchmark each model sequentially:

```powershell
.\scripts\benchmark.ps1 -ModelSize 4B
.\scripts\benchmark.ps1 -ModelSize 8B
```

Each invocation runs 512- and 2048-token prompt processing plus separate 128-token generation tests, with five repetitions after default warmup. Settings are eight CPU threads, batch 512, microbatch 128, Flash Attention on, FP16 K/V cache and 99 requested GPU layers. The verbose log records actual GPU offload. The scripts capture device-wide GPU memory, temperature, utilization, power and clocks every 200 ms and save timestamped JSON, CSV, metadata and local logs under `results/`. Sampling can miss brief allocation peaks. These short synthetic tests do not measure time to first token, model quality or long-context generation speed.

The benchmark wrapper checks power state and requires an explicit `-AllowBattery` flag for a battery-powered run. Results from different power states should be compared separately. The pinned model revision and SHA-256 are saved in model-specific manifests alongside the weights.

## Context-depth experiment

Measure how generation changes as the KV cache fills:

```powershell
.\scripts\context-sweep.ps1 -ModelSize 8B
.\scripts\show-context-results.ps1 -ModelSize 8B
```

The default sweep runs sequentially at populated context depths 0, 512, 1024, 2048 and 4096 tokens. Each point generates 128 tokens five times, with the same Q4_K_M weights, FP16 K/V cache, Flash Attention and batch settings as the baseline. llama-bench's `-d` prefill and context-state restoration happen outside the generation timer. This isolates decode speed after a populated prefix; it does not measure time to first token. The prefix is a synthetic benchmark workload, not a quality evaluation or a real conversation.

Each point runs in its own process and saves `context-8b-q4-d<depth>-<UTC timestamp>` artifacts. Metadata records actual allocated context length and the runtime's CUDA KV-cache buffer size. The allocation accommodates depth plus 128 generated tokens and runtime rounding, so allocated tokens can exceed the populated depth. Device-wide memory sampling includes model loading, prefix processing, cache restoration and generation; the reported peak is not generation-only or an allocator-exact peak. The result viewer shows all completed points, including repeats, with UTC timestamps and power state.

Keep the laptop on AC power and avoid concurrent GPU applications or benchmarks. Inspect sampled peak memory, actual GPU offload in the logs, temperatures, and throughput variability before extending the sweep. A sudden speed collapse near capacity can indicate memory pressure or fallback and warrants investigation. A default-warmup short sweep does not establish thermal steady-state performance. The script stops if a run fails; it retains the failure log and preceding completed results.

To run one point, or extend to a longer depth after inspecting headroom:

```powershell
.\scripts\benchmark.ps1 -ModelSize 8B -ContextDepth 2048
.\scripts\context-sweep.ps1 -ModelSize 8B -Depths 8192
```

The context-depth path was verified at depth 512 on AC power: 37/37 reported layers offloaded to CUDA, allocated context 768 tokens, CUDA KV cache 108 MiB, sampled peak device memory 4753 MiB, and 36.37 +/- 0.26 generated tokens/sec across five repetitions. Other default sweep points remain to be collected.
