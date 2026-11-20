# Environment validation - November 20, 2026

Machine: Intel Core Ultra 9 185H, approximately 32 GB RAM, RTX 4050 Laptop GPU (6141 MiB dedicated VRAM, compute capability 8.9), Windows 11. NVIDIA driver 582.05 retained.

Installed development tools:

| Tool | Version |
| --- | --- |
| MSVC | 19.51.36256.0, existing Visual Studio Build Tools 2026 |
| CUDA compiler | 13.5.37, CUDA redistributable release 13.5.0 |
| CMake | 4.4.5 |
| Ninja package | 1.13.2 |
| Nsight Compute | 2026.4.0 |
| Nsight Systems | 2026.5.2.323 |

The project shell adds these tools to the current process PATH. Large files live under `%LOCALAPPDATA%\LLMOptimization`.

## Verified

- A native `sm_89` CUDA kernel compiled with the installed MSVC host compiler, launched on the RTX 4050, and returned all 256 expected values. Reproduce with `scripts/verify-cuda.ps1`.
- Nsight Systems captured that kernel and exported a readable `cuda_gpu_kern_sum` summary (one `fill_values(int *)` invocation). This verifies CUDA tracing; the tiny smoke kernel is not a meaningful performance benchmark.
- Nsight Compute launches and attaches, but hardware counter collection returns `ERR_NVGPUCTRPERM` for the current non-administrator user. Use administrator privileges or configure counter permissions before collecting those metrics. See [NVIDIA's counter access guidance](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
- Official Qwen3 4B Q4_K_M GGUF downloaded and SHA-256 verified. Source and hash are recorded in `development-model.json`; the Apache 2.0 license is retained beside the weights.

llama.cpp source revision: `8bf88ba9368e70fe3ba1b0001b34e70ffc713716`. The source checkout is unmodified. Build configuration uses Release, CUDA enabled, native `89-real` device code, and HTTPS disabled. The CLI, server and benchmark tool are the build targets.

## Model execution

The CLI, server and benchmark tool built successfully. The CLI answered the prompt "Explain GPU memory bandwidth in two sentences. /no_think" with two coherent sentences, using a 2048-token context and a 128-token generation limit.

The verbose benchmark log confirmed `offloaded 37/37 layers to GPU`, a CUDA model buffer of 2375.91 MiB, and a 304.28 MiB CPU-mapped model buffer. This is the runtime's GPU-layer offload result; it does not mean every byte of the model or runtime resides in VRAM. Buffer sizes are not a measurement of peak total VRAM.

A short setup check used this command after entering the project development shell:

```powershell
& "$env:LLM_OPT_ROOT\build\llama-cuda\bin\llama-bench.exe" -m "$env:LLM_OPT_ROOT\models\Qwen3-4B-Q4_K_M.gguf" -ngl 99 -p 128 -n 64 -r 3 -t 8 -b 512 -ub 128 -fa on -o json -v
```

The laptop was connected to AC power. Default benchmark warmup was enabled. The three-run means were 2583.45 tokens/sec for 128-token prompt processing and 60.44 tokens/sec for 64-token generation. Raw samples and configuration are in `setup-bench.json`. These synthetic tests exclude tokenization and sampling and are not a time-to-first-token measurement or a controlled optimization baseline. Peak VRAM, longer contexts, thermal steady state and model quality remain to be evaluated.
