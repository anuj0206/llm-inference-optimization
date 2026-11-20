# Project reading guide

Start here when returning to the project. The main question is how much useful LLM capability and speed we can retain on an RTX 4050 Laptop GPU with 6 GB dedicated VRAM.

## Read in this order

1. [Project overview and commands](../README.md): hardware, environment, downloads, baseline and context commands.
2. [Concept refresher](concepts.md): model size, weight precision, cache precision, prefill, decode and kernel timing.
3. [Experiment log](experiment-log.md): what we measured, what it established, and the next question.
4. [8B baseline](../results/qwen3-8b-baseline.md): matched 4B/8B speed and memory measurements.
5. [Cache and profiling screen](../results/optimization-screen.md): FP16 versus Q8 cache, profile breakdown and limited output checks.
6. [Kernel investigation](q4-k-investigation.md): exact Q4_K decode path, model shapes and first target.
7. [Isolated harness guide](q4-k-benchmark.md): correctness references, timing methods, commands and limitations.

Raw machine-readable measurements are under `results/`. Timestamped files preserve separate runs rather than replacing older evidence. Heavy weights, toolchains, compiled output and full profiler captures live under `%LOCALAPPDATA%\LLMOptimization`, outside OneDrive, and are excluded from Git. Each report identifies its relevant raw artifacts.

## What counts as a contribution

Stock-runtime configuration changes establish useful operating points. Custom CUDA work must be measured against the same stock operation, preserving numerical correctness and then improving actual model inference. Keep weight quantization, KV-cache quantization, and kernel execution improvements distinct in reports.

## Documentation convention

For each experiment, retain commands/settings, source/model identifiers, raw data, interpretation and limitations. Add a short entry to the experiment log. Update a claim when evidence changes, and distinguish measurements from estimates or hypotheses. Record failed correctness runs rather than silently replacing them with passing output.
