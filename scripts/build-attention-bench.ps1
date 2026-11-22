$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$project = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $env:LLM_OPT_ROOT 'build\llama-attention'
& python (Join-Path $PSScriptRoot 'build-attention-bench.py')
if ($LASTEXITCODE -ne 0) { throw 'CUDA dispatcher build failed.' }
$configure = @(
    '-S', (Join-Path $project 'benchmarks\attention'), '-B', (Join-Path $buildRoot 'harness'),
    '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', "-DSTOCK_SOURCE=$env:LLM_OPT_ROOT\llama.cpp",
    "-DSTOCK_RUNTIME_BUILD=$env:LLM_OPT_ROOT\build\llama-cuda", "-DATTENTION_BUILD=$buildRoot"
)
& cmake @configure
if ($LASTEXITCODE -ne 0) { throw 'Attention benchmark configuration failed.' }
& cmake --build (Join-Path $buildRoot 'harness')
if ($LASTEXITCODE -ne 0) { throw 'Attention benchmark build failed.' }
