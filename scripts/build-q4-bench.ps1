param([switch]$Counters)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
& python (Join-Path $PSScriptRoot 'prepare-q4-bench.py')
if ($LASTEXITCODE -ne 0) { throw 'Fixture/kernel preparation failed.' }
$buildRoot = Join-Path $env:LLM_OPT_ROOT 'build\q4-microbench'
$counterOption = if ($Counters) { 'ON' } else { 'OFF' }
& cmake -S (Join-Path $projectRoot 'benchmarks\q4_k') -B $buildRoot -G Ninja -DCMAKE_BUILD_TYPE=Release "-DQ4_TRAFFIC=$counterOption" "-DSTOCK_SOURCE=$env:LLM_OPT_ROOT\llama.cpp" "-DSTOCK_GENERATED=$buildRoot" "-DSTOCK_RUNTIME_BUILD=$env:LLM_OPT_ROOT\build\llama-cuda"
if ($LASTEXITCODE -ne 0) { throw 'Microbenchmark configure failed.' }
& cmake --build $buildRoot --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'Microbenchmark build failed.' }
