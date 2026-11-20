param([int]$Jobs = 4)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$sourceRoot = Join-Path $env:LLM_OPT_ROOT 'llama.cpp'
$buildRoot = Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda'
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'CMakeLists.txt'))) {
    throw "Clone llama.cpp into $sourceRoot first."
}
# Native device code avoids requiring the installed driver to JIT newer PTX.
& cmake -S $sourceRoot -B $buildRoot -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89-real -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& cmake --build $buildRoot --parallel $Jobs --target llama-cli llama-bench llama-server
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp build failed.' }
