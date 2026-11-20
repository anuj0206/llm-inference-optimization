$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$outputRoot = Join-Path $env:LLM_OPT_ROOT 'build'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$executable = Join-Path $outputRoot 'cuda-smoke.exe'
Push-Location $outputRoot
try {
    & nvcc -gencode arch=compute_89,code=sm_89 (Join-Path $PSScriptRoot 'cuda-smoke.cu') -o $executable
    if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke compilation failed.' }
    & $executable
    if ($LASTEXITCODE -ne 0) { throw 'CUDA device verification failed.' }
} finally {
    Pop-Location
}
