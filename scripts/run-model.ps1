param(
    [string]$Prompt = 'Explain GPU memory bandwidth in two sentences. /no_think',
    [int]$Tokens = 128,
    [int]$Context = 2048
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$cli = Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda\bin\llama-cli.exe'
$model = Join-Path $env:LLM_OPT_ROOT 'models\Qwen3-4B-Q4_K_M.gguf'
& $cli -m $model -ngl 99 -c $Context -n $Tokens -p $Prompt --temp 0 --seed 42 --single-turn
if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
