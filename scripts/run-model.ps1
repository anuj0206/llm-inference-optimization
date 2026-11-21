param(
    [ValidateSet('4B', '8B')][string]$ModelSize = '8B',
    [ValidateSet('f16', 'q8_0')][string]$CacheType = 'f16',
    [string]$Prompt = 'Explain GPU memory bandwidth in two sentences. /no_think',
    [int]$Tokens = 128,
    [int]$Context = 2048,
    [switch]$NgramSpeculation,
    [switch]$AdaptiveNgram
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$cli = Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda\bin\llama-cli.exe'
if ($AdaptiveNgram) {
    $candidate = Join-Path $env:LLM_OPT_ROOT 'build\llama-adaptive-ngram'
    if (-not (Test-Path -LiteralPath (Join-Path $candidate 'build.json'))) {
        throw 'Build the adaptive runtime first with .\scripts\build-adaptive-ngram.ps1.'
    }
    $cli = Join-Path $candidate 'bin\llama-cli.exe'
}
$model = Join-Path $env:LLM_OPT_ROOT "models\Qwen3-$ModelSize-Q4_K_M.gguf"
$speculation = @()
if ($NgramSpeculation -or $AdaptiveNgram) {
    $speculation = @('--spec-type', 'ngram-simple', '--spec-ngram-simple-size-n', '6', '--spec-ngram-simple-size-m', '8')
}
$previousBackoff = $env:LLM_OPT_NGRAM_BACKOFF
try {
    if ($AdaptiveNgram) { $env:LLM_OPT_NGRAM_BACKOFF = '1' }
    & $cli -m $model -ngl 99 -c $Context -n $Tokens -p $Prompt --temp 0 --seed 42 --single-turn -fa on -ctk $CacheType -ctv $CacheType @speculation
    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
} finally {
    if ($AdaptiveNgram) { $env:LLM_OPT_NGRAM_BACKOFF = $previousBackoff }
}
