param(
    [ValidateSet('4B', '8B')][string]$ModelSize = '8B',
    [ValidateSet('f16', 'q8_0')][string]$CacheType = 'f16',
    [string]$Prompt = 'Explain GPU memory bandwidth in two sentences. /no_think',
    [int]$Tokens = 128,
    [int]$Context = 2048,
    [switch]$NgramSpeculation,
    [switch]$AdaptiveNgram,
    [switch]$GroupedQ8Attention
)
$ErrorActionPreference = 'Stop'
if ($GroupedQ8Attention) {
    if ($AdaptiveNgram) { throw 'Grouped Q8_0 attention and adaptive n-gram use separate experimental builds.' }
    if ($PSBoundParameters.ContainsKey('CacheType') -and $CacheType -ne 'q8_0') {
        throw 'Grouped Q8_0 attention requires -CacheType q8_0.'
    }
    $CacheType = 'q8_0'
}
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$cli = Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda\bin\llama-cli.exe'
if ($AdaptiveNgram) {
    $candidate = Join-Path $env:LLM_OPT_ROOT 'build\llama-adaptive-ngram'
    if (-not (Test-Path -LiteralPath (Join-Path $candidate 'build.json'))) {
        throw 'Build the adaptive runtime first with .\scripts\build-adaptive-ngram.ps1.'
    }
    $cli = Join-Path $candidate 'bin\llama-cli.exe'
}
if ($GroupedQ8Attention) {
    $candidate = Join-Path $env:LLM_OPT_ROOT 'build\llama-attention'
    $manifest = Join-Path $candidate 'build.json'
    if (-not (Test-Path -LiteralPath $manifest) -or
        -not (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).grouped_q8_sha256) {
        throw 'Build the grouped attention runtime with .\scripts\build-attention-bench.ps1.'
    }
    $cli = Join-Path $candidate 'bin\llama-cli.exe'
}
$model = Join-Path $env:LLM_OPT_ROOT "models\Qwen3-$ModelSize-Q4_K_M.gguf"
$speculation = @()
if ($NgramSpeculation -or $AdaptiveNgram) {
    $speculation = @('--spec-type', 'ngram-simple', '--spec-ngram-simple-size-n', '6', '--spec-ngram-simple-size-m', '8')
}
$previousBackoff = $env:LLM_OPT_NGRAM_BACKOFF
$previousGrouped = $env:LLM_OPT_ATTENTION_Q8
try {
    if ($AdaptiveNgram) { $env:LLM_OPT_NGRAM_BACKOFF = '1' }
    if ($GroupedQ8Attention) { $env:LLM_OPT_ATTENTION_Q8 = '1' }
    & $cli -m $model -ngl 99 -c $Context -n $Tokens -p $Prompt --temp 0 --seed 42 --single-turn -fa on -ctk $CacheType -ctv $CacheType @speculation
    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
} finally {
    if ($AdaptiveNgram) { $env:LLM_OPT_NGRAM_BACKOFF = $previousBackoff }
    if ($GroupedQ8Attention) { $env:LLM_OPT_ATTENTION_Q8 = $previousGrouped }
}
