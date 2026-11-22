param(
    [ValidateSet('4B', '8B')][string]$ModelSize = '8B',
    [ValidateSet('f16', 'q8_0', 'q4_0')][string]$CacheType = 'f16',
    [string]$Prompt = 'Explain GPU memory bandwidth in two sentences. /no_think',
    [int]$Tokens = 128,
    [int]$Context = 2048,
    [switch]$NgramSpeculation,
    [switch]$AdaptiveNgram,
    [switch]$GroupedQ8Attention,
    [switch]$GroupedQ4Attention
)
$ErrorActionPreference = 'Stop'
if ($GroupedQ8Attention -and $GroupedQ4Attention) {
    throw 'Select one grouped attention format.'
}
$groupedAttention = $GroupedQ8Attention -or $GroupedQ4Attention
$combinedAttention = $groupedAttention -and $AdaptiveNgram
if ($groupedAttention) {
    $groupedCache = if ($GroupedQ4Attention) { 'q4_0' } else { 'q8_0' }
    if ($PSBoundParameters.ContainsKey('CacheType') -and $CacheType -ne $groupedCache) {
        throw "Selected grouped attention requires -CacheType $groupedCache."
    }
    $CacheType = $groupedCache
    $variant = $groupedCache.Substring(0, 2)
    $groupedBuildKey = "grouped_${variant}_sha256"
    $groupedFlag = 'LLM_OPT_ATTENTION_' + $variant.ToUpperInvariant()
}
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$cli = Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda\bin\llama-cli.exe'
if ($combinedAttention) {
    $candidate = Join-Path $env:LLM_OPT_ROOT 'build\llama-combined'
    $manifest = Join-Path $candidate 'build.json'
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw 'Prepare the combined runtime with python .\scripts\prepare_combined_runtime.py.'
    }
    if (-not (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).attention.$groupedBuildKey) {
        throw "Prepare the combined runtime from a build with grouped $groupedCache attention."
    }
    $cli = Join-Path $candidate 'bin\llama-cli.exe'
    $costPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'results\ngram-cost-8k-20261121T225729Z.json'
    $combinedCosts = (Get-Content -LiteralPath $costPath -Raw | ConvertFrom-Json).step_us -join ' '
} elseif ($AdaptiveNgram) {
    $candidate = Join-Path $env:LLM_OPT_ROOT 'build\llama-adaptive-ngram'
    if (-not (Test-Path -LiteralPath (Join-Path $candidate 'build.json'))) {
        throw 'Build the adaptive runtime first with .\scripts\build-adaptive-ngram.ps1.'
    }
    $cli = Join-Path $candidate 'bin\llama-cli.exe'
} elseif ($groupedAttention) {
    $candidate = Join-Path $env:LLM_OPT_ROOT 'build\llama-attention'
    $manifest = Join-Path $candidate 'build.json'
    if (-not (Test-Path -LiteralPath $manifest) -or
        -not (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).$groupedBuildKey) {
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
if ($groupedAttention) { $previousGrouped = [Environment]::GetEnvironmentVariable($groupedFlag) }
$previousPolicy = $env:LLM_OPT_NGRAM_POLICY
$previousCosts = $env:LLM_OPT_NGRAM_STEP_US
try {
    if ($combinedAttention) {
        $env:LLM_OPT_NGRAM_POLICY = 'ranked'
        $env:LLM_OPT_NGRAM_STEP_US = $combinedCosts
    } elseif ($AdaptiveNgram) {
        $env:LLM_OPT_NGRAM_BACKOFF = '1'
    }
    if ($groupedAttention) { [Environment]::SetEnvironmentVariable($groupedFlag, '1') }
    & $cli -m $model -ngl 99 -c $Context -n $Tokens -p $Prompt --temp 0 --seed 42 --single-turn -fa on -ctk $CacheType -ctv $CacheType @speculation
    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE" }
} finally {
    if ($AdaptiveNgram) { $env:LLM_OPT_NGRAM_BACKOFF = $previousBackoff }
    if ($groupedAttention) { [Environment]::SetEnvironmentVariable($groupedFlag, $previousGrouped) }
    if ($combinedAttention) {
        $env:LLM_OPT_NGRAM_POLICY = $previousPolicy
        $env:LLM_OPT_NGRAM_STEP_US = $previousCosts
    }
}
