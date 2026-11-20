param([ValidateSet('4B', '8B')][string]$ModelSize = '8B')
$ErrorActionPreference = 'Stop'
$resultsRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'results'
$rows = Get-ChildItem -LiteralPath $resultsRoot -Filter "context-$($ModelSize.ToLower())-q4-d*.metadata.json" | ForEach-Object {
    $meta = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
    $dataPath = $_.FullName -replace '\.metadata\.json$', '.json'
    $data = @(Get-Content -LiteralPath $dataPath -Raw | ConvertFrom-Json)[0]
    [PSCustomObject]@{
        DepthTokens = $data.n_depth
        AllocatedTokens = ($meta.allocated_context_tokens | Measure-Object -Maximum).Maximum
        TokensPerSec = [math]::Round($data.avg_ts, 2)
        StdDev = [math]::Round($data.stddev_ts, 2)
        MsPerToken = [math]::Round(1000 / $data.avg_ts, 2)
        PeakVRAMMiB = $meta.device_peak_memory_used_mib
        KVCacheMiB = ($meta.cuda_kv_buffer_mib | Measure-Object -Maximum).Maximum
        MaxTempC = $meta.device_max_temperature_c
        Power = $meta.power_state
        FinishedUTC = ([datetime]$meta.finished_at_utc).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss') + 'Z'
    }
}
if (-not $rows) { Write-Host 'No completed context-depth results yet.'; return }
$rows | Sort-Object DepthTokens, FinishedUTC | Format-Table -AutoSize
