param(
    [ValidateSet('4B', '8B')][string]$ModelSize = '8B',
    [int[]]$Depths = @(0, 512, 1024, 2048, 4096),
    [switch]$AllowBattery
)
$ErrorActionPreference = 'Stop'
foreach ($depth in $Depths) {
    if ($depth -lt 0 -or $depth -gt 32768) { throw 'Each depth must be between 0 and 32768 tokens.' }
}
Write-Host 'Each depth gets five 128-token generation runs after default warmup.'
Write-Host 'GPU memory sampling includes loading, prefill and generation. The sweep stops on failure.'
foreach ($depth in $Depths) {
    Write-Host "`n=== $ModelSize, populated context: $depth tokens ==="
    & (Join-Path $PSScriptRoot 'benchmark.ps1') -ModelSize $ModelSize -ContextDepth $depth -AllowBattery:$AllowBattery
}
& (Join-Path $PSScriptRoot 'show-context-results.ps1') -ModelSize $ModelSize
