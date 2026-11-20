param(
    [ValidateSet('4B', '8B')][string]$ModelSize = '8B',
    [ValidateRange(0, 32768)][int]$ContextDepth,
    [switch]$AllowBattery
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$battery = Get-CimInstance Win32_Battery | Select-Object -First 1
$powerState = if ($battery.BatteryStatus -eq 1) { 'battery' } elseif ($battery.BatteryStatus -eq 2) { 'AC' } else { 'unknown' }
if ($powerState -eq 'battery' -and -not $AllowBattery) {
    throw 'Plug in the laptop before benchmarking, or use -AllowBattery to label a battery-powered run.'
}
$benchmarkArgs = @('--size', $ModelSize.ToUpper(), '--power-state', $powerState)
if ($PSBoundParameters.ContainsKey('ContextDepth')) {
    $benchmarkArgs += @('--context-depth', $ContextDepth)
}
& python (Join-Path $PSScriptRoot 'benchmark.py') @benchmarkArgs
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed. Inspect the timestamped log in results.' }
