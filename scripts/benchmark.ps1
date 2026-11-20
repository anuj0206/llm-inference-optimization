param(
    [ValidateSet('4B', '8B')][string]$ModelSize = '8B',
    [switch]$AllowBattery
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$battery = Get-CimInstance Win32_Battery | Select-Object -First 1
$powerState = if ($battery.BatteryStatus -eq 1) { 'battery' } elseif ($battery.BatteryStatus -eq 2) { 'AC' } else { 'unknown' }
if ($powerState -eq 'battery' -and -not $AllowBattery) {
    throw 'Plug in the laptop before benchmarking, or use -AllowBattery to label a battery-powered run.'
}
& python (Join-Path $PSScriptRoot 'benchmark.py') --size $ModelSize --power-state $powerState
if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed. Inspect the timestamped log in results.' }
