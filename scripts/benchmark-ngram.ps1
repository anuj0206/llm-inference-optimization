param([switch]$Adaptive)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
if ((Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus -eq 1) {
    throw 'Plug in the laptop before benchmarking.'
}
if (Get-Process -Name llama-cli,llama-server,llama-bench,q4-microbench,q4-traffic,stream-read -ErrorAction SilentlyContinue) {
    throw 'Close model inference and benchmark processes first.'
}
$benchmarkArgs = @()
if ($Adaptive) { $benchmarkArgs += '--adaptive' }
& python (Join-Path $PSScriptRoot 'benchmark-ngram.py') @benchmarkArgs
if ($LASTEXITCODE -ne 0) { throw 'N-gram benchmark failed. Inspect the timestamped files in results.' }
