$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
if ((Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus -eq 1) {
    throw 'Plug in the laptop before the quality comparison.'
}
& python (Join-Path $PSScriptRoot 'quality-check.py')
if ($LASTEXITCODE -ne 0) { throw 'Quality screen failed. Inspect the quality log files in results.' }
