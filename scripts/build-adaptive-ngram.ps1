param([switch]$Budget)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$buildArgs = @()
if ($Budget) { $buildArgs += '--budget' }
& python (Join-Path $PSScriptRoot 'build-adaptive-ngram.py') @buildArgs
if ($LASTEXITCODE -ne 0) { throw 'Adaptive n-gram build failed.' }
