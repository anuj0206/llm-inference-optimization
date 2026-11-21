$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
& python (Join-Path $PSScriptRoot 'build-adaptive-ngram.py')
if ($LASTEXITCODE -ne 0) { throw 'Adaptive n-gram build failed.' }
