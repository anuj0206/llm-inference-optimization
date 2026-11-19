$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $projectRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    if (Test-Path -LiteralPath $venvPath) {
        throw 'An incomplete .venv already exists. Inspect it before retrying.'
    }
    & python -m venv --without-pip $venvPath
    if ($LASTEXITCODE -ne 0) { throw 'Offline virtual environment creation failed.' }
}

& $venvPython (Join-Path $PSScriptRoot 'check_environment.py') --output (Join-Path $projectRoot 'results\environment.json')
if ($LASTEXITCODE -ne 0) { throw 'Environment inspection failed.' }
Write-Host 'Offline setup complete. No packages were downloaded.'
