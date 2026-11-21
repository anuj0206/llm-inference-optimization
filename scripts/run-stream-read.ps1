$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')

$battery = Get-CimInstance Win32_Battery | Select-Object -First 1
if ($battery.BatteryStatus -eq 1) { throw 'Plug in the laptop before benchmarking.' }
if (Get-Process -Name llama-cli,llama-server,llama-bench,q4-microbench,stream-read -ErrorAction SilentlyContinue) {
    throw 'Close model inference and benchmark processes first.'
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $projectRoot 'benchmarks\stream-read.cu'
$buildRoot = Join-Path $env:LLM_OPT_ROOT 'build'
$executable = Join-Path $buildRoot 'stream-read.exe'
Push-Location $buildRoot
try {
    & nvcc -std=c++17 -O3 -Xcompiler /O2 -arch=sm_89 $source -o $executable
    if ($LASTEXITCODE -ne 0) { throw 'Streaming benchmark compilation failed.' }
} finally {
    Pop-Location
}

$stamp = [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$output = Join-Path $projectRoot "results\stream-read-$stamp.csv"
$gpuFields = '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw'
$before = nvidia-smi $gpuFields --format=csv,noheader
$started = [datetime]::UtcNow.ToString('o')
$rows = & $executable
if ($LASTEXITCODE -ne 0) { throw 'Streaming benchmark failed.' }
$finished = [datetime]::UtcNow.ToString('o')
$after = nvidia-smi $gpuFields --format=csv,noheader
$rows | Set-Content -LiteralPath $output -Encoding ascii
@{
    started_at_utc = $started
    finished_at_utc = $finished
    battery_status = $battery.BatteryStatus
    power_state = if ($battery.BatteryStatus -eq 2) { 'AC' } else { 'unknown' }
    source_sha256 = (Get-FileHash -LiteralPath $source).Hash.ToLower()
    executable_sha256 = (Get-FileHash -LiteralPath $executable).Hash.ToLower()
    gpu_snapshot_fields = $gpuFields -replace '^--query-gpu=', ''
    gpu_before = $before
    gpu_after = $after
} | ConvertTo-Json | Set-Content -LiteralPath ($output -replace '\.csv$', '.run.json') -Encoding utf8
Write-Host "Saved $output"
