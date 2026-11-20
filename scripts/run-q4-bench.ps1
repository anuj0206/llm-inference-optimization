$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
$battery = Get-CimInstance Win32_Battery | Select-Object -First 1
if ($battery.BatteryStatus -eq 1) { throw 'Plug in the laptop before benchmarking.' }
$powerState = if ($battery.BatteryStatus -eq 2) { 'AC' } else { 'unknown' }
if (Get-Process -Name llama-cli,llama-server,llama-bench,q4-microbench -ErrorAction SilentlyContinue) { throw 'Close model inference/benchmark processes first.' }
$root = Join-Path $env:LLM_OPT_ROOT 'build\q4-microbench'
$exe = Join-Path $root 'q4-microbench.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build the harness first: .\scripts\build-q4-bench.ps1' }
$env:PATH = (Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda\bin') + ';' + $env:PATH
$projectRoot = Split-Path -Parent $PSScriptRoot
$stamp = [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$output = Join-Path $projectRoot "results\q4-microbench-$stamp.json"
$before = nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used --format=csv,noheader
$started = [datetime]::UtcNow.ToString('o')
& $exe (Join-Path $root 'fixture.json') $output
$runExit = $LASTEXITCODE
$after = nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used --format=csv,noheader
@{
    started_at_utc = $started
    finished_at_utc = [datetime]::UtcNow.ToString('o')
    power_state = $powerState
    battery_status = $battery.BatteryStatus
    runner_sha256 = (Get-FileHash -LiteralPath (Join-Path $projectRoot 'benchmarks\q4_k\benchmark.cu') -Algorithm SHA256).Hash.ToLower()
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLower()
    gpu_before = $before
    gpu_after = $after
    exit_code = $runExit
} | ConvertTo-Json | Set-Content -LiteralPath ($output -replace '\.json$', '.run.json') -Encoding utf8
if ($runExit -ne 0) { throw "Harness failed. Inspect $output" }
Write-Host "Saved $output"
