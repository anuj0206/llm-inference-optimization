param([ValidateSet(0, 8192)][int]$ContextDepth = 0)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter-dev.ps1')
if ((Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus -eq 1) {
    throw 'Plug in the laptop before profiling.'
}
$profileRoot = Join-Path $env:LLM_OPT_ROOT 'profiles'
New-Item -ItemType Directory -Force -Path $profileRoot | Out-Null
$stamp = [datetime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$prefix = Join-Path $profileRoot "decode-8b-f16-d$ContextDepth-$stamp"
$resultRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'results'
$log = Join-Path $resultRoot "profile-8b-f16-d$ContextDepth-$stamp.log"
$bench = Join-Path $env:LLM_OPT_ROOT 'build\llama-cuda\bin\llama-bench.exe'
$model = Join-Path $env:LLM_OPT_ROOT 'models\Qwen3-8B-Q4_K_M.gguf'
& nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --output $prefix $bench -m $model -ngl 99 -p 0 -d $ContextDepth -n 128 -r 1 -t 8 -b 512 -ub 128 -fa on -ctk f16 -ctv f16 -o json --progress *> $log
if ($LASTEXITCODE -ne 0) { throw "Profiling failed. Inspect $log" }
& nsys export --type sqlite --output "$prefix.sqlite" "$prefix.nsys-rep" >> $log 2>&1
if ($LASTEXITCODE -ne 0) { throw "Profile export failed. Inspect $log" }
Write-Host "Saved $prefix.nsys-rep and $prefix.sqlite"
Write-Host 'The capture includes loading, warmup, prefill and generation; profiled timings are diagnostic.'
