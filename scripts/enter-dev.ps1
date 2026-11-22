$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$workRoot = Join-Path $env:LOCALAPPDATA 'LLMOptimization'
$toolkit = Join-Path $workRoot 'cuda\13.5.0'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio Build Tools were not found.' }
$vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw 'An x64 MSVC toolset is required.' }
& (Join-Path $vsRoot 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if (-not (Test-Path -LiteralPath (Join-Path $toolkit 'bin\nvcc.exe'))) {
    throw 'Install the local CUDA toolkit first: .\.venv\Scripts\python.exe scripts\install_cuda.py'
}
$env:CUDA_PATH = $toolkit
$env:CUDACXX = Join-Path $toolkit 'bin\nvcc.exe'
$env:PATH = (Join-Path $projectRoot '.venv\Scripts') + ';' + (Join-Path $toolkit 'bin') + ';' + (Join-Path $toolkit 'bin\x64') + ';' + $env:PATH
$cupti = Join-Path $toolkit 'lib\x64'
if (Test-Path -LiteralPath (Join-Path $cupti 'cupti64_13.dll')) {
    $env:PATH = $cupti + ';' + $env:PATH
}
foreach ($tool in @('nsight_compute', 'nsight_systems')) {
    $profilerRoot = Join-Path $workRoot "tools\$tool"
    if (Test-Path -LiteralPath $profilerRoot) {
        $executable = Get-ChildItem -LiteralPath $profilerRoot -Recurse -File -Filter $(if ($tool -eq 'nsight_compute') { 'ncu.exe' } else { 'nsys.exe' }) | Where-Object { $_.FullName -notmatch 'armv8|arm64|aarch64' } | Select-Object -First 1
        if ($executable) { $env:PATH = $executable.DirectoryName + ';' + $env:PATH }
    }
}
$env:LLM_OPT_ROOT = $workRoot
Write-Host "Development environment ready: $workRoot"
