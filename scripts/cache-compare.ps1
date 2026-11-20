$ErrorActionPreference = 'Stop'
$benchmark = Join-Path $PSScriptRoot 'benchmark.ps1'
& $benchmark -ModelSize 8B -ContextDepth 0 -CacheType f16
& $benchmark -ModelSize 8B -ContextDepth 0 -CacheType q8_0
& $benchmark -ModelSize 8B -ContextDepth 8192 -CacheType q8_0
& $benchmark -ModelSize 8B -ContextDepth 8192 -CacheType f16
# Repeat the long-context comparison in the opposite cache order.
& $benchmark -ModelSize 8B -ContextDepth 8192 -CacheType f16
& $benchmark -ModelSize 8B -ContextDepth 8192 -CacheType q8_0
