# Measures the peak dedicated GPU memory of one spike-vulkan run, using the Windows "GPU Adapter Memory" counter,
# read with the .NET PerformanceCounter class (about 1 ms per read; Get-Counter needs a second per sample).
# Used to calibrate engine/FitEstimator. Example:
#   powershell -File tools\measure-vram.ps1 -Model models\v1-5-pruned-emaonly.safetensors -Size 512x512 -Steps 8
param(
    [Parameter(Mandatory)] [string]$Model,
    [Parameter(Mandatory)] [string]$Size,
    [int]$Steps = 8,
    [string]$Extra = "",
    [string]$Exe = ""
)
if (-not $Exe) { $Exe = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\build\bin\x64\Release\spike-vulkan.exe" }

$category = New-Object System.Diagnostics.PerformanceCounterCategory('GPU Adapter Memory')
$counters = foreach ($instance in $category.GetInstanceNames()) {
    New-Object System.Diagnostics.PerformanceCounter('GPU Adapter Memory', 'Dedicated Usage', $instance, $true)
}
function Read-Max { ($counters | ForEach-Object { $_.NextValue() } | Measure-Object -Maximum).Maximum / 1MB }

[void](Read-Max)
Start-Sleep -Milliseconds 300
$baseline = Read-Max
$out = Join-Path $env:TEMP 'vram-out.txt'; $err = Join-Path $env:TEMP 'vram-err.txt'
$arguments = @('--model', $Model, '--backend', 'vulkan0', '--steps', $Steps, '--size', $Size, '--seed', 1, '--out', (Join-Path $env:TEMP 'vram-probe.png'))
if ($Extra) { $arguments += $Extra.Split(' ') }
$process = Start-Process -FilePath $Exe -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
$peak = $baseline; $reads = 0
while (-not $process.HasExited) {
    $now = Read-Max; $reads++
    if ($now -gt $peak) { $peak = $now }
}
$result = (Get-Content $out | Select-String 'RESULT').Line
$weights = (Get-Content $err | Select-String 'total params memory size = ([0-9.]+)MB').Matches | ForEach-Object { $_.Groups[1].Value } | Select-Object -First 1
[pscustomobject]@{
    Model = Split-Path $Model -Leaf; Size = $Size; Reads = $reads; BaselineMB = [int]$baseline; PeakMB = [int]$peak
    GrowthMB = [int]($peak - $baseline); WeightsMB = $weights; Result = $result
} | Format-List
