# Saves a picture of every tab of the desktop UI (offscreen, so nothing appears on screen) into build\shots.
# The data folder is a throw-away one under build\uihome, so the real settings and models are not touched; pass -Home to
# look at a prepared one instead. Fonts come from the Windows font folder because the offscreen platform has none.
param(
    [string]$Config = "Release",
    [string]$Out = "$PSScriptRoot\..\build\shots",
    [string]$Home = "$PSScriptRoot\..\build\uihome"
)
$exe = Join-Path $PSScriptRoot "..\build\bin\x64\$Config\GraphicsHost.exe"
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:\Windows\Fonts"
$env:GRAPHICSHOST_HOME = (Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force $Home)).Path
& $exe --screenshot-dir (New-Item -ItemType Directory -Force $Out).FullName
Get-ChildItem $Out -Filter *.png | Select-Object Name, Length
