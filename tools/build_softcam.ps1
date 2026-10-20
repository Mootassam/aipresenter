<#
.SYNOPSIS
  Builds softcam (MIT, https://github.com/tshino/softcam) and copies the DLL + installer to
  third_party/softcam. Needs Visual Studio 2022 Build Tools with the "Desktop development with
  C++" workload. GitHub Actions does the same thing (.github/workflows/softcam.yml) if you don't
  want to install the compiler.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\build_softcam.ps1
  powershell -ExecutionPolicy Bypass -File tools\build_softcam.ps1 -Register   # admin prompt
#>
param(
    [string]$Version = "v1.8.1",
    [switch]$Register
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $repo "third_party\softcam"
$work = Join-Path $env:TEMP "softcam-build"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio Build Tools not found. See docs/PHASE1.md." }
$msbuild = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild not found. Install the 'Desktop development with C++' workload." }

if (Test-Path $work) { Remove-Item -Recurse -Force $work }
git clone --depth 1 --branch $Version https://github.com/tshino/softcam.git $work
if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

& $msbuild (Join-Path $work "softcam.sln") /m /p:Configuration=Release /p:Platform=x64 /v:minimal
if ($LASTEXITCODE -ne 0) { throw "build failed" }

New-Item -ItemType Directory -Force $dest | Out-Null
foreach ($name in "softcam.dll", "softcam_installer.exe") {
    $f = Get-ChildItem -Path $work -Recurse -Filter $name |
        Where-Object { $_.FullName -match "x64" -and $_.FullName -match "Release" } |
        Select-Object -First 1
    if (-not $f) { throw "$name not found in build output" }
    Copy-Item $f.FullName $dest -Force
}
Copy-Item (Join-Path $work "LICENSE") (Join-Path $dest "LICENSE.softcam.txt") -Force
Write-Host "softcam built into $dest"

if ($Register) {
    # Registration writes to HKLM, so Windows shows one admin (UAC) prompt.
    Start-Process -Verb RunAs -Wait -FilePath (Join-Path $dest "softcam_installer.exe") `
        -ArgumentList "register", "`"$(Join-Path $dest 'softcam.dll')`""
    Write-Host "Registered. The camera appears as 'DirectShow Softcam' in OBS / Zoom / Discord."
}
