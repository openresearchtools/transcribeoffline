param(
  [string]$BundleDir = "",
  [string]$TargetTriple = "",
  [switch]$Locked = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($BundleDir)) {
  $BundleDir = Join-Path $repoRoot "artifacts\\bundles\\transcribe-offline-win-x64-bundle"
}

Write-Host "BundleDir: $BundleDir"

if (Test-Path $BundleDir) {
  Remove-Item -Recurse -Force $BundleDir
}
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

$env:CARGO_TARGET_DIR = Join-Path $repoRoot "artifacts\\target"
$buildArgs = @("build", "--release", "--bin", "transcribe-offline")
if ($Locked) {
  $buildArgs += "--locked"
}
if (-not [string]::IsNullOrWhiteSpace($TargetTriple)) {
  $buildArgs += @("--target", $TargetTriple)
}

Push-Location $repoRoot
try {
  cargo @buildArgs | Out-Host
} finally {
  Pop-Location
}

$targetRelease = if ([string]::IsNullOrWhiteSpace($TargetTriple)) {
  Join-Path $env:CARGO_TARGET_DIR "release"
} else {
  Join-Path (Join-Path $env:CARGO_TARGET_DIR $TargetTriple) "release"
}

$mainExe = Join-Path $targetRelease "transcribe-offline.exe"
if (!(Test-Path $mainExe)) { throw "Missing built app exe: $mainExe" }

Copy-Item -Force $mainExe (Join-Path $BundleDir "Transcribe Offline.exe")

# Runtime is intentionally not bundled; app downloads/repairs runtime in-app.
# Copy runtime manifest lookup files used by in-app runtime checks.
$manifestSrc = Join-Path $repoRoot "runtime-manifests"
if (Test-Path $manifestSrc) {
  Copy-Item -Recurse -Force $manifestSrc (Join-Path $BundleDir "runtime-manifests")
}

# Copy app-side license/notices docs shipped with this wrapper.
$licensesSrc = Join-Path $repoRoot "licenses"
if (Test-Path $licensesSrc) {
  Copy-Item -Recurse -Force $licensesSrc (Join-Path $BundleDir "licenses")
}
$rootLicense = Join-Path $repoRoot "LICENSE"
if (Test-Path $rootLicense) {
  Copy-Item -Force $rootLicense (Join-Path $BundleDir "LICENSE")
}

Write-Host "Bundle ready: $BundleDir"
