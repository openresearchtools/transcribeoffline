<# 
  setup_transcribe_offline.ps1 — Windows x64, PowerShell 5.1+, Python 3.11

  • Creates Transcribe_Offline in real Downloads (OneDrive-aware)
  • Finds base Python 3.11 (with Tkinter)
  • Creates .\.venv and installs pinned wheels there (pip --no-deps)
  • Downloads app files (main.py, models.py), vendors (FFmpeg, llama.cpp), and assets
  • Writes launchers that ACTIVATE the venv and prefer pythonw.exe (no console)
      - download_models.ps1/.bat  → models.py
      - run_transcribe_offline.ps1/.bat → main.py
  • Creates shortcuts with icon:
      - Desktop: "Transcribe Offline"
      - In-folder: "Transcribe Offline", "Download Models"
  • Auto-opens Models Downloader at the end (hidden; **direct pythonw.exe**, not a new PowerShell)
#>

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Log([string]$Message) { Write-Host ("[ {0} ] {1}" -f (Get-Date -f HH:mm:ss), $Message) }
function Halt([string]$Message) { throw "ERROR: $Message" }
function Safe-Remove([string]$Path) { if ($Path -and (Test-Path -LiteralPath $Path)) { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue } }

if (-not [Environment]::Is64BitOperatingSystem) { Halt "This setup is for Windows x64 only." }

# --- Resolve real Downloads (OneDrive-aware) ----------------------------------
function Get-DownloadsPath {
  try {
    $p = (New-Object -ComObject Shell.Application).NameSpace('shell:Downloads').Self.Path
    if ($p -and (Test-Path -LiteralPath $p)) { return (Resolve-Path $p).Path }
  } catch { }
  $cands = @()
  if ($env:USERPROFILE) { $cands += (Join-Path $env:USERPROFILE 'Downloads') }
  if ($env:OneDrive)    { $cands += (Join-Path $env:OneDrive    'Downloads') }
  foreach ($c in $cands) {
    if ($c -and (Test-Path -LiteralPath $c)) { return (Resolve-Path $c).Path }
  }
  $root = if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { $HOME } else { $env:USERPROFILE }
  $fallback = Join-Path $root 'Downloads'
  if (-not (Test-Path -LiteralPath $fallback)) { New-Item -ItemType Directory -Path $fallback | Out-Null }
  return (Resolve-Path $fallback).Path
}

$Downloads = Get-DownloadsPath
Write-Log "Downloads: $Downloads"

# --- Project paths -------------------------------------------------------------
$ProjDir = Join-Path $Downloads 'Transcribe_Offline'
$Dirs = [ordered]@{
  proj_dir = $ProjDir
  content  = Join-Path $ProjDir 'content'
  licenses = Join-Path $ProjDir 'content\licenses'
  vendor   = Join-Path $ProjDir 'content\vendor'
  ffmpeg   = Join-Path $ProjDir 'content\vendor\ffmpeg'
  llama    = Join-Path $ProjDir 'content\vendor\llama.cpp'
  models   = Join-Path $ProjDir 'content\models'
}
$Dirs.Values | ForEach-Object { if (-not (Test-Path -LiteralPath $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null } }
Set-Location $ProjDir
Write-Log "Working in: $ProjDir"

# --- Find base Python 3.11 ----------------------------------------------------
function Get-Python311 {
  try {
    $exe = (& py -3.11 -c "import sys, pathlib; print(pathlib.Path(sys.executable))" 2>$null).Trim()
    if ($exe -and (Test-Path -LiteralPath $exe)) { return $exe }
  } catch { }
  foreach ($cand in @('python','python3')) {
    try {
      $ver = (& $cand -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null).Trim()
      if ($ver -match '^3\.11(\.|$)') {
        $exe = (& $cand -c "import sys, pathlib; print(pathlib.Path(sys.executable))" 2>$null).Trim()
        if ($exe -and (Test-Path -LiteralPath $exe)) { return $exe }
      }
    } catch { }
  }
  foreach ($p in @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "$env:ProgramFiles(x86)\Python311\python.exe"
  )) { if (Test-Path -LiteralPath $p) { return $p } }
  $null
}

$PythonExe = Get-Python311
if (-not $PythonExe) { Halt "Python 3.11 not found. Install CPython 3.11 (with Tcl/Tk) and re-run." }
Write-Log "Base Python 3.11: $PythonExe"

# Ensure pip on base
try { & $PythonExe -m ensurepip --upgrade | Out-Null } catch { }

# --- Tkinter check on base ----------------------------------------------------
Write-Log "Checking Tkinter on base Python…"
try {
  $tk = (& $PythonExe -c "import tkinter as tk; print('OK')" 2>&1) -join "`n"
  if ($tk -notmatch '^OK') { Halt "Tkinter not available in the found Python 3.11." }
} catch { Halt "Tkinter not available in the found Python 3.11." }
Write-Log "Tkinter OK on base."

# --- Create project-local venv -------------------------------------------------
$VenvDir = Join-Path $ProjDir '.venv'
$VenvPy  = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPy)) {
  Write-Log "Creating virtual environment: $VenvDir"
  & $PythonExe -m venv "$VenvDir"
}
if (-not (Test-Path -LiteralPath $VenvPy)) { Halt "Failed to create virtual environment at $VenvDir." }

Write-Log "Upgrading pip/setuptools/wheel in venv…"
& $VenvPy -m pip install --upgrade pip setuptools wheel

# --- Pinned requirements (pip --no-deps) --------------------------------------
$ReqPath = Join-Path $ProjDir 'requirements-pinned.txt'
@"
asteroid-filterbanks==0.4.0
av==15.1.0
certifi==2025.8.3
cffi==2.0.0
charset-normalizer==2.1.0
colorama==0.4.6
colorlog==6.9.0
ctranslate2==4.6.0
cycler==0.12.1
einops==0.8.1
faster-whisper==1.2.0
filelock==3.19.1
fsspec==2025.9.0
huggingface-hub==0.34.4
HyperPyYAML==1.2.2
idna==3.10
joblib==1.5.2
julius==0.2.7
kiwisolver==1.4.9
lightning==2.5.5
lightning-utilities==0.15.2
matplotlib==3.10.6
mpmath==1.3.0
networkx==3.5
nltk==3.9.1
numpy==2.3.3
omegaconf==2.3.0
onnxruntime==1.22.1
optuna==4.5.0
packaging==25.0
pandas==2.3.2
pillow==11.3.0
primePy==1.3
pyannote-core==6.0.0
pyannote-database==6.0.0
pyannote-metrics==4.0.0
pyannote-pipeline==4.0.0
pyannote.audio==3.4.0
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytorch-lightning==2.5.5
pytorch-metric-learning==2.9.0
pytz==2025.2
PyYAML==6.0.2
regex==2025.9.1
requests==2.32.5
rich==14.1.0
safetensors==0.6.2
scikit-learn==1.7.2
scipy==1.16.2
semver==3.0.4
six==1.17.0
sortedcontainers==2.4.0
sounddevice==0.5.2
soundfile==0.13.1
speechbrain==1.0.3
sympy==1.14.0
tabulate==0.9.0
tensorboardX==2.6.4
threadpoolctl==3.6.0
tokenizers==0.22.0
torch==2.8.0
torch-audiomentations==0.12.0
torch-pitch-shift==1.2.5
torchaudio==2.8.0
torchmetrics==1.8.2
tqdm==4.67.1
transformers==4.56.1
typing-extensions==4.15.0
urllib3==2.5.0
whisperx==3.4.2
"@.Trim() | Set-Content -LiteralPath $ReqPath -Encoding ASCII

Write-Log "Installing pinned wheels into venv (pip --no-deps)…"
& $VenvPy -m pip install --no-deps -r $ReqPath
if ($LASTEXITCODE -ne 0) { Halt "pip install --no-deps into venv failed." }

# Tkinter sanity check in venv
Write-Log "Checking Tkinter in venv…"
try {
  $tkv = (& $VenvPy -c "import tkinter as tk; print('OK')" 2>&1) -join "`n"
  if ($tkv -notmatch '^OK') { Halt "Tkinter not available inside the venv (ensure base Python has Tcl/Tk)." }
} catch { Halt "Tkinter not available inside the venv." }

# --- Download files (Windows path on GitHub) ----------------------------------
function Download-File([string]$Url, [string]$Dest, [ValidateSet('Text','Binary')]$Mode='Binary') {
  Write-Log "Downloading: $Url"
  try {
    if ($Mode -eq 'Binary') {
      Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest -MaximumRedirection 5 -TimeoutSec 600
    } else {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -MaximumRedirection 5 -TimeoutSec 600
      Set-Content -LiteralPath $Dest -Value $resp.Content -Encoding UTF8
    }
  } catch { Halt "Failed to download: $Url" }
  if (-not (Test-Path -LiteralPath $Dest) -or ((Get-Item $Dest).Length -eq 0)) { Halt "Failed to download: $Url" }
}

$RAW = 'https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/Windows'
Download-File "$RAW/main.py"    (Join-Path $ProjDir 'main.py')
Download-File "$RAW/models.py"  (Join-Path $ProjDir 'models.py')
Download-File "$RAW/content/AppIcon.ico"                       (Join-Path $Dirs.content  'AppIcon.ico')
Download-File "$RAW/content/licenses/LICENSE.txt"              (Join-Path $Dirs.licenses 'LICENSE.txt') -Mode Text
Download-File "$RAW/content/licenses/README.md"                (Join-Path $Dirs.licenses 'README.md')   -Mode Text
Download-File "$RAW/content/licenses/Third-Party-Licenses.txt" (Join-Path $Dirs.licenses 'Third-Party-Licenses.txt') -Mode Text
if (-not (Test-Path -LiteralPath (Join-Path $ProjDir 'main.py'))) { Halt "main.py missing after download." }

# --- Vendors: FFmpeg (filtered) -----------------------------------------------
Add-Type -AssemblyName System.IO.Compression.FileSystem
$ffmpegZipUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2025-07-31-14-15/ffmpeg-N-120424-g03b9180fe3-win64-lgpl-shared.zip'
$ffmpegZip    = Join-Path $ProjDir 'ffmpeg.zip'
Download-File $ffmpegZipUrl $ffmpegZip
Write-Log "Extracting FFmpeg (filtered)…"
$tmpFF = Join-Path $ProjDir '__tmp_ffmpeg'
New-Item -ItemType Directory -Path $tmpFF -Force | Out-Null
[System.IO.Compression.ZipFile]::ExtractToDirectory($ffmpegZip, $tmpFF)
$allFF = Get-ChildItem -LiteralPath $tmpFF -Recurse -File
$keepPat = @('ffmpeg\.exe$','avcodec-.*\.dll$','avdevice-.*\.dll$','avfilter-.*\.dll$','avformat-.*\.dll$','avutil-.*\.dll$','swresample-.*\.dll$','swscale-.*\.dll$','^LICENSE(\.|$)')
$selFF = $allFF | Where-Object { $n=$_.Name; (($keepPat | Where-Object { $n -imatch $_ }).Count) -gt 0 }
New-Item -ItemType Directory -Path $Dirs.ffmpeg -Force | Out-Null
foreach ($f in $selFF) { Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $Dirs.ffmpeg $f.Name) -Force }
Safe-Remove $tmpFF; Safe-Remove $ffmpegZip
if (-not (Test-Path -LiteralPath (Join-Path $Dirs.ffmpeg 'ffmpeg.exe'))) { Halt "ffmpeg.exe not found after extraction." }
Write-Log "FFmpeg OK."

# --- Vendors: llama.cpp (filtered) --------------------------------------------
$llamaZipUrl = 'https://github.com/ggml-org/llama.cpp/releases/download/b6435/llama-b6435-bin-win-cpu-x64.zip'
$llamaZip    = Join-Path $ProjDir 'llama.zip'
Download-File $llamaZipUrl $llamaZip
Write-Log "Extracting llama.cpp (filtered)…"
$tmpLL = Join-Path $ProjDir '__tmp_llama'
New-Item -ItemType Directory -Path $tmpLL -Force | Out-Null
[System.IO.Compression.ZipFile]::ExtractToDirectory($llamaZip, $tmpLL)
$allLL = Get-ChildItem -LiteralPath $tmpLL -Recurse -File
$keepLLPat = @('llama-cli\.exe$','llama\.dll$','mtmd\.dll$','^ggml.*\.dll$','^libcurl-.*\.dll$','^libomp.*\.dll$','^LICENSE.*')
$selLL = $allLL | Where-Object { $n=$_.Name; (($keepLLPat | Where-Object { $n -imatch $_ }).Count) -gt 0 }
New-Item -ItemType Directory -Path $Dirs.llama -Force | Out-Null
foreach ($f in $selLL) { Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $Dirs.llama $f.Name) -Force }
Safe-Remove $tmpLL; Safe-Remove $llamaZip
if (-not (Test-Path -LiteralPath (Join-Path $Dirs.llama 'llama-cli.exe'))) { Halt "llama-cli.exe not found after extraction." }
Write-Log "llama.cpp OK."

# --- Write launcher .ps1 + .bat (prefer pythonw.exe; forward args) ------------
function New-Launcher {
  param(
    [Parameter(Mandatory)] [string]$Ps1Path,
    [Parameter(Mandatory)] [string]$TargetPyFile  # 'models.py' or 'main.py'
  )

  # SINGLE-QUOTED here-string (no expansion); we replace __TARGET_FILE__ below.
  $template = @'
# Auto-generated launcher. Activates venv and runs __TARGET_FILE__ without a console.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
$ErrorActionPreference = 'Stop'
function Halt([string]$m) { throw "ERROR: $m" }

$proj = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $proj) { $proj = (Get-Location).Path }

# Require project venv
$venvDir = Join-Path $proj '.venv'
$venvPy  = Join-Path $venvDir 'Scripts\python.exe'
$venvWPy = Join-Path $venvDir 'Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $venvPy)) { Halt 'Project virtual environment missing. Re-run setup.' }

# Activate venv
$activate = Join-Path $venvDir 'Scripts\Activate.ps1'
if (Test-Path -LiteralPath $activate) { . $activate }

# Vendors + caches
$ffmpeg = Join-Path $proj 'content\vendor\ffmpeg'
$llama  = Join-Path $proj 'content\vendor\llama.cpp'
$env:KMP_DUPLICATE_LIB_OK   = 'TRUE'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HOME                = Join-Path $proj 'content\models'
$env:HUGGINGFACE_HUB_CACHE  = $env:HF_HOME
$env:TRANSFORMERS_CACHE     = $env:HF_HOME
$env:PATH = ($ffmpeg + ';' + $llama + ';' + $env:PATH)

# Target
$app = Join-Path $proj '__TARGET_FILE__'
if (-not (Test-Path -LiteralPath $app)) { Halt '__TARGET_FILE__ missing.' }

# Args
$baseArgs = @('-u', $app)
$more = @()
if ($Args) { foreach ($x in $Args) { if ($x) { $more += $x } } }
$allArgs = $baseArgs + $more

# Prefer pythonw.exe (no console)
$pyExe = if (Test-Path -LiteralPath $venvWPy) { $venvWPy } else { $venvPy }

Start-Process -FilePath $pyExe -ArgumentList $allArgs -WorkingDirectory $proj -WindowStyle Hidden | Out-Null
'@

  $ps1 = $template.Replace('__TARGET_FILE__', $TargetPyFile)
  Set-Content -LiteralPath $Ps1Path -Value $ps1 -Encoding UTF8

  $batPath = [System.IO.Path]::ChangeExtension($Ps1Path, '.bat')
  $ps1Base = Split-Path -Leaf $Ps1Path
  $bat = @"
@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
start "" /b powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%SCRIPT_DIR%$ps1Base" %*
exit /b 0
"@
  Set-Content -LiteralPath $batPath -Value $bat -Encoding ASCII
  try { Unblock-File -Path $Ps1Path, $batPath -ErrorAction SilentlyContinue } catch { }
  Write-Log ("Wrote launcher: {0} and {1}" -f $Ps1Path, $batPath)
}

New-Launcher -Ps1Path (Join-Path $ProjDir 'download_models.ps1')        -TargetPyFile 'models.py'
New-Launcher -Ps1Path (Join-Path $ProjDir 'run_transcribe_offline.ps1') -TargetPyFile 'main.py'

# --- Shortcut helper ----------------------------------------------------------
function New-Shortcut {
  param(
    [Parameter(Mandatory)][string]$Path,
    [Parameter(Mandatory)][string]$Ps1Target,
    [string]$IconPath = $null,
    [string]$WorkingDir = $null
  )
  $wsh = New-Object -ComObject WScript.Shell
  $s = $wsh.CreateShortcut($Path)
  $s.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
  $s.Arguments  = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Ps1Target`""
  if ($WorkingDir) { $s.WorkingDirectory = $WorkingDir }
  if ($IconPath -and (Test-Path -LiteralPath $IconPath)) { $s.IconLocation = "$IconPath,0" }
  $s.WindowStyle = 7
  $s.Save()
}

function Get-DesktopPath {
  try { return [Environment]::GetFolderPath('Desktop') } catch { }
  try { return (New-Object -ComObject Shell.Application).NameSpace('shell:Desktop').Self.Path } catch { }
  return (Join-Path $env:USERPROFILE 'Desktop')
}

$IconPath = Join-Path $Dirs.content 'AppIcon.ico'
$Desktop  = Get-DesktopPath

# Desktop shortcut (main app)
New-Shortcut -Path (Join-Path $Desktop  'Transcribe Offline.lnk') `
             -Ps1Target (Join-Path $ProjDir 'run_transcribe_offline.ps1') `
             -IconPath $IconPath -WorkingDir $ProjDir

# In-folder aliases
New-Shortcut -Path (Join-Path $ProjDir 'Transcribe Offline.lnk') `
             -Ps1Target (Join-Path $ProjDir 'run_transcribe_offline.ps1') `
             -IconPath $IconPath -WorkingDir $ProjDir

New-Shortcut -Path (Join-Path $ProjDir 'Download Models.lnk') `
             -Ps1Target (Join-Path $ProjDir 'download_models.ps1') `
             -IconPath $IconPath -WorkingDir $ProjDir

# --- Auto-open Models Downloader (hidden, direct pythonw.exe) -----------------
Write-Log ""
Write-Log "Launching Models Downloader (hidden, direct)…"

# Match launcher env so downloads cache in content\models and vendors are on PATH
$env:KMP_DUPLICATE_LIB_OK   = 'TRUE'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HOME               = $Dirs.models
$env:HUGGINGFACE_HUB_CACHE = $Dirs.models
$env:TRANSFORMERS_CACHE    = $Dirs.models
$env:PATH = (Join-Path $Dirs.vendor 'ffmpeg') + ';' + (Join-Path $Dirs.vendor 'llama.cpp') + ';' + $env:PATH

$VenvWPy  = Join-Path $VenvDir 'Scripts\pythonw.exe'
$PyForRun = if (Test-Path -LiteralPath $VenvWPy) { $VenvWPy } else { $VenvPy }

Start-Process -FilePath $PyForRun -ArgumentList @(
  '-u', (Join-Path $ProjDir 'models.py')
) -WorkingDirectory $ProjDir -WindowStyle Hidden | Out-Null

# --- Final info ---------------------------------------------------------------
Write-Log ""
Write-Log ("DONE. Project: {0}" -f $ProjDir)
Write-Log ("Venv Python:       {0}" -f $VenvPy)
Write-Log ("Desktop shortcut:  {0}" -f (Join-Path $Desktop 'Transcribe Offline.lnk'))
Write-Log ("In-folder:         {0}" -f (Join-Path $ProjDir 'Transcribe Offline.lnk'))
Write-Log ("                   {0}" -f (Join-Path $ProjDir 'Download Models.lnk'))
Write-Log ("Launchers:         {0}" -f (Join-Path $ProjDir 'download_models.bat'))
Write-Log ("                   {0}" -f (Join-Path $ProjDir 'run_transcribe_offline.bat'))
