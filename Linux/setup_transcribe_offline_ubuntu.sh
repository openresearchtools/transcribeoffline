#!/usr/bin/env bash
# setup_transcribe_offline_ubuntu.sh
# Ubuntu x64 setup for "Transcribe Offline"
#
# - Requires system ffmpeg (we DO NOT install it; we print how)
# - Ensures Python 3.11 + tkinter + venv (sudo; Deadsnakes fallback)
# - Creates ~/Downloads/Transcribe_Offline with a venv
# - Installs pinned wheels with --no-deps (EXCLUDES torch/torchaudio)
# - Installs CPU-only torch/torchaudio from PyTorch CPU index (BEFORE any imports)
# - Hardens sound libs (libsndfile/portaudio)
# - Hardens ctranslate2 → CPU (fallback to 4.4.0 if needed)
# - Extracts llama.cpp (Ubuntu x64) to content/vendor/llama.cpp
# - Writes launchers that ALWAYS use the venv and CPU path
# - Creates Desktop, in-folder, and app-menu launchers using content/AppIcon.png
# - Fetches main.py, models.py, icon, and licenses from GitHub (Linux folder)
# - Auto-launches the Models Downloader at the end (background)
set -Eeuo pipefail

log()  { printf "[ %(%H:%M:%S)T ] %s\n" -1 "$*"; }
halt() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

# ----- Preflight -----
[[ "$(uname -s)" == "Linux" ]] || halt "This setup is for Linux only."
arch="$(uname -m)"
[[ "$arch" == "x86_64" || "$arch" == "amd64" ]] || halt "This setup expects x86_64/amd64. Found: $arch"

# ----- Resolve Downloads/Desktop dirs -----
if command -v xdg-user-dir >/dev/null 2>&1; then
  DL="$(xdg-user-dir DOWNLOAD || true)"; [[ -n "${DL:-}" ]] || DL="$HOME/Downloads"
  DESK="$(xdg-user-dir DESKTOP || true)"; [[ -n "${DESK:-}" ]] || DESK="$HOME/Desktop"
else
  DL="$HOME/Downloads"
  DESK="$HOME/Desktop"
fi
mkdir -p "$DL" "$DESK"
log "Downloads: $DL"
log "Desktop:   $DESK"

# ----- Project paths -----
PROJ_DIR="$DL/Transcribe_Offline"
CONTENT_DIR="$PROJ_DIR/content"
LICENSES_DIR="$CONTENT_DIR/licenses"
VENDOR_DIR="$CONTENT_DIR/vendor"
LLAMA_DIR="$VENDOR_DIR/llama.cpp"
MODELS_DIR="$CONTENT_DIR/models"

mkdir -p "$PROJ_DIR" "$CONTENT_DIR" "$LICENSES_DIR" "$VENDOR_DIR" "$LLAMA_DIR" "$MODELS_DIR"
log "Working in: $PROJ_DIR"
cd "$PROJ_DIR"

# Keep all model caches in-project
export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$MODELS_DIR"
export TRANSFORMERS_CACHE="$MODELS_DIR"

# =====================================================================
# 0) Fetch app files from GitHub (Linux folder)
# =====================================================================
RAW_BASE='https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/Linux'

# Ensure we can download
ensure_downloader() {
  if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then
    return 0
  fi
  log "Installing 'curl' (sudo)…"
  sudo apt-get update -y
  sudo apt-get install -y curl
}

# download_file URL OUTPATH [mode:text|bin]
download_file() {
  local url="$1"; local out="$2"; local mode="${3:-bin}"
  mkdir -p "$(dirname "$out")"
  local tmp="$out.__tmp__"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsSL -o "$tmp" "$url"; then
      rm -f "$tmp"; return 1
    fi
  else
    if ! wget -qO "$tmp" "$url"; then
      rm -f "$tmp"; return 1
    fi
  fi
  # For text mode, normalize line endings lightly (no-op for most Linux content)
  if [[ "$mode" == "text" ]]; then
    # Strip CR if present
    tr -d '\r' < "$tmp" > "${tmp}.t" && mv -f "${tmp}.t" "$tmp"
  fi
  mv -f "$tmp" "$out"
}

fetch_if_missing() {
  local url="$1"; local out="$2"; local mode="${3:-bin}"
  if [[ -s "$out" ]]; then
    log "Exists: $(realpath --relative-to="$PROJ_DIR" "$out") — skipping download"
    return 0
  fi
  log "Downloading: $url -> $out"
  if ! download_file "$url" "$out" "$mode"; then
    halt "Failed to download: $url"
  fi
}

ensure_downloader

# Files to fetch
fetch_if_missing "$RAW_BASE/main.py"                                   "$PROJ_DIR/main.py"                  text
fetch_if_missing "$RAW_BASE/models.py"                                 "$PROJ_DIR/models.py"                text
fetch_if_missing "$RAW_BASE/content/AppIcon.png"                       "$CONTENT_DIR/AppIcon.png"           bin
fetch_if_missing "$RAW_BASE/content/licenses/LICENSE.txt"              "$LICENSES_DIR/LICENSE.txt"          text
fetch_if_missing "$RAW_BASE/content/licenses/README.MD"                "$LICENSES_DIR/README.MD"            text
fetch_if_missing "$RAW_BASE/content/licenses/Third-Party-Licenses.txt" "$LICENSES_DIR/Third-Party-Licenses.txt" text

# Validate main.py presence
[[ -s "$PROJ_DIR/main.py" ]] || halt "main.py missing after download."

# =====================================================================
# 1) FFmpeg check (do NOT install; suggest command)
# =====================================================================
if ! command -v ffmpeg >/dev/null 2>&1; then
  cat <<'EOF'

FFmpeg is not installed on this system. The app depends on the system's ffmpeg.
We will NOT install it for you. Please install it, then re-run this setup.

Suggested commands (Ubuntu/Debian):
  sudo apt-get update
  sudo apt-get install -y ffmpeg

After installing, re-run:
  bash setup_transcribe_offline_ubuntu.sh

EOF
  exit 1
fi
log "FFmpeg found: $(command -v ffmpeg)"

# =====================================================================
# 2) Ensure Python 3.11 + tkinter + venv (with Deadsnakes fallback)
# =====================================================================
PY311=""

have_py311() { command -v python3.11 >/dev/null 2>&1; }

install_py311_via_apt() {
  log "Trying Ubuntu repos for Python 3.11 (sudo)…"
  set +e
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-tk
  rc=$?
  set -e
  return $rc
}

install_py311_via_deadsnakes() {
  log "Adding Deadsnakes PPA for Python 3.11 (sudo)…"
  set +e
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-tk
  rc=$?
  set -e
  return $rc
}

if have_py311; then
  PY311="python3.11"
else
  if ! install_py311_via_apt; then
    install_py311_via_deadsnakes || halt "Failed to install Python 3.11 via Deadsnakes."
  fi
  have_py311 || halt "Python 3.11 still not available after installation."
  PY311="python3.11"
fi

# Tkinter check on base 3.11
log "Checking Tkinter on base Python 3.11…"
if ! "$PY311" - <<'PY'
import tkinter as tk  # noqa: F401
print("OK")
PY
then
  halt "Tkinter not available in Python 3.11. Ensure 'python3.11-tk' is installed."
fi
log "Tkinter OK on base."

# =====================================================================
# 3) Create local venv & install pinned wheels (NO torch/torchaudio here)
# =====================================================================
VENV_DIR="$PROJ_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PY311" -m venv "$VENV_DIR"
fi
[[ -x "$VENV_PY" ]] || halt "Failed to create venv at $VENV_DIR"

log "Upgrading pip/setuptools/wheel in venv…"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

REQ_FILE="$PROJ_DIR/requirements-pinned.txt"
# IMPORTANT: torch and torchaudio are intentionally OMITTED here; we install CPU wheels later.
cat >"$REQ_FILE" <<'REQS'
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
# torch==2.8.0           <-- moved to CPU-only section
torch-audiomentations==0.12.0
torch-pitch-shift==1.2.5
# torchaudio==2.8.0      <-- moved to CPU-only section
torchmetrics==1.8.2
tqdm==4.67.1
transformers==4.56.1
typing-extensions==4.15.0
urllib3==2.5.0
whisperx==3.4.2
REQS

log "Installing pinned wheels into venv (pip --no-deps)…"
"$VENV_PY" -m pip install --no-deps -r "$REQ_FILE"

# Provide 'hf' shim if only 'huggingface-cli' exists
if [[ ! -x "$VENV_DIR/bin/hf" && -x "$VENV_DIR/bin/huggingface-cli" ]]; then
  log "Creating 'hf' shim in venv…"
  cat >"$VENV_DIR/bin/hf" <<'HFWRAP'
#!/usr/bin/env bash
exec "$(dirname "$0")/huggingface-cli" "$@"
HFWRAP
  chmod +x "$VENV_DIR/bin/hf"
fi

# =====================================================================
# 3a) Install CPU-only PyTorch & Torchaudio FIRST (no CUDA)
# =====================================================================
log "Installing CPU-only torch/torchaudio…"
"$VENV_PY" -m pip uninstall -y torch torchaudio || true
set +e
"$VENV_PY" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" "torchaudio==2.8.0"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  log "Falling back to torch/torchaudio 2.4.1 CPU…"
  "$VENV_PY" -m pip uninstall -y torch torchaudio || true
  "$VENV_PY" -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1" "torchaudio==2.4.1"
fi

# Verify CPU-only Torch
"$VENV_PY" - <<'PY'
import torch
print("TORCH", torch.__version__, "cuda_available", torch.cuda.is_available())
assert not torch.cuda.is_available(), "CUDA build of torch installed; expected CPU wheel"
PY

# =====================================================================
# 3b) HARDEN audio deps: ensure soundfile/sounddevice import OK
# =====================================================================
log "Verifying sound libs inside venv…"
if ! "$VENV_PY" - <<'PY'
try:
    import soundfile, sounddevice  # noqa: F401
except Exception:
    raise SystemExit(1)
print("OK")
PY
then
  log "Installing system audio libraries (sudo)…"
  sudo apt-get update
  sudo apt-get install -y libsndfile1 libportaudio2
  log "Re-checking Python wheels for sound libs…"
  "$VENV_PY" -m pip install --no-deps soundfile==0.13.1 sounddevice==0.5.2
  "$VENV_PY" - <<'PY'
import soundfile, sounddevice
print("OK", soundfile.__version__, sounddevice.__version__)
PY
fi

# =====================================================================
# 3c) HARDEN faster-whisper/ctranslate2 runtime (force CPU wheel if needed)
# =====================================================================
needs_fix=0
if ! "$VENV_PY" - <<'PY'
try:
    import faster_whisper  # noqa: F401
except Exception as e:
    print("IMPORT_FAIL:", e)
    raise SystemExit(2)
print("OK")
PY
then
  needs_fix=1
fi

if [[ "$needs_fix" == "1" ]]; then
  log "Installing system libs for CPU runtime (sudo)…"
  set +e
  sudo apt-get update
  sudo apt-get install -y libomp5 || sudo apt-get install -y "libomp5-18" || sudo apt-get install -y libomp-dev
  sudo apt-get install -y libzstd1 libnuma1 libgomp1
  set -e

  log "Forcing CPU-only CTranslate2 wheel…"
  "$VENV_PY" -m pip uninstall -y ctranslate2 || true
  "$VENV_PY" -m pip install --no-deps --only-binary=:all: "ctranslate2==4.6.0"

  # Re-check import; if still failing, fall back to 4.4.0 CPU wheel
  if ! "$VENV_PY" - <<'PY'
try:
    import faster_whisper  # noqa: F401
except Exception as e:
    import sys
    print("IMPORT_FAIL:", e)
    sys.exit(2)
print("OK")
PY
  then
    log "CPU wheel 4.6.0 still failed — falling back to 4.4.0 (CPU)…"
    "$VENV_PY" -m pip uninstall -y ctranslate2 || true
    "$VENV_PY" -m pip install --no-deps --only-binary=:all: "ctranslate2==4.4.0"

    "$VENV_PY" - <<'PY'
import faster_whisper
print("OK (CT2=4.4.0)")
PY
  fi
fi

# =====================================================================
# 4) llama.cpp (Ubuntu x64): download & extract to vendor
# =====================================================================
if ! command -v unzip >/dev/null 2>&1; then
  log "Installing 'unzip' (sudo)…"
  sudo apt-get update
  sudo apt-get install -y unzip
fi

LLAMA_ZIP_URL="https://github.com/ggml-org/llama.cpp/releases/download/b6435/llama-b6435-bin-ubuntu-x64.zip"
LLAMA_ZIP="$PROJ_DIR/llama-b6435-bin-ubuntu-x64.zip"

if [[ ! -s "$LLAMA_ZIP" ]]; then
  log "Downloading llama.cpp (Ubuntu x64)…"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$LLAMA_ZIP" "$LLAMA_ZIP_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$LLAMA_ZIP" "$LLAMA_ZIP_URL"
  else
    log "Installing 'curl' (sudo)…"
    sudo apt-get update && sudo apt-get install -y curl
    curl -L --fail -o "$LLAMA_ZIP" "$LLAMA_ZIP_URL"
  fi
fi
[[ -s "$LLAMA_ZIP" ]] || halt "Failed to download llama.cpp zip."

log "Extracting llama.cpp…"
TMP_LL="$(mktemp -d)"
unzip -q "$LLAMA_ZIP" -d "$TMP_LL"

# Copy tools + libs into vendor/llama.cpp (cover both bin/ and lib/ layouts)
if [[ -d "$TMP_LL/build/bin" ]]; then
  cp -a "$TMP_LL/build/bin/." "$LLAMA_DIR/"
fi
if [[ -d "$TMP_LL/build/lib" ]]; then
  cp -a "$TMP_LL/build/lib/." "$LLAMA_DIR/"
fi
# Fallback: unexpected layout
if [[ ! -e "$LLAMA_DIR/llama-cli" && ! -e "$LLAMA_DIR/bin/llama-cli" ]]; then
  cp -a "$TMP_LL/." "$LLAMA_DIR/" || true
fi
rm -rf "$TMP_LL"

# Make executables usable
find "$LLAMA_DIR" -maxdepth 1 -type f -name "llama*" -exec chmod +x {} \; || true

# Verify presence of llama-cli
if [[ ! -x "$LLAMA_DIR/llama-cli" && ! -x "$LLAMA_DIR/bin/llama-cli" ]]; then
  halt "llama-cli not found after extraction (checked $LLAMA_DIR and $LLAMA_DIR/bin)."
fi
log "llama.cpp OK."

# =====================================================================
# 5) Write launchers that ALWAYS use the venv (PATH + same Python)
# =====================================================================
RUN_MAIN="$PROJ_DIR/run_transcribe_offline.sh"
RUN_MODELS="$PROJ_DIR/download_models.sh"

cat >"$RUN_MAIN" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Project venv missing at $VENV_DIR. Re-run setup." >&2
  exit 1
fi

# Use the SAME venv Python and make its tools first on PATH (includes 'hf')
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$SCRIPT_DIR/content/vendor/llama.cpp:$PATH"
unset PYTHONHOME

# Keep model caches in-project
export HF_HOME="$SCRIPT_DIR/content/models"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"

# Misc envs used by the app
export TOKENIZERS_PARALLELISM=false
export KMP_DUPLICATE_LIB_OK=TRUE
# Ensure llama shared libs are discoverable if needed
export LD_LIBRARY_PATH="$SCRIPT_DIR/content/vendor/llama.cpp:${LD_LIBRARY_PATH:-}"
# Force CPU-only path for CT2 / Torch
export CUDA_VISIBLE_DEVICES=""

exec "$VENV_DIR/bin/python" -u "$SCRIPT_DIR/main.py" "$@"
SH
chmod +x "$RUN_MAIN"

cat >"$RUN_MODELS" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Project venv missing at $VENV_DIR. Re-run setup." >&2
  exit 1
fi

# Use the SAME venv Python and make its tools first on PATH (includes 'hf')
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$SCRIPT_DIR/content/vendor/llama.cpp:$PATH"
unset PYTHONHOME

# Keep model caches in-project
export HF_HOME="$SCRIPT_DIR/content/models"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"

export TOKENIZERS_PARALLELISM=false
export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH="$SCRIPT_DIR/content/vendor/llama.cpp:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""

exec "$VENV_DIR/bin/python" -u "$SCRIPT_DIR/models.py" "$@"
SH
chmod +x "$RUN_MODELS"

log "Launchers written:"
log "  $RUN_MAIN"
log "  $RUN_MODELS"

# =====================================================================
# 6) Desktop/menu launchers with icon (double-click friendly)
# =====================================================================
APP_ICON_PNG="$CONTENT_DIR/AppIcon.png"
if [[ -f "$APP_ICON_PNG" ]]; then
  APP_ICON="$APP_ICON_PNG"
else
  APP_ICON="utilities-terminal"  # fallback system icon
fi

# We use Tk's default WM class ("Tk") so we don't have to change your Python.
# If you later set a custom WM class in main.py, update STARTUP_WMCLASS here to match.
STARTUP_WMCLASS="Tk"

make_desktop() {
  local name="$1" exec_path="$2" icon="$3" out="$4" wmclass="$5" workdir="$6"
  # NOTE: Exec must be unquoted; Path sets working directory.
  cat >"$out" <<EOF
[Desktop Entry]
Type=Application
Name=$name
Comment=$name
Exec=$exec_path
Icon=$icon
Terminal=false
Path=$workdir
Categories=AudioVideo;Utility;
StartupNotify=true
StartupWMClass=$wmclass
X-GNOME-UsesNotifications=false
DBusActivatable=false
EOF
  chmod +x "$out"
}

# Desktop entries (may need right-click ▸ Allow Launching once)
DESK_MAIN="$DESK/Transcribe Offline.desktop"
DESK_MODELS="$DESK/Download Models.desktop"
make_desktop "Transcribe Offline" "$RUN_MAIN" "$APP_ICON" "$DESK_MAIN" "$STARTUP_WMCLASS" "$PROJ_DIR"
make_desktop "Download Models" "$RUN_MODELS" "$APP_ICON" "$DESK_MODELS" "$STARTUP_WMCLASS" "$PROJ_DIR"

# App menu entries
APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
APP_MAIN_DESKTOP="$APPS_DIR/transcribe-offline.desktop"
APP_MODELS_DESKTOP="$APPS_DIR/transcribe-offline-models.desktop"
make_desktop "Transcribe Offline" "$RUN_MAIN" "$APP_ICON" "$APP_MAIN_DESKTOP" "$STARTUP_WMCLASS" "$PROJ_DIR"
make_desktop "Download Models" "$RUN_MODELS" "$APP_ICON" "$APP_MODELS_DESKTOP" "$STARTUP_WMCLASS" "$PROJ_DIR"

# In-folder launchers (with icons)
IN_FOLDER_MAIN="$PROJ_DIR/Transcribe Offline.desktop"
IN_FOLDER_MODELS="$PROJ_DIR/Download Models.desktop"
make_desktop "Transcribe Offline" "$RUN_MAIN" "$APP_ICON" "$IN_FOLDER_MAIN" "$STARTUP_WMCLASS" "$PROJ_DIR"
make_desktop "Download Models" "$RUN_MODELS" "$APP_ICON" "$IN_FOLDER_MODELS" "$STARTUP_WMCLASS" "$PROJ_DIR"

# Refresh desktop DB (best effort)
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" || true

# Mark as trusted so GNOME doesn’t warn (best effort)
if command -v gio >/dev/null 2>&1; then
  gio set "$DESK_MAIN"          metadata::trusted true 2>/dev/null || true
  gio set "$DESK_MODELS"        metadata::trusted true 2>/dev/null || true
  gio set "$IN_FOLDER_MAIN"     metadata::trusted true 2>/dev/null || true
  gio set "$IN_FOLDER_MODELS"   metadata::trusted true 2>/dev/null || true
fi

# =====================================================================
# 7) FINAL INFO + AUTO-LAUNCH MODELS DOWNLOADER (background)
# =====================================================================
echo
log "DONE. Project: $PROJ_DIR"
log "Venv Python:  $VENV_PY"
echo
echo "Launch from Desktop:"
echo "  • Double-click 'Transcribe Offline' (you may need to right-click ▸ Allow Launching once)."
echo "Or from terminal:"
echo "  \"$RUN_MAIN\""
echo
echo "Model downloader:"
echo "  • Desktop icon 'Download Models' or:"
echo "  \"$RUN_MODELS\""
echo

# Auto-open the Model Downloader now (background) if models.py is present
if [[ -f "$PROJ_DIR/models.py" ]]; then
  log "Launching Models Downloader with venv…"
  nohup "$RUN_MODELS" >/dev/null 2>&1 &
else
  log "models.py not found in $PROJ_DIR — skipping auto-launch."
fi

