#!/usr/bin/env bash
set -euo pipefail

# ===== helpers =====
log(){ printf "\n%s\n" "$*"; }
err(){ printf "\nERROR: %s\n" "$*" >&2; }

# ===== 0) macOS + CLT =====
if [[ "$(uname -s)" != "Darwin" ]]; then
  err "This script is for macOS."
  exit 1
fi

# Ensure Xcode Command Line Tools (clang, make, SDK headers)
if ! xcode-select -p >/dev/null 2>&1; then
  log "Xcode Command Line Tools not found. Starting installation..."
  xcode-select --install || true
  echo
  echo "A system dialog should appear to install the Command Line Tools."
  echo "Please complete that install, then re-run this script."
  exit 1
else
  log "Xcode Command Line Tools found."
fi

# ===== 1) Homebrew (auto-install if missing) =====
if ! command -v brew >/dev/null 2>&1; then
  log "Homebrew not found. Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ "$(uname -m)" == "arm64" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
  else
    eval "$(/usr/local/bin/brew shellenv)"
    echo 'eval "$(/usr/local/bin/brew shellenv)"' >> "$HOME/.zprofile"
  fi
else
  log "Homebrew found."
fi

log "Updating Homebrew..."
brew update

# ===== 2) Homebrew Python + Tkinter =====
log "Installing Homebrew python-tk@3.11 (Tkinter for Python 3.11)..."
brew install python-tk@3.11 || true

# Resolve the brew Python 3.11 path
PY311_BIN="$(brew --prefix python@3.11 2>/dev/null)/bin/python3.11"
if [[ ! -x "$PY311_BIN" ]]; then
  PY311_BIN="$(command -v python3.11 || true)"
fi
if [[ -z "$PY311_BIN" || ! -x "$PY311_BIN" ]]; then
  err "python3.11 not found. Ensure Homebrew python@3.11 is installed."
  exit 1
fi

log "Verifying Tkinter with Homebrew python3.11..."
"$PY311_BIN" - <<'PY'
import sys
try:
    import tkinter as tk
    print("Tkinter OK:", tk.TkVersion)
except Exception as e:
    print("Tkinter import failed:", e, file=sys.stderr)
    sys.exit(1)
PY

# ===== 3) Minimal extra tool for FFmpeg detection =====
if ! command -v pkg-config >/dev/null 2>&1; then
  log "Installing pkg-config (required by FFmpeg to find libopus)..."
  brew install pkg-config
fi

# ===== 4) Paths & staging =====
TARGET_DIR="$HOME/Downloads/transcribe offline installer"
mkdir -p "$TARGET_DIR/content/licences" "$TARGET_DIR/content/vendor" "$TARGET_DIR/content/models"
log "Preparing project at: $TARGET_DIR"
cd "$TARGET_DIR"

# Hard-coded sources (GitHub raw) — British spelling "licences"
MAIN_URL="https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/macOS/main.py"
ICON_URL="https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/macOS/content/AppIcon.icns"
LICENCE_URL="https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/macOS/content/licences/LICENCE.txt"
README_URL="https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/macOS/content/licences/README.md"
THIRD_URL="https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/macOS/content/licences/Third-Party-Licences.txt"

log "Downloading application files from GitHub (British 'licences')..."
curl -fL "$MAIN_URL"    -o "./main.py"
curl -fL "$ICON_URL"    -o "./content/AppIcon.icns"
curl -fL "$LICENCE_URL" -o "./content/licences/LICENCE.txt"
curl -fL "$README_URL"  -o "./content/licences/README.md"
curl -fL "$THIRD_URL"   -o "./content/licences/Third-Party-Licences.txt"

# ===== 5) Venv with Homebrew python3.11 (with Tk) =====
log "Creating virtual environment with Homebrew Python 3.11..."
"$PY311_BIN" -m venv .venv
VENV_PY="$TARGET_DIR/.venv/bin/python"
VENV_PIP="$TARGET_DIR/.venv/bin/pip"
if [[ ! -x "$VENV_PY" ]]; then
  err "Virtualenv creation failed."
  exit 1
fi

log "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip

log "Installing your pinned packages (no deps)..."
"$VENV_PIP" install --no-deps \
  Jinja2==3.1.6 \
  MarkupSafe==3.0.2 \
  PyYAML==6.0.2 \
  asteroid-filterbanks==0.4.0 \
  certifi==2025.8.3 \
  cffi==1.17.1 \
  charset-normalizer==3.4.3 \
  colorlog==6.9.0 \
  cycler==0.12.1 \
  einops==0.8.1 \
  filelock==3.19.1 \
  fsspec==2025.7.0 \
  huggingface-hub==0.34.4 \
  idna==3.10 \
  joblib==1.5.1 \
  julius==0.2.7 \
  kiwisolver==1.4.9 \
  lightning-utilities==0.15.2 \
  llvmlite==0.44.0 \
  matplotlib==3.10.5 \
  mlx==0.28.0 \
  mlx-lm==0.26.3 \
  mlx-metal==0.28.0 \
  mlx-whisper==0.4.2 \
  mpmath==1.3.0 \
  networkx==3.5 \
  nltk==3.9.1 \
  numba==0.61.2 \
  numpy==2.2.6 \
  optuna==4.5.0 \
  packaging==25.0 \
  pandas==2.3.1 \
  pillow==11.3.0 \
  primePy==1.3 \
  psutil==7.0.0 \
  pyannote.audio==3.4.0 \
  pyannote.core==6.0.0 \
  pyannote.database==6.0.0 \
  pyannote.metrics==4.0.0 \
  pyannote.pipeline==4.0.0 \
  pyparsing==3.2.3 \
  python-dateutil==2.9.0.post0 \
  pytorch-lightning==2.5.3 \
  pytorch-metric-learning==2.9.0 \
  pytz==2025.2 \
  regex==2025.7.34 \
  requests==2.32.5 \
  rich==14.1.0 \
  safetensors==0.6.2 \
  scikit-learn==1.7.1 \
  scipy==1.16.1 \
  semver==3.0.4 \
  six==1.17.0 \
  sortedcontainers==2.4.0 \
  sounddevice==0.5.2 \
  soundfile==0.13.1 \
  sympy==1.14.0 \
  tabulate==0.9.0 \
  threadpoolctl==3.6.0 \
  tiktoken==0.11.0 \
  tokenizers==0.21.4 \
  torch==2.8.0 \
  torch-audiomentations==0.12.0 \
  torch_pitch_shift==1.2.5 \
  torchaudio==2.8.0 \
  torchmetrics==1.8.1 \
  tqdm==4.67.1 \
  transformers==4.55.2 \
  typing_extensions==4.14.1 \
  urllib3==2.5.0 \
  whisperx==3.4.2

# ===== 6) Download model repositories into content/models (REQUIRED) =====
log "Setting up REQUIRED model downloads into content/models/..."

"$VENV_PY" - <<'PY'
import os, sys, pathlib, getpass, time
from typing import Optional, List
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

base = pathlib.Path.cwd() / "content" / "models"
base.mkdir(parents=True, exist_ok=True)

# Public models (token not required)
# For wav2vec2-base-960h, filter to .bin weights + necessary JSON/TXT configs (no .h5 or .safetensors).
PUBLIC_TARGETS = [
    # repo_id, subdir, allow_patterns, ignore_patterns
    ("facebook/wav2vec2-base-960h",            "facebook__wav2vec2-base-960h",
     ["*.bin", "*.json", "*.txt"],             ["*.h5", "*.safetensors"]),
    ("mlx-community/whisper-large-v3-turbo",   "mlx-community__whisper-large-v3-turbo",
     None,                                     None),
    ("Qwen/Qwen3-8B-MLX-4bit",                 "Qwen__Qwen3-8B-MLX-4bit",
     None,                                     None),
    ("pyannote/wespeaker-voxceleb-resnet34-LM","pyannote__wespeaker-voxceleb-resnet34-LM",
     None,                                     None),
]

# Protected models (must accept terms + valid token)
PROTECTED_TARGETS = [
    ("pyannote/segmentation-3.0",        "pyannote__segmentation-3.0"),
    ("pyannote/speaker-diarization-3.1", "pyannote__speaker-diarization-3.1"),
]

def info(msg): print(f"\n{msg}")
def warn(msg): print(f"\n[WARN] {msg}")
def err(msg): print(f"\n[ERROR] {msg}", file=sys.stderr)

def already_present(dest: pathlib.Path) -> bool:
    return dest.exists() and any(dest.iterdir())

def fetch_public(repo_id: str, subdir: str, allow: Optional[List[str]], ignore: Optional[List[str]]):
    dest = base / subdir
    if already_present(dest):
        info(f"✔ {repo_id} already present -> {dest}")
        return
    info(f"↓ Downloading public model {repo_id} -> {dest}")
    kwargs = dict(
        repo_id=repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        revision="main",
    )
    if allow: kwargs["allow_patterns"] = allow
    if ignore: kwargs["ignore_patterns"] = ignore
    snapshot_download(**kwargs)
    info(f"✔ Done: {repo_id}")

def have_access_token(tok: Optional[str]) -> bool:
    if not tok:
        return False
    try:
        HfApi().whoami(token=tok)
        return True
    except Exception:
        return False

def prompt_token_until_valid() -> str:
    """
    Keep asking the user for a valid HF token until whoami() succeeds.
    Also show links for account, forms, and token creation.
    """
    links = """
You need a Hugging Face account + an access token for the protected models:
  1) Sign in / create account: https://huggingface.co/
  2) Accept the terms while logged in:
       - https://huggingface.co/pyannote/segmentation-3.0
       - https://huggingface.co/pyannote/speaker-diarization-3.1
  3) Generate a token: https://huggingface.co/settings/tokens
     (setup guide: https://huggingface.co/docs/hub/en/security-tokens)
"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    while True:
        if token and have_access_token(token):
            info("✔ Token looks valid.")
            return token
        if token:
            warn("The provided token appears invalid or expired.")
        print(links)
        try:
            token = getpass.getpass("Enter your Hugging Face token (hidden): ").strip()
        except KeyboardInterrupt:
            err("Aborted by user while entering token.")
            sys.exit(1)
        if not token:
            warn("Empty token. Please paste a valid token.")
            continue

def fetch_protected(repo_id: str, subdir: str):
    """
    Require a valid token AND access approval on the repo page.
    Keep prompting until download succeeds.
    """
    dest = base / subdir
    if already_present(dest):
        info(f"✔ {repo_id} already present -> {dest}")
        return

    token = prompt_token_until_valid()

    while True:
        info(f"↓ Downloading protected model {repo_id} -> {dest}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                token=token,
                revision="main",
            )
            info(f"✔ Done: {repo_id}")
            return
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (401, 403):
                warn(f"Access denied for {repo_id} (HTTP {code}).")
                print(f"""
Make sure you have:
  • Logged into your Hugging Face account in a browser
  • Accepted the model terms on the repo page:
      https://huggingface.co/{repo_id}
  • Generated and pasted a valid access token:
      https://huggingface.co/settings/tokens
    (setup guide: https://huggingface.co/docs/hub/en/security-tokens)
""")
                try:
                    token = getpass.getpass("Re-enter a valid Hugging Face token: ").strip()
                except KeyboardInterrupt:
                    err("Aborted by user while entering token.")
                    sys.exit(1)
                if not token:
                    warn("Empty token. Try again.")
                    continue
            else:
                warn(f"Unexpected error for {repo_id}: {e}")
                print("Retrying in a few seconds...")
                time.sleep(3)
        except KeyboardInterrupt:
            err("Aborted by user during download.")
            sys.exit(1)

# Public models (with filtering for wav2vec2 .bin)
for repo, sub, allow, ignore in PUBLIC_TARGETS:
    fetch_public(repo, sub, allow, ignore)

# Protected models (will keep prompting until success)
for repo, sub in PROTECTED_TARGETS:
    fetch_protected(repo, sub)
PY

# ===== 7) Build FFmpeg 8.0 + static Opus 1.5.2 (LGPL-safe) =====
FFMPEG_VER="8.0"
OPUS_VER="1.5.2"
WORK="$HOME/ffmpeg-lgpl-${FFMPEG_VER}-opus-${OPUS_VER}"
PREFIX_OPUS="${WORK}/opus-local"
DEST="$TARGET_DIR/content/vendor/ffmpeg"

log "Setting up FFmpeg workspace: $WORK"
mkdir -p "$WORK" "$(dirname "$DEST")"
cd "$WORK"

log "Fetching Opus ${OPUS_VER}..."
curl -fL -o "opus-${OPUS_VER}.tar.gz" "https://downloads.xiph.org/releases/opus/opus-${OPUS_VER}.tar.gz"
tar xf "opus-${OPUS_VER}.tar.gz"
cd "opus-${OPUS_VER}"

log "Configuring & building Opus (static)..."
if [[ -x "./configure" ]]; then
  ./configure --disable-shared --enable-static --prefix="$PREFIX_OPUS"
else
  err "Opus ./configure not found. Ensure Xcode Command Line Tools are installed."
  exit 1
fi
make -j"$(sysctl -n hw.ncpu)"
make install

cd "$WORK"
log "Fetching FFmpeg ${FFMPEG_VER}..."
curl -fL -o "ffmpeg-${FFMPEG_VER}.tar.xz" "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VER}.tar.xz"
tar xf "ffmpeg-${FFMPEG_VER}.tar.xz"
cd "ffmpeg-${FFMPEG_VER}"

log "Configuring FFmpeg (LGPL-safe)..."
export PKG_CONFIG_PATH="${PREFIX_OPUS}/lib/pkgconfig"
./configure \
  --arch=arm64 \
  --cc=clang \
  --disable-debug --disable-doc \
  --disable-ffplay --disable-ffprobe \
  --enable-static --disable-shared \
  --enable-pthreads \
  --pkg-config-flags="--static" \
  --extra-cflags="-I${PREFIX_OPUS}/include" \
  --extra-ldflags="-L${PREFIX_OPUS}/lib" \
  --enable-libopus

log "Building FFmpeg..."
make -j"$(sysctl -n hw.ncpu)"

log "Installing FFmpeg binary into project..."
cp -f "./ffmpeg" "$DEST"
chmod +x "$DEST"

# ===== 8) Build .app only, move to Applications, verify launch, clean =====
log "Installing PyInstaller into the venv..."
"$VENV_PIP" install pyinstaller

APP_NAME="Transcribe Offline"
ENTRY="$TARGET_DIR/main.py"
BUNDLE_ID="com.local.transcribeoffline"

cd "$TARGET_DIR"
rm -rf build dist "${APP_NAME}.spec"

log "Building .app with PyInstaller..."
"$TARGET_DIR/.venv/bin/pyinstaller" \
  --name "$APP_NAME" \
  --windowed \
  --noconfirm \
  --osx-bundle-identifier "$BUNDLE_ID" \
  --target-arch arm64 \
  --add-data "content:content" \
  --icon "content/AppIcon.icns" \
  --collect-all sounddevice \
  --collect-all soundfile \
  --collect-all cffi \
  --collect-all psutil \
  --collect-all mlx \
  --collect-all mlx_lm \
  --collect-all mlx_whisper \
  --collect-all whisperx \
  --collect-all pyannote.audio \
  --collect-all pyannote.core \
  --collect-all torch \
  --collect-all torchaudio \
  --collect-all tokenizers \
  --collect-all safetensors \
  --hidden-import mlx_lm.sample_utils \
  "$ENTRY"

# Remove any extra folder build; keep only the .app
if [[ -d "dist/${APP_NAME}" ]]; then
  rm -rf "dist/${APP_NAME}"
fi

APP_PATH="dist/${APP_NAME}.app"
if [[ ! -d "$APP_PATH" ]]; then
  err "Build failed: ${APP_PATH} not found."
  exit 1
fi

log "Moving ${APP_NAME}.app to /Applications..."
DEST_APPS="/Applications"
if [[ -d "$HOME/Applications" && -w "$HOME/Applications" ]]; then
  DEST_APPS="$HOME/Applications"
fi
ditto "$APP_PATH" "$DEST_APPS/${APP_NAME}.app"
APP_DEST="$DEST_APPS/${APP_NAME}.app"
rm -rf "$APP_PATH"

# ---- Try launching app; if it fails, clear quarantine and retry ----
is_running() {
  if osascript -e 'tell application "System Events" to (exists process "Transcribe Offline")' 2>/dev/null | grep -qi true; then
    return 0
  fi
  pgrep -ix "Transcribe Offline" >/dev/null 2>&1
}

try_open() {
  open -n "$APP_DEST" >/dev/null 2>&1 || return 1
  sleep 8
  is_running
}

log "Verifying app launch..."
if try_open; then
  log "✅ App launched successfully."
else
  log "App didn't launch. Checking for quarantine attribute and removing it..."
  if xattr -p com.apple.quarantine "$APP_DEST" >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$APP_DEST" || true
    log "Removed quarantine attribute. Retrying launch..."
  else
    log "No quarantine attribute found. Retrying launch anyway..."
  fi

  if try_open; then
    log "✅ App launched successfully after removing quarantine."
  else
    err "App still didn't launch.
If macOS blocked it, try:
  • System Settings → Privacy & Security → 'Open Anyway' for 'Transcribe Offline'
  • Or launch once from Finder: right-click the app → Open → Open
After first launch, it should open normally."
  fi
fi

log "Removing venv and installer folder..."
rm -rf "$TARGET_DIR/.venv"
cd "$HOME"
rm -rf "$TARGET_DIR"

log "Done. App installed at: $APP_DEST"
