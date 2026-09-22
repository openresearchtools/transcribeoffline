#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/../TRANSCRIBEbuilds/linux/deb"
target_dir="${repo_root}/../TRANSCRIBEbuilds/linux/cargo-target"
binary_path=""
package_version=""
architecture="amd64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --target-dir)
      target_dir="${2:-}"
      shift 2
      ;;
    --binary)
      binary_path="${2:-}"
      shift 2
      ;;
    --architecture)
      architecture="${2:-}"
      shift 2
      ;;
    --version)
      package_version="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "package-linux-deb.sh must run on Linux." >&2
  exit 1
fi
if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required (install the dpkg package)." >&2
  exit 1
fi

if [[ -z "$package_version" ]]; then
  package_version="$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$repo_root/Cargo.toml")"
fi
if [[ -z "$package_version" ]]; then
  echo "Unable to determine package version." >&2
  exit 1
fi

case "$architecture" in
  amd64) rust_target="x86_64-unknown-linux-gnu"; engine_depends="openresearchtools-engine, openresearchtools-engine-cuda"; elf_machine="62" ;;
  arm64) rust_target="aarch64-unknown-linux-gnu"; engine_depends="openresearchtools-engine (>= 1.17)"; elf_machine="183" ;;
  *) echo "Unsupported architecture: $architecture" >&2; exit 2 ;;
esac
mkdir -p "$output_dir" "$target_dir"
output_dir="$(realpath "$output_dir")"
target_dir="$(realpath "$target_dir")"
if [[ -z "$binary_path" ]]; then
  CARGO_TARGET_DIR="$target_dir" cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --release \
    --locked \
    --target "$rust_target" \
    --bin transcribe-offline
  binary_path="$target_dir/$rust_target/release/transcribe-offline"
fi
if [[ ! -x "$binary_path" ]]; then
  echo "Missing executable app binary: $binary_path" >&2
  exit 1
fi

python3 - "$binary_path" "$elf_machine" <<'CHECK_ELF'
import sys
from pathlib import Path
header = Path(sys.argv[1]).read_bytes()[:20]
assert header[:5] == b"\x7fELF\x02" and int.from_bytes(header[18:20], "little") == int(sys.argv[2]), "binary architecture does not match package"
CHECK_ELF

work_root="$(mktemp -d "$output_dir/.transcribe-offline-deb.XXXXXX")"
trap 'rm -rf "$work_root"' EXIT
package_root="$work_root/root"
install_root="$package_root/usr/lib/transcribe-offline"
doc_root="$package_root/usr/share/doc/transcribe-offline"

mkdir -p \
  "$package_root/DEBIAN" \
  "$install_root" \
  "$package_root/usr/bin" \
  "$package_root/usr/share/applications" \
  "$package_root/usr/share/icons/hicolor/256x256/apps" \
  "$doc_root/licenses"

install -m 0755 "$binary_path" "$install_root/transcribe-offline"
ln -s ../lib/transcribe-offline/transcribe-offline \
  "$package_root/usr/bin/transcribe-offline"
install -m 0644 "$repo_root/assets/icons/AppIcon.png" \
  "$package_root/usr/share/icons/hicolor/256x256/apps/transcribe-offline.png"
install -m 0644 "$repo_root/LICENSE" "$doc_root/copyright"
cp -a "$repo_root/licenses/." "$doc_root/licenses/"

cat > "$package_root/usr/share/applications/transcribe-offline.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=Transcribe Offline
Comment=Offline transcription and diarization powered by Openresearchtools-Engine
Exec=transcribe-offline
Icon=transcribe-offline
StartupWMClass=transcribe-offline
Terminal=false
Categories=AudioVideo;Audio;
StartupNotify=true
DESKTOP

installed_size="$(du -sk "$package_root/usr" | awk '{print $1}')"
native_depends="libasound2, libc6, libgcc-s1"
if command -v dpkg-shlibdeps >/dev/null 2>&1; then
  mkdir -p "$work_root/debian"
  cat > "$work_root/debian/control" <<'SHLIBS_CONTROL'
Source: transcribe-offline
Section: sound
Priority: optional
Maintainer: OpenResearchTools <openresearchtools@users.noreply.github.com>

Package: transcribe-offline
Architecture: any
Description: Local transcription desktop application
SHLIBS_CONTROL
  shlibs_line="$(
    cd "$work_root"
    dpkg-shlibdeps -O "$install_root/transcribe-offline"
  )"
  native_depends="${shlibs_line#shlibs:Depends=}"
fi

cat > "$package_root/DEBIAN/control" <<CONTROL
Package: transcribe-offline
Version: ${package_version}
Section: sound
Priority: optional
Architecture: ${architecture}
Maintainer: OpenResearchTools <openresearchtools@users.noreply.github.com>
Installed-Size: ${installed_size}
Depends: ${engine_depends}, ${native_depends}, alsa-utils, libudev1, libx11-6, libxcb1, libxkbcommon0, libwayland-client0, xdg-utils
Description: Local transcription, diarization, chat, and transcript review
 Native desktop application using the APT-installed Openresearchtools-Engine
 runtime, with automatic Vulkan selection on Linux ARM64.
CONTROL

find "$package_root" -type d -exec chmod 0755 {} +
find "$doc_root" -type f -exec chmod 0644 {} +
chmod 0644 \
  "$package_root/DEBIAN/control" \
  "$package_root/usr/share/applications/transcribe-offline.desktop" \
  "$package_root/usr/share/icons/hicolor/256x256/apps/transcribe-offline.png"

asset_path="$output_dir/transcribe-offline_${package_version}_${architecture}.deb"
dpkg-deb --build --root-owner-group "$package_root" "$asset_path"
echo "Debian package ready: $asset_path"
