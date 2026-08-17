#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_root}/../TRANSCRIBEbuilds/linux/deb"
target_dir="${repo_root}/../TRANSCRIBEbuilds/linux/cargo-target"
binary_path=""
package_version=""

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

mkdir -p "$output_dir" "$target_dir"
if [[ -z "$binary_path" ]]; then
  CARGO_TARGET_DIR="$target_dir" cargo build \
    --manifest-path "$repo_root/Cargo.toml" \
    --release \
    --locked \
    --target x86_64-unknown-linux-gnu \
    --bin transcribe-offline
  binary_path="$target_dir/x86_64-unknown-linux-gnu/release/transcribe-offline"
fi
if [[ ! -x "$binary_path" ]]; then
  echo "Missing executable app binary: $binary_path" >&2
  exit 1
fi

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
Architecture: amd64
Maintainer: OpenResearchTools <openresearchtools@users.noreply.github.com>
Installed-Size: ${installed_size}
Depends: openresearchtools-engine, openresearchtools-engine-cuda, ${native_depends}, alsa-utils, libudev1, libx11-6, libxcb1, libxkbcommon0, libwayland-client0, xdg-utils
Description: Local transcription, diarization, chat, and transcript review
 Native desktop application using the APT-installed Openresearchtools-Engine
 Vulkan or CUDA runtime selected in the application settings.
CONTROL

find "$package_root" -type d -exec chmod 0755 {} +
find "$doc_root" -type f -exec chmod 0644 {} +
chmod 0644 \
  "$package_root/DEBIAN/control" \
  "$package_root/usr/share/applications/transcribe-offline.desktop" \
  "$package_root/usr/share/icons/hicolor/256x256/apps/transcribe-offline.png"

asset_path="$output_dir/transcribe-offline_${package_version}_amd64.deb"
dpkg-deb --build --root-owner-group "$package_root" "$asset_path"
echo "Debian package ready: $asset_path"
