#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle_dir=""
target_triple=""
locked=1
os_name="$(uname -s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      bundle_dir="${2:-}"
      shift 2
      ;;
    --target)
      target_triple="${2:-}"
      shift 2
      ;;
    --locked)
      locked=1
      shift
      ;;
    --no-locked)
      locked=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$bundle_dir" ]]; then
  case "$os_name" in
    Darwin)
      bundle_dir="$repo_root/artifacts/bundles/transcribe-offline-macos-arm64-bundle"
      ;;
    Linux)
      bundle_dir="$repo_root/artifacts/bundles/transcribe-offline-ubuntu-x64-bundle"
      ;;
    *)
      echo "Unsupported OS for package-unix.sh" >&2
      exit 1
      ;;
  esac
fi

app_version="$(awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$repo_root/Cargo.toml")"
if [[ -z "$app_version" ]]; then
  app_version="0.0.0"
fi

copy_common_bundle_files() {
  local target_dir="$1"

  # Runtime is intentionally not bundled; app downloads/repairs runtime in-app.
  # Copy runtime manifest lookup files used by in-app runtime checks.
  if [[ -d "$repo_root/runtime-manifests" ]]; then
    cp -R "$repo_root/runtime-manifests" "$target_dir/runtime-manifests"
  fi

  # Copy app-side license/notices docs shipped with this wrapper.
  if [[ -d "$repo_root/licenses" ]]; then
    cp -R "$repo_root/licenses" "$target_dir/licenses"
  fi
  if [[ -f "$repo_root/LICENSE" ]]; then
    cp "$repo_root/LICENSE" "$target_dir/LICENSE"
  fi
}

echo "BundleDir: $bundle_dir"
rm -rf "$bundle_dir"
mkdir -p "$bundle_dir"

export CARGO_TARGET_DIR="$repo_root/artifacts/target"
build_args=(build --release --bin transcribe-offline)
if [[ "$locked" -eq 1 ]]; then
  build_args+=(--locked)
fi
if [[ -n "$target_triple" ]]; then
  build_args+=(--target "$target_triple")
fi

(
  cd "$repo_root"
  cargo "${build_args[@]}"
)

if [[ -n "$target_triple" ]]; then
  target_release="$CARGO_TARGET_DIR/$target_triple/release"
else
  target_release="$CARGO_TARGET_DIR/release"
fi

main_bin="$target_release/transcribe-offline"
if [[ ! -f "$main_bin" ]]; then
  echo "Missing built app binary: $main_bin" >&2
  exit 1
fi

if [[ "$os_name" == "Darwin" ]]; then
  app_name="Transcribe Offline"
  app_dir="$bundle_dir/${app_name}.app"
  contents_dir="$app_dir/Contents"
  macos_dir="$contents_dir/MacOS"
  resources_dir="$contents_dir/Resources"

  mkdir -p "$macos_dir" "$resources_dir"
  cp "$main_bin" "$macos_dir/transcribe-offline"
  chmod +x "$macos_dir/transcribe-offline" || true

  if [[ -f "$repo_root/assets/icons/AppIcon.icns" ]]; then
    cp "$repo_root/assets/icons/AppIcon.icns" "$resources_dir/AppIcon.icns"
  fi

  cat > "$contents_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>${app_name}</string>
  <key>CFBundleDisplayName</key>
  <string>${app_name}</string>
  <key>CFBundleIdentifier</key>
  <string>com.openresearchtools.transcribeoffline</string>
  <key>CFBundleVersion</key>
  <string>${app_version}</string>
  <key>CFBundleShortVersionString</key>
  <string>${app_version}</string>
  <key>CFBundleExecutable</key>
  <string>transcribe-offline</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon.icns</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Transcribe Offline needs microphone access for live transcription and diarization.</string>
</dict>
</plist>
EOF

  copy_common_bundle_files "$macos_dir"

  # Ad-hoc sign when possible to improve local execution ergonomics.
  if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "$app_dir" >/dev/null 2>&1 || true
  fi

  dmg_name="transcribe-offline-macos-arm64.dmg"
  dmg_path="$bundle_dir/$dmg_name"
  dmg_staging="$bundle_dir/.dmg-staging"
  rm -rf "$dmg_staging" "$dmg_path"
  mkdir -p "$dmg_staging"
  cp -R "$app_dir" "$dmg_staging/"
  ln -s /Applications "$dmg_staging/Applications"

  if command -v hdiutil >/dev/null 2>&1; then
    hdiutil create -volname "$app_name" -srcfolder "$dmg_staging" -ov -format UDZO "$dmg_path" >/dev/null
    echo "DMG ready: $dmg_path"
  else
    echo "Warning: hdiutil not found; skipping DMG creation." >&2
  fi

  rm -rf "$dmg_staging"
else
  cp "$main_bin" "$bundle_dir/transcribe-offline"
  chmod +x "$bundle_dir/transcribe-offline" || true
  copy_common_bundle_files "$bundle_dir"
fi

# Ubuntu convenience installer: sets up app menu entry for downloaded bundle.
if [[ "$os_name" == "Linux" ]]; then
  if [[ -f "$repo_root/assets/icons/AppIcon.png" ]]; then
    cp "$repo_root/assets/icons/AppIcon.png" "$bundle_dir/transcribe-offline.png"
  fi

  cat > "$bundle_dir/install-ubuntu.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install_dir="$HOME/.local/share/transcribe-offline"
bin_dir="$HOME/.local/bin"
desktop_dir="$HOME/.local/share/applications"
icon_dir="$HOME/.local/share/icons/hicolor/256x256/apps"

echo "Installing Transcribe Offline into user profile..."
rm -rf "$install_dir"
mkdir -p "$install_dir" "$bin_dir" "$desktop_dir" "$icon_dir"

cp -a "$script_dir/." "$install_dir/"
rm -f "$install_dir/install-ubuntu.sh" "$install_dir/uninstall-ubuntu.sh"
chmod +x "$install_dir/transcribe-offline" || true

cat > "$bin_dir/transcribe-offline" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
exec "$HOME/.local/share/transcribe-offline/transcribe-offline" "$@"
WRAP
chmod +x "$bin_dir/transcribe-offline"

if [[ -f "$install_dir/transcribe-offline.png" ]]; then
  cp "$install_dir/transcribe-offline.png" "$icon_dir/transcribe-offline.png"
fi

cat > "$desktop_dir/transcribe-offline.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Transcribe Offline
Comment=Offline transcription and diarization GUI powered by Openresearchtools-Engine
Exec=$bin_dir/transcribe-offline
Icon=$icon_dir/transcribe-offline.png
StartupWMClass=transcribe-offline
Terminal=false
Categories=AudioVideo;Utility;
StartupNotify=true
DESKTOP

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$desktop_dir" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
fi

echo "Install complete."
echo "App menu entry: Transcribe Offline"
echo "CLI launcher: $bin_dir/transcribe-offline"
EOF

  cat > "$bundle_dir/uninstall-ubuntu.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

rm -rf "$HOME/.local/share/transcribe-offline"
rm -f "$HOME/.local/bin/transcribe-offline"
rm -f "$HOME/.local/share/applications/transcribe-offline.desktop"
rm -f "$HOME/.local/share/icons/hicolor/256x256/apps/transcribe-offline.png"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$HOME/.local/share/applications" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true
fi

echo "Transcribe Offline removed from user profile."
EOF

  chmod +x "$bundle_dir/install-ubuntu.sh" "$bundle_dir/uninstall-ubuntu.sh" || true
fi

echo "Bundle ready: $bundle_dir"
