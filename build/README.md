## Build

From this repository root:

Runtime behavior:
- playback/transcription decode uses FFmpeg shared libraries from ENGINE runtime location
  (`<runtime_dir>/vendor/ffmpeg/*`) via in-app runtime PATH setup.
- FFmpeg shared libraries are not bundled in this repo; they are resolved from the external runtime directory
  (`%APPDATA%\\OpenResearchTools\\engine` by default, or configured `runtime_dir`).

```bash
cargo check --locked
cargo run --locked
```

Release (core binaries only):

```bash
cargo build --release --locked
```

Single app executable:
- `artifacts/target/release/transcribe-offline.exe` -> `Transcribe Offline.exe`

Build outputs are standardized under `artifacts/`:
- Cargo build outputs: `artifacts/target/*`
- Packaged app bundles: `artifacts/bundles/*`

### Windows bundle

Create a distribution folder with the app executable:

```powershell
.\build\package-win-x64.ps1
```

Optional flags:
- `-BundleDir "<path>"` to control output location.
- `-TargetTriple "<target>"` to package a specific Rust target output.

### macOS/Linux bundle

Create a distribution folder on macOS arm64 or Ubuntu x64:

```bash
./build/package-unix.sh --locked
```

Optional flags:
- `--bundle-dir "<path>"` to control output location.
- `--target "<triple>"` to package a specific Rust target output.

macOS bundle outputs:
- `Transcribe Offline.app` (native app bundle)
- `transcribe-offline-macos-arm64.dmg` (drag-to-Applications installer image)

Runtime note:
- build bundles do **not** include engine runtime binaries.
- runtime is installed/repaired in-app using `runtime-manifests/engine-manifest.json` per platform.

Ubuntu app-menu install from downloaded bundle:

```bash
cd artifacts/bundles/transcribe-offline-ubuntu-x64
bash install-ubuntu.sh
```

This creates:
- launcher: `~/.local/bin/transcribe-offline`
- desktop entry: `~/.local/share/applications/transcribe-offline.desktop`
- icon: `~/.local/share/icons/hicolor/256x256/apps/transcribe-offline.png`

## Runtime bootstrap

`Transcribe Offline.exe` runs runtime checks on startup and in Settings > Runtime:
- runtime health,
- minimum model presence (Whisper + diarization).

If missing, it opens setup with install/repair options for:
- Openresearchtools-Engine runtime,
- Whisper model,
- diarization model pack,
- optional chat model.

Manifest lookup order:
- `./runtime-manifests/engine-manifest.json`
- user-data cache `.../runtime-manifests/engine-manifest.json`
- remote URLs from `./runtime-manifests/engine-manifest-sources.json`

Default remote source:
- `https://github.com/openresearchtools/engine/releases/latest/download/engine-manifest.json`
