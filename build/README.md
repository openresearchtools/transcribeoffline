## Build

From this repository root:

Runtime behavior:
- playback/transcription decode uses FFmpeg shared libraries from ENGINE runtime location
  (`<runtime_dir>/vendor/ffmpeg/*`) via in-app runtime PATH setup.
- FFmpeg shared libraries are not bundled in this repo; they are resolved from the external runtime directory
  (`%APPDATA%\\OpenResearchTools\\engine` on Windows, the app runtime on macOS, or the selected
  APT-managed runtime under `/opt/openresearchtools/engine/` on Linux).

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

### macOS bundle

Create a distribution folder on macOS arm64:

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
- Windows and macOS runtime behavior remains manifest-driven.

### Linux Debian package

Build the Linux x64 application and an installable Debian package:

```bash
./build/package-linux-deb.sh
```

By default, Cargo outputs and the `.deb` are written outside this repository under
`../TRANSCRIBEbuilds/linux/`. Use `--target-dir` and `--output-dir` to override them.

The package:
- is named `transcribe-offline` and installs the `transcribe-offline` launcher;
- depends on both `openresearchtools-engine` and `openresearchtools-engine-cuda`;
- does not bundle, download, or modify either engine runtime.

On Linux, the runtime selector maps literally to:
- Vulkan: `/opt/openresearchtools/engine/vulkan`
- CUDA: `/opt/openresearchtools/engine/cuda`

## Runtime bootstrap

Transcribe Offline runs runtime checks on startup and in Settings > Runtime:
- runtime health,
- minimum model presence (Whisper + diarization).

If missing, it opens setup for:
- Openresearchtools-Engine runtime (Windows/macOS install/repair; Linux APT package check),
- Whisper model,
- diarization model pack,
- optional chat model.

Windows/macOS manifest lookup order:
- `./runtime-manifests/engine-manifest.json`
- user-data cache `.../runtime-manifests/engine-manifest.json`
- remote URLs from `./runtime-manifests/engine-manifest-sources.json`

Default remote source:
- `https://github.com/openresearchtools/engine/releases/latest/download/engine-manifest.json`
