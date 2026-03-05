# Transcribe Offline (Rust + egui)

![Transcribe Offline Demo](Demo.png)

**Transcribe Offline** is an open-source desktop app for local transcription,
speaker diarization, and transcript review/edit workflows.
It is implemented as a native Rust/egui GUI on top of
[`Openresearchtools-Engine`](https://github.com/openresearchtools/engine).

It focuses on:
- transcription workflows (`speech` / `subtitle` / `transcript`),
- speaker diarization,
- local transcript chat and anonymisation,
- and desktop-first editing/playback flows.

## What it does

### 1) Local transcription modes
Runs the engine audio path for:
- `speech` output,
- `subtitle` output,
- `transcript` output with speaker diarization.

### 2) Speaker diarization
Uses the engine's native C++ pyannote-style diarization integration and model packs. 

### 3) Transcript review and editing
Provides side-by-side transcript/edit views, playback-linked navigation, autosave,
speaker rename tools, and anonymisation pass tooling.

### 4) Local LLM helpers
Uses the engine bridge chat path with local GGUF models for transcript Q&A and
anonymisation extraction.

> Designed to be local‑first. However, no software can guarantee absolute privacy or security. Please consider your threat model and institutional policies before processing sensitive material.

## Unsigned Build Notice

This app is an open-source hobby development effort by the repository owner.
We do not currently have funding for full paid code-signing and notarization
pipelines across all platforms/releases.

Because of that, operating-system protections or hardened security environments
(for example Windows SmartScreen, enterprise endpoint controls, or macOS
Gatekeeper policies) may block unsigned binaries.

If your environment blocks unsigned binaries, the recommended path is:
- build this desktop app from source on the target device,
- build Openresearchtools-Engine from source on the same target device,
- and use those locally-built artifacts in your deployment.

### Windows (when blocked)

- If SmartScreen shows "Windows protected your PC", use `More info` ->
  `Run anyway` only if your policy allows it.
- In the app, go to `Settings -> Runtime Setup` and run:
  - `Download/Repair runtime`
  - `Unblock unsigned runtime`
  - `Recheck`
- The Windows unblock script clears Mark-of-the-Web flags in the selected
  runtime directory by running `Unblock-File` recursively on runtime files.

### macOS (when blocked)

- Try `Right click -> Open` on first launch.
- If blocked by Gatekeeper, use `System Settings -> Privacy & Security ->
  Open Anyway` when available and policy permits.
- In the app, after runtime install/repair, click `Unblock unsigned runtime`
  then `Recheck`.
- The macOS unblock script removes quarantine attributes recursively
  (`xattr -dr com.apple.quarantine`) and restores executable bits for runtime
  binaries/scripts where needed (`chmod +x` on relevant files).

---

## Highlights

- Offline-first runtime flow with in-app runtime install/repair.
- Native desktop orchestration of Openresearchtools-Engine (`llama-server-bridge`).
- Single device selection model (CPU or selected GPU) for runtime execution.
- Built-in transcript editing, playback follow, anonymisation, and export workflow.


---

## How it works (in this repo)

- This app is a GUI/orchestration layer.
- Openresearchtools-Engine provides the local runtime components.
- The app invokes runtime features through `llama-server-bridge`.
- Playback decode in the app uses the Rust `Symphonia` stack; runtime-side media conversion uses engine FFmpeg components.

## Project relationship

- `Transcribe Offline` is a reference example of integrating Openresearchtools-Engine in a native desktop GUI.
- This app uses `llama-server-bridge` from Openresearchtools-Engine.
- Openresearchtools-Engine runs on a modified `llama.cpp` runtime path to support a native C++ pyannote-style diarization pipeline.
- This desktop app itself is a wrapper/orchestrator around that runtime.
- This desktop app relies on Openresearchtools-Engine runtime media components (FFmpeg/PDFium) at runtime.

## Acknowledgements (used in this app/runtime path)

- [`Openresearchtools-Engine`](https://github.com/openresearchtools/engine):
  embeddable runtime used by this app (`llama-server-bridge`, runtime orchestration, and model/device execution path).
- [`egui`](https://github.com/emilk/egui) / [`eframe`](https://github.com/emilk/egui/tree/master/crates/eframe):
  native immediate-mode GUI framework used to build this desktop application UI.
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) and [`ggml`](https://github.com/ggml-org/ggml):
  core inference runtime and device/offload mechanics used through Openresearchtools-Engine.
- [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp):
  transcription backbone used by the engine audio pipeline.
- [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) and [`WeSpeaker`](https://github.com/wenet-e2e/wespeaker):
  diarization lineage and model/pipeline provenance used by the engine's native C++ diarization path.
- [`FFmpeg`](https://github.com/FFmpeg/FFmpeg) (LGPL shared runtime builds):
  media decoding/normalization path used by Openresearchtools-Engine runtime for audio conversion.
- [`Symphonia`](https://github.com/pdeljanov/Symphonia):
  Rust audio decoding stack used by this desktop app for local playback.

## Non-endorsement statement

This project is independent work by OpenResearchTools.
This project is **not affiliated with, sponsored by, or endorsed by** the maintainers/owners of any third-party projects listed above or in the bundled notices/license files.

All third-party names and marks remain property of their respective owners.

## Runtime bootstrap

`Transcribe Offline.exe` runs runtime checks on startup and in Settings > Runtime:
- runtime health (bridge + FFmpeg + PDFium),
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

## Licensing and notices

The Transcribe Offline application source code is licensed under the MIT License; third-party dependencies and bundled components remain licensed under their respective original licenses.

Read these files in this repo:

- License index: `licenses/README.md`
- Consolidated notice page (app + models + engine): `licenses/THIRD_PARTY_NOTICES_ALL.md`
- Full app third-party licenses (full text per package): `licenses/APP_THIRD_PARTY_LICENSES_FULL.md`
- Full engine third-party licenses (full text per package/file): `licenses/ENGINE_THIRD_PARTY_LICENSES_FULL.md`
- `Help -> Notices`
- `Help -> App licenses`
- `Help -> Engine licenses`


## Models and conversions

- Whisper model binaries are fetched from `ggerganov/whisper.cpp` model assets from Huggingface repo.
- Diarization model pack is fetched from
  `openresearchtools/speaker-diarization-community-1-GGUF`.
- OpenResearchTools publishes converted model artifacts for runtime compatibility.

Converted-model note:
- These converted artifacts are provided for interoperability with this runtime.
- They are not upstream-official releases and are not endorsed by upstream model owners.
- Use follows the original model licenses and model-card terms.

For citations and model lineage references, see the bundled Notices document:
- `licenses/THIRD_PARTY_NOTICES_ALL.md`

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

## How to cite

Suggested citation:

Rutkauskas, L. (2026). *Transcribe Offline* (Version 2.0.0) [Computer software].
OpenResearchTools. <https://github.com/openresearchtools/transcribeoffline>.

BibTeX:

```bibtex
@software{Rutkauskas_TranscribeOffline_2026,
  author    = {Rutkauskas, L.},
  title     = {Transcribe Offline},
  version   = {2.0.0},
  date      = {2026-03-04},
  url       = {https://github.com/openresearchtools/transcribeoffline},
  publisher = {OpenResearchTools},
  license   = {MIT}
}
```
