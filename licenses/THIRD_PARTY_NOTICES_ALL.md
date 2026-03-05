# Third-Party Notices

## App notices

Transcribe Offline is distributed under the MIT License.
The Transcribe Offline application source code is licensed under MIT, while third-party dependencies and bundled components remain licensed under their respective original licenses.
Full third-party license texts are available in the in-app Third-party licenses view.
Model notices and citations are listed below.

### GUI framework (egui / eframe)

- The desktop GUI is built with the Rust `egui` / `eframe` crates.
- License: MIT OR Apache-2.0.
- Integration mode: upstream crates linked through Cargo dependencies; no local fork of egui/eframe in this app repository.
- Full license text is included in Third-party licenses (crate entries for `egui` / `eframe` and related crates).

### Audio playback decoder (Symphonia)

- Playback decoding uses the Rust `symphonia` crate family.
- License: MPL-2.0.
- Integration mode: linked as upstream crates via Cargo dependencies; no local fork/modification in this app repository.
- Full license text is included in Third-party licenses (Symphonia crate entries).

---

## Model notices

# Model Notices and Citations

Citations below are upstream original model/research citations.
Non-endorsement: inclusion, conversion, or runtime use in Transcribe Offline does not imply endorsement, sponsorship, affiliation, or approval by original authors/maintainers/institutions.
Runtime/framework package licenses are listed in Engine licenses.
Converted/interoperability artifacts in this app are not official upstream model releases.

## Whisper models

- Download source used by app (converted artifacts): https://huggingface.co/ggerganov/whisper.cpp
- Original upstream model card reference: https://huggingface.co/openai/whisper-large-v3
- Model card license reference at time of writing: apache-2.0.
- Runtime artifact format: GGML.
- Conversion notice: distributed artifacts are conversion/interoperability outputs and may differ from original upstream packaging.
- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.
- Users are responsible for complying with upstream model terms.

## Diarization models (pyannote lineage)

- Download source used by app (converted artifacts): https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF
- Original upstream model card reference: https://huggingface.co/pyannote/speaker-diarization-community-1
- Model card license reference at time of writing: cc-by-4.0.
- Runtime artifact format: GGUF.
- Conversion notice: distributed artifacts are conversion/interoperability outputs.
- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.

## WeSpeaker embedding model

- Used in diarization pipeline lineage and cited here as an original upstream model component.
- Primary citation is listed below (Wang et al., ICASSP 2023).
- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.

## Upstream original citations

Upstream original speaker segmentation citation:
@inproceedings{Plaquet23, title={Powerset multi-class cross entropy loss for neural speaker diarization}, year=2023, booktitle={Proc. INTERSPEECH 2023}}

Upstream original speaker embedding citation:
@inproceedings{Wang2023, title={Wespeaker: A research and production oriented speaker embedding learning toolkit}, year=2023, booktitle={ICASSP 2023}}

Upstream original speaker clustering citation:
@article{Landini2022, title={Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization}, year=2022, journal={Computer Speech & Language}}

---

## Engine notices

# Engine Notices

These notices describe bundled runtime provenance/build facts for Openresearchtools-Engine.
Full legal texts (FFmpeg LGPL, PDFium licenses, NVIDIA CUDA EULA/runtime notice, whisper.cpp, pyannote.audio, and others) are in Engine licenses.

## FFmpeg runtime (LGPL shared)

- Purpose: raw-audio normalization for transcription requests (convert to WAV 16-bit mono 16 kHz before endpoint call).
- Windows x64 retrieval source: https://github.com/BtbN/FFmpeg-Builds
- Windows workflow reference: https://github.com/openresearchtools/engine/blob/main/.github/workflows/windows-x64.yml
- Windows asset pattern: *win64-lgpl-shared*.zip
- Ubuntu x64 retrieval source: https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest
- Ubuntu asset name: ffmpeg-master-latest-linux64-lgpl-shared.tar.xz
- Ubuntu workflow reference: https://github.com/openresearchtools/engine/blob/main/.github/workflows/ubuntu-x64.yml
- macOS arm64 source-build reference: https://github.com/openresearchtools/engine/blob/main/.github/workflows/macos-arm64.yml
- macOS pinned FFmpeg tag/commit (upstream notice): n8.0.1 / 894da5ca7d742e4429ffb2af534fcda0103ef593
- macOS LGPL configure flags (upstream workflow): --enable-shared --disable-static --disable-gpl --disable-version3 --disable-nonfree --disable-autodetect --disable-xlib --disable-libxcb --disable-libxcb-shm --disable-libxcb-xfixes --disable-libxcb-shape --disable-vulkan --disable-libplacebo --enable-pic --disable-programs --disable-doc --cc=clang --arch=arm64 --target-os=darwin
- FFmpeg license texts are in Engine licenses.

## PDFium runtime

- Purpose: PDF rasterization for pdf/pdfvlm modules.
- Engine runtime location: third_party/pdfium
- App runtime lookup by OS: vendor/pdfium/pdfium.dll (Windows), vendor/pdfium/libpdfium.dylib (macOS), vendor/pdfium/libpdfium.so (Linux).
- Binary source used by engine: https://github.com/bblanchon/pdfium-binaries
- Upstream notice reference: https://github.com/openresearchtools/engine/blob/main/third_party/licenses/README.md
- PDFium license texts are in Engine licenses.

## CUDA runtime (Windows optional)

- Windows users may choose a CUDA runtime build or a Vulkan runtime build.
- NVIDIA CUDA terms apply only when a Windows CUDA runtime build bundles/uses NVIDIA CUDA runtime binaries (for example, cudart/cublas DLLs).
- Typical CUDA DLLs in CUDA builds: cublas64_13.dll, cublasLt64_13.dll, cudart64_13.dll.
- Official NVIDIA CUDA EULA page: https://docs.nvidia.com/cuda/eula/index.html
- NVIDIA CUDA EULA and runtime notice full texts are in Engine licenses.

## pyannote.audio lineage notice (engine runtime)

- Engine diarization path uses a native C++ pipeline derived from pyannote.audio structure/metadata semantics.
- Runtime endpoint path does not invoke Python.
- Non-endorsement: this C++ reimplementation and its use in Transcribe Offline are not endorsed by original pyannote authors/maintainers.
- pyannote.audio license text is in Engine licenses.
