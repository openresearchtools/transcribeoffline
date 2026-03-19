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

- Download sources used by app (converted artifacts):
  - https://huggingface.co/openresearchtools/whisper-large-v3-turbo-GGML
  - https://huggingface.co/openresearchtools/whisper-large-v3-GGML
- Original upstream model card references:
  - https://huggingface.co/openai/whisper-large-v3-turbo
  - https://huggingface.co/openai/whisper-large-v3
- Model card license reference at time of writing: apache-2.0.
- Runtime artifact format: GGML.
- Conversion notice: distributed artifacts are conversion/interoperability outputs and may differ from original upstream packaging.
- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.
- Users are responsible for complying with upstream model terms.

## Qwen 3.5 9B chat model

- Suggested download source used by app (GGUF artifacts): https://huggingface.co/openresearchtools/Qwen3.5-9B-GGUF
- Original upstream model family reference: https://huggingface.co/Qwen/Qwen3.5-9B
- Runtime artifact format: GGUF.
- Recommendation notice: Transcribe Offline recommends this model as a capable small local chat model for transcript Q&A and anonymisation workflows.
- License and usage terms: users should review the current model repository pages and their linked license/usage terms before use.
- Non-affiliation notice: this recommendation does not imply affiliation with, endorsement by, sponsorship by, or approval from the original Qwen model authors or maintainers.

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

The section below is sourced from the current Openresearchtools-Engine third-party notices file and adapted for embedding inside Transcribe Offline's combined notices document.

This repository contains shipped runtime code, bundled runtime dependencies, native adaptations, and a small amount of repo-kept conversion/parity tooling.

Openresearchtools-Engine source code is licensed under the MIT License. Third-party dependencies and bundled components remain licensed under their respective original licenses.

This file is primarily a provenance map for source code and tooling that were
directly imported, adapted, or rewritten from named upstream projects.
Package-managed dependencies that are consumed in their normal published form
remain listed here at license-inventory level only; they are not broken down
by internal source-file lineage unless this repo directly imported/adapted
their source.

Where mixed-source files are involved, the file lists below are identigying  the main ENGINE files materially informed by a given
upstream project; they are not a claim that every listed line came verbatim
from that upstream.

## Bundle-facing license files

- In this app, full ENGINE third-party license text is available in the embedded `Engine licenses` document.
- This section is based on the current `Openresearchtools-Engine/third_party/README.md` notice file.
- In packaged ENGINE apps, that notice file is shipped as `Third-Party-Notices.md`.

In packaged ENGINE builds, one overinclusive `LICENSES.md` plus one `Third-Party-Notices.md` are shipped at bundle root. The combined license file intentionally over-includes text so one file covers all supported bundle variants.

## First-party shipped ENGINE code

Openresearchtools-Engine source in its own repository, including new native C++ and Rust modules added by that project, is covered by the ENGINE repository MIT license and by the `Openresearchtools-Engine [LICENSE]` section at the top of `Engine licenses`.

This includes shipped ENGINE-authored native/runtime layers such as:

- `bridge/llama_server_bridge.cpp`
- `bridge/llama_server_bridge.h`
- `bridge/llama_server_audio_capture.cpp`
- `bridge/llama_server_audio_capture.h`
- `bridge/llama_server_cluster.cpp`
- `bridge/llama_server_cluster.h`
- `bridge/llama_server_multi_node_server.cpp`
- `bridge/llama_server_multi_node_server.h`
- `multi_node_server/`
- `clusterui/`

The third-party entries below are for upstream code, bundled dependencies, and source provenance that those ENGINE-owned files build on, embed, adapt, or distribute alongside.

## Directly imported or adapted source in shipped runtime

### 1) llama.cpp

- Role: primary upstream C/C++ runtime base that the core ENGINE inference/server stack is built on.
- Upstream source snapshot kept in this repository: `third_party/llama.cpp`
- ENGINE code heavily builds on and extends that base through prepared-build patches plus repo integration layers.
- These ENGINE-added layers are where the native audio/session, In process bridge, Whisper.cpp, Sortformer, and Voxtral features are integrated on top of original llama cpp; they are not claims about upstream `llama.cpp` shipping those features by itself.
- 
- License type: MIT
- License file: `llama.cpp-LICENSE.txt`

### 1a) Additional licenses pulled into this app through the llama.cpp build

The `llama.cpp` CMake build used by this app aggregates extra third-party licenses into its generated `license.cpp`.

- `yhirose/cpp-httplib`
  - Upstream: <https://github.com/yhirose/cpp-httplib>
  - License type: MIT
  - License file: `cpp-httplib-LICENSE.txt`
- `nlohmann/json`
  - Upstream: <https://github.com/nlohmann/json>
  - License type: MIT
  - License file: `jsonhpp-LICENSE.txt`
- `google/boringssl`
  - Upstream: <https://github.com/google/boringssl>
  - License type: ISC-style / BoringSSL license
  - License file: `boringssl-LICENSE.txt`

Build note: BoringSSL is fetched by CMake in build profiles that enable `LLAMA_BUILD_BORINGSSL`.

### 1b) ggml

- Role: low-level tensor/runtime framework used by `llama.cpp`, `whisper.cpp`, and the native realtime subsystems.
- Upstream repository: `ggml-org/ggml`
- Upstream URL: <https://github.com/ggml-org/ggml>
- Source locations in this repository layout:
  - `third_party/llama.cpp/ggml`
  - `third_party/whisper.cpp/ggml`
- License type: MIT
- License file: `ggml-LICENSE.txt`

### 1c) miniaudio

- Role: embedded audio decode/capture helper used by the native whisper audio path.
- Upstream repository: `mackron/miniaudio`
- Upstream URL: <https://github.com/mackron/miniaudio>
- Source location in this repository layout:
  - `third_party/llama.cpp/vendor/miniaudio/miniaudio.h`
- Build-use location in this repository layout:
  - `diarize/addons/overlay/llama.cpp/tools/whisper/whisper-common-audio.cpp`
- License type: Public Domain OR MIT-0
- License file: `miniaudio-LICENSE.txt`

### 1d) webrtc-audio-processing

- Role: optional live microphone cleanup stack used by the native `llama-server-audio` capture bridge before PCM is forwarded into the main bridge session path.
- Upstream repository: `cross-platform/webrtc-audio-processing`
- Upstream URL: <https://github.com/cross-platform/webrtc-audio-processing>
- Runtime bundle location when enabled:
  - `vendor/webrtc-audio-processing`
- License type: BSD-3-Clause style
- License file: `webrtc-audio-processing-LICENSE.txt`

### 1e) Additional ggml CPU-component attributions

- YaRN reference implementation attribution inside ggml CPU rope path
  - Upstream: <https://github.com/jquesnelle/yarn>
  - License type: MIT
  - License file: `yarn-LICENSE.txt`
- llamafile SGEMM component used by ggml CPU backend
  - Upstream: <https://github.com/Mozilla-Ocho/llamafile>
  - License type: MIT
  - License file: `llamafile-sgemm-LICENSE.txt`
- KleidiAI source attribution used by ggml CPU backend when enabled
  - Upstream: <https://github.com/ARM-software/kleidiai>
  - License type: MIT
  - License file: `kleidiai-LICENSE.txt`

### 1f) Additional C/C++ source-attribution licenses

- `openvinotoolkit/openvino`
  - License type: Apache-2.0
  - License file: `openvino-LICENSE.txt`
- `ARM-software/optimized-routines`
  - License type: MIT OR Apache-2.0 WITH LLVM-exception
  - License file: `arm-optimized-routines-LICENSE.txt`
- `cmp-nct/ggllm.cpp`
  - License type: MIT
  - License file: `ggllm.cpp-LICENSE.txt`
- `ivanyu/string-algorithms`
  - License type: Public Domain / Unlicense text
  - License file: `string-algorithms-LICENSE.txt`
- `LostRuins/koboldcpp`
  - License type: MIT
  - License file: `koboldcpp-LICENSE.txt`
- `llvm/llvm-project`
  - License type: Apache-2.0 WITH LLVM-exception
  - License file: `llvm-project-LICENSE.TXT`

### 2) whisper.cpp

- Role: native Whisper transcription implementation integrated into the in-process audio flow.
- Source location in this repository layout: `third_party/whisper.cpp`
- License type: MIT
- License file: `whisper.cpp-LICENSE.txt`

### 3) voxtral-cpp

- Role: primary native `ggml` implementation base adapted for the current Voxtral realtime runtime, and the source origin for the repo-kept Voxtral GGUF conversion tooling.
- ENGINE-owned runtime files adapted from or heavily informed by `voxtral-cpp`:
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.h`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-backend.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-backend.h`
- ENGINE-kept conversion tooling adapted from `voxtral-cpp`:
  - `build/voxtral/convert_voxtral_to_gguf.py`
- Upstream reference points:
  - upstream runtime/math implementation in `voxtral-cpp/src/voxtral.cpp`
  - upstream GGUF conversion tooling in `voxtral-cpp/tools/convert_voxtral_to_gguf.py`
- Upstream reference: local source base mirrored from `voxtral-cpp`
- License type: MIT
- License file: `voxtral-cpp-LICENSE.txt`

### 4) NVIDIA NeMo / Sortformer references

- Role: reference/source for Sortformer archive semantics, config/tensor naming, and parity validation used by the native Sortformer conversion and validation flow only.
- The current Sortformer converter in `build/sortformer/convert_nemo_sortformer_to_gguf.py` is a repo-written ENGINE script, not a copied upstream converter.
- Its source basis was:
  - NVIDIA NeMo Sortformer archive/checkpoint structure
  - NeMo config/tensor naming semantics from the `.nemo` archive
  - NeMo parity/reference runs used during bring-up
- Main ENGINE files materially informed by NVIDIA NeMo / Sortformer references:
  - `build/sortformer/convert_nemo_sortformer_to_gguf.py`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-gguf.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-schema.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-postprocess.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/realtime-smoke.cpp`
- License type: Apache-2.0
- License file: `nvidia-nemo-LICENSE.txt`

### 4a) parakeet.cpp

- Role: external C++ Sortformer reference studied during native Sortformer bring-up, especially for conversion/runtime structure cross-checking.
- Main ENGINE files materially cross-checked or informed against `parakeet.cpp` Sortformer references:
  - `build/sortformer/convert_nemo_sortformer_to_gguf.py`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-model.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-streaming.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-frontend.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/sortformer/sortformer-encoder.cpp`
- Upstream: <https://github.com/frikallo/parakeet.cpp>
- License type: MIT
- License file: `parakeet-cpp-LICENSE.txt`

### 5) docling

- Role: reference logic for VLM document-conversion behavior used by `pdfvlm`.
- Upstream source: <https://github.com/docling-project/docling>
- License type: MIT
- License file: `docling-LICENSE.txt`

### 6) pdfium-render

- Role: PDF rasterization binding used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file: `pdfium-render-LICENSE.md`

### 7) PDFium runtime binaries

- Runtime location: `third_party/pdfium`
- Binary source used in this project: <https://github.com/bblanchon/pdfium-binaries>
- License type: BSD-3-Clause + Apache-2.0 + additional third-party notices
- License file: `pdfium-LICENSE.txt`
- Binary-source license type: MIT
- Binary-source license file: `pdfium-binaries-LICENSE.txt`

Main PDFium upstream notice:

```text
Copyright 2014 The PDFium Authors

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The full PDFium license text, including the Apache License 2.0 portion and the
separately bundled `pdfium-binaries` license, is included in `LICENSES.md`.

### 8) FFmpeg runtime conversion

- Role: in-memory audio normalization and decode/resample path for raw/file audio ingress.
- Windows/Linux binary fetch source: <https://github.com/BtbN/FFmpeg-Builds>
- Source-build reference for macOS arm64: <https://github.com/FFmpeg/FFmpeg>
- License type: LGPL 2.1-or-later intent for the shared-runtime builds staged by this repo
- License files:
  - `ffmpeg-builds-LICENSE.txt`
  - `ffmpeg-LGPL-2.1.txt`
  - `ffmpeg-SOURCE.txt`
  - `ffmpeg-SOURCE-windows-x64.txt`
  - `ffmpeg-SOURCE-ubuntu-x64.txt`
  - `ffmpeg-SOURCE-macos-arm64.txt`

### 9) NVIDIA CUDA runtime libraries

- Role: GPU acceleration runtime libraries used by CUDA backend bundles.
- License type: NVIDIA CUDA EULA
- License files:
  - `nvidia-cuda-EULA.txt`
  - `nvidia-cuda-runtime-NOTICE.txt`

## External reference and validation repositories (not linked into shipped runtime)

These repositories were used as behavior, validation, or benchmarking references during native bring-up. They are not bundled as runtime code by this repository.

### vLLM

- Role: realtime behavior and throughput reference for Voxtral evaluation.
- Main ENGINE files whose Voxtral behavior was materially cross-checked or informed against `vLLM` references:
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.cpp`
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-runtime.h`
- Upstream: <https://github.com/vllm-project/vllm>
- License type: Apache-2.0
- License file: `vllm-LICENSE.txt`

### voxtral.c

- Role: streaming API/reference behavior input for Voxtral session semantics.
- Main ENGINE files materially informed by `voxtral.c` streaming/session ideas:
  - `diarize/addons/overlay/llama.cpp/tools/realtime/voxtral/voxtral-backend.cpp`
  - `bridge/llama_server_bridge.cpp`
  - `engine/src/llama_bridge.rs`
- Upstream: <https://github.com/antirez/voxtral.c>
- License type: MIT
- License file: `voxtral.c-LICENSE.txt`

### voxtral-mini-realtime-rs

- Role: external Rust/GGUF reference for Voxtral realtime integration patterns and session wiring study.
- Upstream: <https://github.com/TrevorS/voxtral-mini-realtime-rs>
- License type: Apache-2.0
- License file: `voxtral-mini-realtime-rs-LICENSE.txt`

### mlx-audio

- Role: external MLX-side reference for Mistral/Voxtral audio model behavior and implementation study.
- Upstream: <https://github.com/Blaizzy/mlx-audio>
- License type: MIT
- License file: `mlx-audio-LICENSE.txt`

## Directly imported or adapted repo-kept conversion / parity tooling (not required at runtime)

This section covers the Python dependency surface used by the repo-kept conversion/parity tooling already described above:

- `build/sortformer/convert_nemo_sortformer_to_gguf.py`
- `build/voxtral/convert_voxtral_to_gguf.py`

Primary direct Python dependencies used by those repo-kept converters:

- NumPy
  - Role: tensor/array handling in both the Sortformer converter and the repo-kept Voxtral GGUF converter
  - License type: BSD-3-Clause with bundled third-party notices
  - License file: `numpy-LICENSE.txt`
- PyTorch (`torch`)
  - Role: checkpoint tensor loading during Sortformer conversion/parity workflows
  - License type: BSD-3-Clause with additional notices
  - License files: `torch-LICENSE.txt`, `torch-NOTICE.txt`
- PyYAML
  - Role: NeMo archive config parsing during Sortformer conversion tooling
  - License type: MIT
  - License file: `PyYAML-LICENSE.txt`

## Runtime integration notes

- Audio patch/overlay mechanism for upstream sync is maintained in `diarize/addons/overlay/llama.cpp/`.
- Bridge runtime integration code is maintained in `bridge/`.
- PDF orchestration modules are in `pdf/` and `pdfvlm/`.
- Bridge raw-audio path supports in-memory conversion via FFmpeg when bridge is built with `LLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON`.

## Package-managed dependency mapping (license inventory only)

### `serde_json`

- Role: JSON parsing/serialization in Rust runtime modules.
- License type: MIT OR Apache-2.0
- License files:
  - `serde_json-LICENSE-MIT.txt`
  - `serde_json-LICENSE-APACHE.txt`

### `anyhow`

- Role: error propagation/context in Rust runtime modules.
- License type: MIT OR Apache-2.0
- License files:
  - `anyhow-LICENSE-MIT.txt`
  - `anyhow-LICENSE-APACHE.txt`

### `clap`

- Role: command-line argument parsing for CLI binaries.
- License type: MIT OR Apache-2.0
- License files:
  - `clap-LICENSE-MIT.txt`
  - `clap-LICENSE-APACHE.txt`

### `once_cell`

- Role: one-time/lazy static initialization in runtime modules.
- License type: MIT OR Apache-2.0
- License files:
  - `once_cell-LICENSE-MIT.txt`
  - `once_cell-LICENSE-APACHE.txt`

### `regex`

- Role: regular-expression matching used by runtime text processing paths.
- License type: MIT OR Apache-2.0
- License files:
  - `regex-LICENSE-MIT.txt`
  - `regex-LICENSE-APACHE.txt`

### `walkdir`

- Role: filesystem traversal in runtime file-processing paths.
- License type: Unlicense OR MIT
- License files:
  - `walkdir-UNLICENSE.txt`
  - `walkdir-LICENSE-MIT.txt`
  - `walkdir-COPYING.txt`

### `image`

- Role: image buffer/format handling in runtime document/VLM processing paths.
- License type: MIT OR Apache-2.0
- License files:
  - `image-LICENSE-MIT.txt`
  - `image-LICENSE-APACHE.txt`

### `encoding_rs`

- Role: encoding conversion for text handling in runtime paths.
- License type: Apache-2.0 OR MIT plus WHATWG text
- License files:
  - `encoding_rs-LICENSE-MIT.txt`
  - `encoding_rs-LICENSE-APACHE.txt`
  - `encoding_rs-LICENSE-WHATWG.txt`

### `pdfium-render`

- Role: Rust binding layer to PDFium used by `pdf` and `pdfvlm`.
- License type: MIT OR Apache-2.0
- License file: `pdfium-render-LICENSE.md`

### `eframe` / `egui`

- Role: native immediate-mode app shell, widgets, layout, rendering integration, and controller UI foundations for `clusterui`.
- License type: MIT OR Apache-2.0
- License files:
  - `eframe-LICENSE-MIT.txt`
  - `eframe-LICENSE-APACHE.txt`
  - `egui-LICENSE-MIT.txt`
  - `egui-LICENSE-APACHE.txt`

### `egui_commonmark`

- Role: Markdown/README rendering inside the controller UI.
- License type: MIT OR Apache-2.0
- License files:
  - `egui_commonmark-LICENSE-MIT.txt`
  - `egui_commonmark-LICENSE-APACHE.txt`

### `axum` / `tokio` / `tower-http`

- Role: embedded cluster HTTP API server, async orchestration, and HTTP middleware for `clusterui`.
- License types:
  - `axum`: MIT
  - `tokio`: MIT
  - `tower-http`: MIT
- License files:
  - `axum-LICENSE.txt`
  - `tokio-LICENSE.txt`
  - `tower-http-LICENSE.txt`

### `reqwest`

- Role: controller-side HTTP client for runtime downloads, metadata fetches, and remote artifact transfer helpers.
- License type: MIT OR Apache-2.0
- License files:
  - `reqwest-LICENSE-MIT.txt`
  - `reqwest-LICENSE-APACHE.txt`

### `rfd` / `tray-icon`

- Role:
  - `rfd`: native file/folder dialogs in the controller UI
  - `tray-icon`: system tray integration for the controller app
- License types:
  - `rfd`: MIT
  - `tray-icon`: MIT OR Apache-2.0
- License files:
  - `rfd-LICENSE.txt`
  - `tray-icon-LICENSE-MIT.txt`
  - `tray-icon-LICENSE-APACHE.txt`

### `base64` / `bincode` / `sha2`

- Role: transport encoding, binary protocol framing, and artifact integrity hashing in `clusterui`.
- License types:
  - `base64`: MIT OR Apache-2.0
  - `bincode`: MIT
  - `sha2`: MIT OR Apache-2.0
- License files:
  - `base64-LICENSE-MIT.txt`
  - `base64-LICENSE-APACHE.txt`
  - `bincode-LICENSE.txt`
  - `sha2-LICENSE-MIT.txt`
  - `sha2-LICENSE-APACHE.txt`

### `sysinfo` / `time`

- Role:
  - `sysinfo`: node/device/process telemetry and resource reporting in `clusterui`
  - `time`: runtime timestamps, static clock labels, and controller-side time formatting
- License types:
  - `sysinfo`: MIT
  - `time`: MIT OR Apache-2.0
- License files:
  - `sysinfo-LICENSE.txt`
  - `time-LICENSE-MIT.txt`
  - `time-LICENSE-APACHE.txt`

### `tar` / `zip`

- Role: bundle/runtime archive staging and packaged dependency handling in the controller/runtime installer paths.
- License types:
  - `tar`: MIT OR Apache-2.0
  - `zip`: MIT
- License files:
  - `tar-LICENSE-MIT.txt`
  - `tar-LICENSE-APACHE.txt`
  - `zip-LICENSE.txt`

### `windows-sys`

- Role: Windows-specific networking, interface discovery, and system integration paths used by the controller app.
- License type: MIT OR Apache-2.0
- License files:
  - `windows-sys-LICENSE-MIT.txt`
  - `windows-sys-LICENSE-APACHE.txt`

## Rust transitive license export

- Full transitive Rust crate export (current non-dev shipped graphs):
  - workspace Windows graph
  - `clusterui` macOS graph
  - exported to `rust-full/`

## Checked-in tooling license snapshot

This repo also keeps a Python-tooling license snapshot at:

- <https://github.com/openresearchtools/engine/tree/main/third_party/licenses/tooling-full>

It is not part of the shipped runtime bundle.
