# Transcribe Offline v1.2

**Transcribe Offline** is a desktop application that allows you to transcribe audio and video fully offline,
with optional speaker diarization and word-level alignment. It can also generate subtitles and integrate
with local large language models (LLMs) for summarization and editing. The app is designed to keep all
media private on your device, but as globally cyber threats are increasing, when working with sensitive data,
we would recommend setting up additional firewall or using on the machines with no access to network.

## Primary packages used

* [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — high‑performance Whisper inference (CTranslate2 backend)
* [WhisperX](https://github.com/m-bain/whisperX) — word‑level alignment and VAD support
* [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
* [Transformers](https://github.com/huggingface/transformers) — model utilities and tokenizers
* [Sounddevice / Soundfile](https://python-sounddevice.readthedocs.io/) — audio playback
* **FFmpeg** (LGPL v2.1+) bundled **win64 lgpl‑shared** build from BtbN.

**Local LLMs:**

* [llama.cpp](https://github.com/ggml-org/llama.cpp) — CPU‑only binaries used to run local models
* Default model reference: **Qwen3 4‑bit**

---

## How to cite

**Suggested citation**  
Rutkauskas, L. (2025). *Transcribe Offline* (Version 1.2) [Computer software]. openresearchtools.com. https://github.com/openresearchtools/transcribeoffline. MIT Licence. Released 25 September 2025.

---

***Disclaimer & Licence***

Transcribe Offline is open-source software distributed under the MIT Licence. A copy of the Licence is included with the project and is named Licence.

A list of third-party Licences is documented and included (see Third Party Licences).

In summary, the software is provided “as is”, without warranty of any kind, expressed or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

By downloading, installing, or using this software you acknowledge that you do so at your own risk and that the authors, maintainers, and contributors accept no responsibility for any loss, damage, or other consequences resulting from its use.

Note: This project may interface with third-party models or tools that are subject to their own Licences and terms. You are responsible for ensuring your use complies with those terms.
