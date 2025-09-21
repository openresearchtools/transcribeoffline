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



## License information

This project’s own code is licensed under the MIT License (see LICENSE).
A list of third‑party licenses is documented and included (see **Third-Party-Licenses.txt**).

