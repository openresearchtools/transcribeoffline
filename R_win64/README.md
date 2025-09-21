# Transcribe Offline (Windows, RStudio)

Local-first audio/video transcription with optional **word alignment** and **speaker diarisation**, plus built‑in tools for **translation**, **summarisation**, and **grammar correction** using a local LLM. Everything runs **offline on your PC** — no uploads.
---

## Highlights

- **Fast transcription:** `faster-whisper` (CTranslate2) for high‑performance Whisper on CPU.  
- **Accurate timestamps:** Optional **WhisperX alignment** (English only) refines word timings.  
- **Who said what:** **pyannote.audio** diarisation assigns speakers by segment or word.  
- **Subtitles & exports:** Save **TXT / JSON / CSV / SRT / VTT**.  
- **Local LLM tooling:** Run translation, summarisation, or copy‑editing with **llama.cpp** (CPU only).  
- **Zero network:** App blocks sockets to guarantee **offline** processing.

---

## Primary packages used

- **faster-whisper** — high‑performance Whisper inference (CTranslate2 backend)  
- **WhisperX** — word‑level alignment and VAD support  
- **pyannote.audio** — speaker diarization  
- **Transformers** — model utilities and tokenizers  
- **sounddevice / soundfile** — audio playback  
- **FFmpeg (LGPL v2.1+)** — bundled win64 `lgpl-shared` build from BtbN (not statically linked)

### Local LLMs
- **llama.cpp** — CPU‑only binaries used to run local models  
- **Default model reference:** Qwen3 4‑bit

---

## Install & Run via RStudio (Windows 64‑bit)

> **Important:** Supported **only in RStudio on Windows x64** (tested in RStudio). Other R IDEs are not supported.

### One‑time setup

1. Open **RStudio (Windows 64‑bit)**.  
2. Download **`setup_transcribe_offline.R`**.  
3. Open the script in RStudio, **select all**, and click **Run**.  
   - By default, the script installs the app into your **Downloads** folder.  
   - After installation, a **model download GUI** appears so you can fetch speech/LLM models.

### Launching the app after install

- Use the generated **`run_transcribe_offline.R`** (created by the setup script).  
- If you see errors on first run, set the working directory to the app folder and retry:
  - RStudio → **Session → Set Working Directory → Choose Directory…** → select **`Transcribe_Offline`**
  - Run **`run_transcribe_offline.R`** again.

---

## Basic usage

1. Start the app.  
2. **Add media** (audio/video) in the “Media files (inputs)” list.  
3. Pick **Language**, **Mode** (Transcribe or Subtitles), and toggles:  
   - **Alignment (English only)** — improves word timings for English.  
   - **Diarisation** — assigns speakers. Supports “auto” or a fixed number.  
   - **Max subtitle duration** — hard cap per subtitle when exporting SRT/VTT.  
4. Choose **Auto‑save** formats (TXT / JSON / CSV / SRT / VTT).  
5. Click **Run Batch ▶**. Results appear in “Output files (results)”.  
6. Double‑click any transcript line to **play from that time**. Use **speed** control as needed.  
7. (Optional) Use **Translate / Summarise / Correct text / Custom prompt** to run the local LLM on the selected output file.

### Output formats

- **TXT:** Speaker‑merged lines like  
  `[00:00–00:06] Speaker01: Hello there…`  
- **SRT / VTT:** Timed captions; **Max subtitle dur** limits the per‑cue length.  
- **JSON / CSV:** Machine‑readable exports with `start`, `end`, and `text` fields.

---

## Alignment & diarisation logic (Windows build)

- **Transcribe → Align → Diarise** with **memory flush between stages**.  
- **Alignment (English only):** Uses a local alignment model to add word‑level timestamps. If alignment can’t run, the app continues with segment‑level timestamps.  
- **Diarisation:**  
  - If aligned word timestamps exist, the app performs **word‑level speaker assignment**.  
  - Otherwise, it falls back to **segment‑level overlap** assignment.  
- During export, the app merges segments **by speaker** (word‑level if available), producing clean speaker turns.

---

## Privacy

- The app **blocks all network sockets** at startup.  
- Models and audio are processed locally. No telemetry, no uploads.

---

## System requirements (minimum)

- **Windows 10/11 64‑bit**  
- **RStudio (Windows x64)** installed  
- Sufficient disk space for models (several GB for Whisper/LLM models)  
- CPU with multiple cores recommended for faster processing  
- Audio output device (for in‑app playback)

---

## Troubleshooting

**The app didn’t start from RStudio**  
- Ensure you ran **`setup_transcribe_offline.R`** to completion.  
- Set working directory to **`Transcribe_Offline`** and run **`run_transcribe_offline.R`** again.  

**Model download GUI didn’t appear**  
- Re‑run **`setup_transcribe_offline.R`**; it will re‑validate content and prompt for models.

**No audio playback**  
- The app requires `sounddevice`/`soundfile` and a working audio device. Make sure Windows Output is not muted.

**Alignment skipped**  
- Alignment is **English only**. For other languages, diarisation still works at segment level.

**High memory usage**  
- Long files are supported, but memory use scales with duration. The app frees caches between **Transcribe → Align → Diarise** stages to stay within limits.

---

## Licensing notes

- **FFmpeg** is bundled as a **win64 `lgpl-shared`** build (BtbN), **not statically linked**, and is covered by **LGPL v2.1+**.  
- Other components are used under their respective licenses. See the **licenses** folder (installed by the setup script) for details.

---

## Uninstall

1. Delete the **Transcribe_Offline** folder created by the setup script (default in **Downloads**).  
2. Optionally remove any downloaded model folders (they can be several GB).

---

© Transcribe Offline — Windows (RStudio) edition.
