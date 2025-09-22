# Transcribe Offline (Windows x64)
<img width="1265" height="934" alt="potatoes" src="https://github.com/user-attachments/assets/e73618c4-ccd1-4fe8-a7cc-e2800aa65ef2" />

</p>**Transcribe Offline** by <a href="https://openresearchtools.com">openresearchtools.com</a> is a desktop application for **fully local** speech‑to‑text with optional **speaker diarisation** and **word‑level alignment**. It can also create **subtitles** and plug into local LLMs for **summaries** and **light edits** — all on your machine, with no cloud uploads.

---

## Highlights

- **Fast transcription:** `faster-whisper` (CTranslate2) for high‑performance Whisper on CPU.  
- **Accurate timestamps:** Optional **WhisperX alignment** (English only) refines word timings.  
- **Who said what:** **pyannote.audio** diarisation assigns speakers by segment or word.  
- **Subtitles & exports:** Save **TXT / JSON / CSV / SRT / VTT**.  
- **Local LLM tooling:** Run translation, summarisation, or copy‑editing with **llama.cpp** (CPU only).  
---

## Primary packages used

- **faster-whisper** — high‑performance Whisper inference (CTranslate2 backend)  
- **WhisperX** — word‑level alignment and VAD support  
- **pyannote.audio** — speaker diarization  
- **Transformers** — model utilities and tokenizers  
- **sounddevice / soundfile** — audio playback  
- **FFmpeg (LGPL v2.1+)** — bundled win64 `lgpl-shared` build from BtbN

### Local LLMs
- **llama.cpp** — CPU‑only binaries used to run local models  
- **Default model reference:** Qwen3 4‑bit

---

## Option 2 Install & Run via RStudio (Windows 64‑bit)

> **Important:** Supported **only in RStudio on Windows x64** (tested in RStudio). Other R IDEs are not supported.

### One‑time setup

1. Open **RStudio (Windows 64‑bit)**.
2. Download **`[setup_transcribe_offline.R](https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/R_win64/setup_transcribe_offline.R)
`**.  
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

## System requirements (minimum)

- **Windows 10/11 64‑bit**  
- **RStudio (Windows x64)** installed  
- **16GB RAM***
- **10GB Disk Space***
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

***Disclaimer & Licence***

Transcribe Offline is open-source software distributed under the MIT Licence. A copy of the Licence is included with the project and is named Licence.

A list of third-party Licences is documented and included (see Third Party Licences).

In summary, the software is provided “as is”, without warranty of any kind, expressed or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

By downloading, installing, or using this software you acknowledge that you do so at your own risk and that the authors, maintainers, and contributors accept no responsibility for any loss, damage, or other consequences resulting from its use.

Note: This project may interface with third-party models or tools that are subject to their own Licences and terms. You are responsible for ensuring your use complies with those terms.
