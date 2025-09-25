# Transcribe Offline (Built using Ubuntu 24.04 LTS)
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
- FFmpeg — not bundled but user installation required
### Local LLMs
- **llama.cpp** — CPU‑only binaries used to run local models  
- **Default model reference:** Qwen3 4‑bit

---
## Run via Python 3.11-Tk (Important part is TK, as default Python does not ship Tkinter on Linux)

### One‑time setup

1. Make sure you have **FFmpeg** installed. If not, open **Terminal** and use the following command:
    ```
    sudo apt-get update && sudo apt-get install -y ffmpeg
    ```
2. Run the following command in **Terminal**.
    ```
   bash -c 'set -e; URL="https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/Linux/setup_transcribe_offline_ubuntu.sh"; if command -v curl >/dev/null 2>&1; then curl -fsSL "$URL" | bash; elif command -v wget >/dev/null 2>&1; then wget -qO- "$URL" | bash; else sudo apt-get update -y && sudo apt-get install -y curl && curl -fsSL "$URL" | bash; fi'
    ```
By default, script checks if you have **Python 3.11-Tk** (Important part is TK, as default Python does not ship Tkinter on Linux), if not, then it installs Python 3.11-Tk, creates virtual environment and installs the app into your **Downloads** folder.  
3. Follow the instructions for gated models and download the models in the window that opens


### Launching the app after install
**Desktop Shortcuts**, for model downloader and the app itself will be created, or you can run from app folder by using **secondary click --> run as a program** on **run_transcribe_offline.sh**

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

- **Ubuntu based 64‑bit system** 
- **16GB RAM***
- **10GB Disk Space***
- CPU with multiple cores recommended for faster processing  
- Audio output device (for in‑app playback)

---

***Disclaimer & Licence***

Transcribe Offline is open-source software distributed under the MIT Licence. A copy of the Licence is included with the project and is named Licence.

A list of third-party Licences is documented and included (see Third Party Licences).

In summary, the software is provided “as is”, without warranty of any kind, expressed or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

By downloading, installing, or using this software you acknowledge that you do so at your own risk and that the authors, maintainers, and contributors accept no responsibility for any loss, damage, or other consequences resulting from its use.

Note: This project may interface with third-party models or tools that are subject to their own Licences and terms. You are responsible for ensuring your use complies with those terms.
