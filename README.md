

# Transcribe Offline
<p align="center">
  <img src="https://github.com/user-attachments/assets/9b133851-6849-42fd-8d0e-a5cbc60b7e35" width="760" alt="Transcribe Offline">
</p>**Transcribe Offline** by <a href="https://openresearchtools.com">openresearchtools.com</a> is a desktop application for **fully local** speech‑to‑text with optional **speaker diarisation** and **word‑level alignment**. It can also create **subtitles** and plug into local LLMs for **summaries** and **light edits** — all on your machine, with no cloud uploads.

---

## Highlights
- Robust transcription (Whisper / WhisperX)
- Automatic **speaker diarisation** (pyannote)
- **Word‑level alignment** for crisp subtitles
- Optional local LLM helpers (summarise / translate / correct / custom prompts)
- Exports: **TXT, JSON, CSV, SRT, VTT**
- **Privacy by design** — audio never leaves your computer

---

## Platform support

- **macOS (Apple Silicon)** only for now (M1/M2/M3/M4).  
- **Intel Macs** are **not supported**.  
- **Windows** version is coming soon!.

---
## Option 1) Pre‑compiled app

You can download a pre‑built app here:  
**Downloads:** https://openresearchtools.com/downloadto/

We’re a small open‑source team and do not currently hold a paid Apple Developer subscription for notarisation. As a result, macOS may block the first launch.

### Opening an unnotarised app on **macOS Sequoia (15)**

1. Attempt to open **Transcribe Offline** once (double‑click in **Applications**). It will be blocked.
2. Open **System Settings** → **Privacy & Security**.
3. Scroll to **Security**. You should see *“‘Transcribe Offline’ was blocked from use because it is not from an identified developer.”*
4. Click **Open Anyway**, then confirm **Open** when prompted.

After this first approval, it should open normally.

## Option 2) One‑command install (builds on your system)

This compiles the app **on your machine** and installs it into **/Applications**.
Open Terminal and paste the following command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/macOS/install.sh)"
```

**What the installer does**
- Checks / installs **Xcode Command Line Tools**, **Homebrew**, and **python‑tk@3.11**.
- Creates an isolated environment and installs pinned Python packages.
- Downloads required models 
- For diarisation models, it **prompts for a Hugging Face token** (see below).
- Builds FFmpeg with static Opus (LGPL 2.1 licensed).
- Bundles the app with PyInstaller, moves **Transcribe Offline.app** to **/Applications**.
- Tries to launch it.

> Building locally avoids “unidentified developer” warnings when you first open the app.

---

## !!!IMPORTANT!!! Access for diarisation models 

1. **Sign in / create** a Hugging Face account: https://huggingface.co/  
2. While signed in, **accept the terms** on each model page:  
   - https://huggingface.co/pyannote/segmentation-3.0  
   - https://huggingface.co/pyannote/speaker-diarization-3.1  
3. **Generate an access token:** https://huggingface.co/settings/tokens  
   *(setup guide: https://huggingface.co/docs/hub/en/security-tokens)*

During installation you’ll be **prompted for the token two times, just paste it and press enter**. If it’s invalid or access isn’t granted yet, the installer explains what to do and asks again until it works.

---

## Quick use

1. **Open** *Transcribe Offline*.
2. **Add…** audio/video files.
3. Choose **Transcribe** (or **Subtitles**) and toggle **Diarisation** if needed.
4. Optionally **Summarise / Translate / Correct** using the built‑in tools.
5. Export to **TXT / JSON / CSV / SRT / VTT**.
6. Click any line to jump playback; highlighting follows as audio plays.

---

## Licence

Released under the **MIT Licence** (see `LICENCE`).  
Third‑party models and tools are subject to their own licences and terms — please review the linked model pages.

---

<p align="center">
  <sub>Built by a small team making open tools for science. If you find this useful, stars and issues are welcome.</sub>
</p>
