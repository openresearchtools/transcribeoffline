

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

## Primary packages used

- [MLX](https://github.com/ml-explore/mlx) — Apple’s array/ML framework used as the on-device compute runtime (Metal) on Apple Silicon  
- [mlx_whisper](https://pypi.org/project/mlx-whisper/) — Whisper transcription running on MLX (fast on-device inference)  
- [mlx_lm](https://pypi.org/project/mlx-lm/) — Local LLM inference on MLX for Translate / Summarise / Correct / Custom Prompt  

- [Whisper / WhisperX](https://github.com/m-bain/whisperX) for transcription and alignment  
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarisation  
- [Transformers](https://github.com/huggingface/transformers) for LLM and model utilities  
- [Sounddevice / Soundfile](https://python-sounddevice.readthedocs.io/) for audio playback  
- FFmpeg bundled with Opus.
- QWEN 3 (https://huggingface.co/Qwen) for LLMs.

These in turn depend on other packages and libraries. All dependencies are included in the 
PyInstaller build and distributed with the application.

---

## Platform support

- **macOS (Apple Silicon)** only for now (M1/M2/M3/M4 - with recommended minimum 16GB of RAM, smaller audio files can work on 8GB systems).  
- **Intel Macs** are **not supported**.  
--**Windows Version** is here https://github.com/openresearchtools/transcribeoffline/tree/main/Windows 
---
## Option 1) Pre‑compiled app

You can download a pre‑built app here:  
**Downloads:** [Transcribe Offline v1.2 MacOS](https://drive.google.com/file/d/1Kbqw0hG9O01G_oUT2V5FdNDdQGzrTv1R/view?usp=sharing)

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

## Disclaimer & Licence

Transcribe Offline is open-source software distributed under the MIT Licence. A copy of the Licence is included with the project and is named Licence.

A list of third-party Licences is documented and included (see Third Party Licences).

In summary, the software is provided “as is”, without warranty of any kind, expressed or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

By downloading, installing, or using this software you acknowledge that you do so at your own risk and that the authors, maintainers, and contributors accept no responsibility for any loss, damage, or other consequences resulting from its use.

Note: This project may interface with third-party models or tools that are subject to their own Licences and terms. You are responsible for ensuring your use complies with those terms.
