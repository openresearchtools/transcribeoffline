# Transcribe Offline

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b133851-6849-42fd-8d0e-a5cbc60b7e35" width="760" alt="Transcribe Offline">
</p>

**Transcribe Offline** is an open‑source desktop application for **on‑device transcription** with **speaker diarisation** (assigning who said what) and **word‑level alignment** for crisp subtitles. It also includes local LLM helpers for summarisation, translation, and light editing — all without sending audio to remote servers by default.

> *Designed to be local‑first. However, no software can guarantee absolute privacy or security. Please consider your threat model and institutional policies before processing sensitive material.*

---

## Highlights

- **Offline by default.** Process recordings on your own machine — no accounts or routine uploads.  
- **Predictable costs.** Open source and free to use — no per‑minute API bills; compute runs on your CPU/GPU.  
- **Research‑friendly.** Reproducible settings and versioned models help you document methods for scholarly work.  
- **Flexible exports.** Clean transcripts, subtitles, and data‑friendly formats that drop into existing workflows.

---

## What it does

### 1) Transcription you can trust
Built on **OpenAI Whisper**, producing robust, punctuated transcripts across diverse accents and domains.

### 2) Word‑level alignment (English)
**WhisperX** and open‑source audio‑labelling pipelines refine timestamps down to each word, enabling:

- Frame‑accurate **SRT/VTT** with natural line breaks  
- Click‑to‑play navigation (double‑click a line to jump audio)  
- Playback‑synced highlighting that follows words in real time

### 3) Speaker diarisation
**pyannote** separates speakers and assigns consistent labels, producing:

- Speaker‑attributed paragraphs (Speaker 01/02/…)  
- When alignment is available, **speaker labels at the word level** (excellent for subtitles)

### 4) AI‑assisted editing on your desk
Run a small **local LLM (Qwen 3)** for:

- Summaries of long sessions  
- Draft translations (quality varies by language)  
- Grammar and punctuation clean‑ups  
- Custom prompts to “talk to your transcript” for outlines, show notes, or action lists  

---

## Outputs

- **Readable transcripts** TXT with timestamps and Speaker 01/02 labels  
- **Subtitles:** SRT / VTT  
- **Data‑friendly formats:** CSV / JSON for analysis and search  
- **Editor quality‑of‑life:** playback‑synced highlighting; double‑click any line to jump; edits autosave alongside your audio

---

## Who is it for?

Researchers, lecturers, and professional practitioners who need dependable, local transcription:

- **Qualitative & ethnographic research:** interviews, focus groups, field recordings  
- **Lecture capture & seminars:** searchable notes and teaching materials  
- **Media & podcasts:** subtitle passes and episode notes  
- **Linguistics & HCI:** fine‑grained timing for annotation workflows

---

## How it works (under the bonnet)

- **Transcription:** Whisper models  
- **Alignment:** WhisperX + open‑source audio labelling for word‑level timings (English)  
- **Diarisation:** pyannote pipelines for speaker turns (labels such as Speaker 01/02/…)  
- **Local LLM:** Qwen 3 for summarise/translate/correct/custom prompts  ---

## Compatibility & performance

- Designed for modern desktops and laptops.  
- **Memory:** ~16 GB RAM is a sensible baseline for comfortable use on MacOS and a must on Windows machines.

---

## Downloads & project links

- **macOS project page:** <https://github.com/openresearchtools/transcribeoffline/tree/main/macOS>  
- **Windows project page:** <https://github.com/openresearchtools/transcribeoffline/tree/main/Windows>  
- **RStudio integration on Windows:** <https://github.com/openresearchtools/transcribeoffline/tree/main/R_win64>

> If you’re evaluating for institutional use, test with non‑sensitive audio first, then review logs/exports with your IT or data guardian.

---

## Quick tour

1. **Add audio/video.** Drag files in and queue for processing.  
2. **Choose tasks.** Transcribe; optionally enable diarisation and word‑alignment.  
3. **(Optional) Apply local LLM helpers.** Summarise, translate, or tidy the prose.  
4. **Export.** Save TXT/JSON/CSV or SRT/VTT and drop them straight into your analysis or editing workflow.  
5. **Navigate.** Double‑click any line to jump playback; highlighting follows the audio.

---

## Limitations

- **Alignment:** word‑level alignment is currently only available in English.  
- **Diarisation accuracy:** depends on audio quality and speaker overlap; manual review is recommended for high‑stakes use.  
- **LLM helpers:** outputs may contain mistakes; treat them as drafting aids rather than ground truth.  
- **Accents & domains:** Whisper is strong overall, but niche jargon or heavy code‑switching may need light edits.

---

## Contributing

We welcome issues and pull requests. For sizeable changes, please open an issue first to discuss what you’d like to add or modify. Bug reports that include logs, hardware specs, OS details, and a small sample help us reproduce problems quickly.

---

## Licence

Released under the **MIT Licence**. Third‑party models and tools are subject to their own licences and terms — please review the linked model pages.

---

<p align="center">
  <sub>Built as an open tool for research and teaching. Stars, issues, and community examples are very welcome.</sub>
</p>
