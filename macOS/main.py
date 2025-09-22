from __future__ import annotations
# ======================= Transcribe Offline — main.py =======================
# Single-file, offline-only app (Apple Silicon).
# - No internet access (blocks ALL sockets including localhost)
# - Uses bundled ffmpeg at content/vendor/ffmpeg
# - Uses ONLY bundled models at content/models/*
# - No downloads at runtime

import os, sys, gc, json, csv, time, queue, shutil, traceback, threading, re, subprocess, contextlib, tempfile, importlib
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont  # NEW: for min-height calculations
import numpy as np
import multiprocessing as mp  # subprocess stages

# --------------------------- Offline guard (strict) ---------------------------
def _offline_enable():
    import socket
    if getattr(_offline_enable, "_ON", False):
        return
    _offline_enable._ON = True
    _orig_socket = socket.socket
    class _NoNetSock(socket.socket):
        def connect(self, *a, **k): raise OSError("Network blocked (Transcribe Offline)")
        def connect_ex(self, *a, **k): raise OSError("Network blocked (Transcribe Offline)")
    def _raise(*a, **k): raise OSError("Network blocked (Transcribe Offline)")
    socket.socket = _NoNetSock
    socket.create_connection = _raise
    socket.getaddrinfo = _raise
    if hasattr(socket, "create_server"):
        socket.create_server = _raise
_offline_enable()

# Caches → temp so nothing lands in user home
TMP_CACHE = Path(tempfile.mkdtemp(prefix="transcribe_offline_cache_"))
os.environ.update({
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "NO_PROXY": "*", "no_proxy": "*",
    "HF_HOME": str(TMP_CACHE / "hf"),
    "HUGGINGFACE_HUB_CACHE": str(TMP_CACHE / "hf"),
    "HF_DATASETS_CACHE": str(TMP_CACHE / "hf"),
    "TORCH_HOME": str(TMP_CACHE / "torch"),
    "XDG_CACHE_HOME": str(TMP_CACHE / "xdg"),
    "PYANNOTE_CACHE": str(TMP_CACHE / "pyannote"),
    "TOKENIZERS_PARALLELISM": "false",
})

# --------------------------- Packaging helpers ---------------------------
APP_NAME = "Transcribe Offline"

def _resource_base() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

CONTENT = _resource_base() / "content"
MODELS  = CONTENT / "models"
VENDOR  = CONTENT / "vendor"

def _dep_missing(pkg: str, feature: str = ""):
    frozen = getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")
    if frozen:
        lines = [f"This build is missing a required component: {pkg}.", ""]
        if feature: lines.append(f"Feature: {feature}")
        lines.append("Please rebuild/download the app that includes it.")
        msg = "\n".join(lines)
    else:
        msg = f"Missing dependency: {pkg}.\nInstall in your environment:\n\n    pip install {pkg}\n"
    messagebox.showerror("Missing dependency", msg)

def ensure_ffmpeg_on_path() -> str:
    ff = VENDOR / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")
    if not ff.exists():
        messagebox.showerror("FFmpeg missing", f"Bundled ffmpeg not found at:\n{ff}\n\nPlace ffmpeg in content/vendor/")
        raise SystemExit(1)
    os.environ["PATH"] = f"{str(VENDOR)}{os.pathsep}{os.environ.get('PATH','')}"
    try:
        subprocess.run([str(ff), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        messagebox.showerror("FFmpeg error", "Bundled ffmpeg is not executable.")
        raise SystemExit(1)
    return str(ff)

def ffmpeg_path() -> str:
    return ensure_ffmpeg_on_path()

def _model_dir(label: str) -> Path | None:
    base = MODELS
    p = base / label
    if p.exists(): return p
    p2 = base / label.replace("/", "__")
    if p2.exists(): return p2
    parts = label.split("__")
    p3 = base.joinpath(*parts)
    if p3.exists(): return p3
    last = label.split("/")[-1].split("__")[-1]
    for cand in base.rglob(last):
        if cand.is_dir(): return cand
    return None

# --------------------------- Hardcoded models (bundled) ---------------------------
WHISPER_DIR_LABEL = "mlx-community__whisper-large-v3-turbo"
ALIGN_EN_LABEL    = "facebook__wav2vec2-base-960h"
PYA_SEG_LABEL     = "pyannote__segmentation-3.0"
PYA_EMB_LABEL     = "pyannote__wespeaker-voxceleb-resnet34-LM"
QWEN_DIR_LABEL    = "Qwen__Qwen3-8B-MLX-4bit"

# --------------------------- LLM backends (optional) ---------------------------
HAS_MLX_LM=False
try:
    from mlx_lm import load as _mlx_load, generate as _mlx_generate; HAS_MLX_LM=True
    try:
        from mlx_lm.sample_utils import make_sampler as _make_sampler
    except Exception:
        _make_sampler = None
except Exception: pass

# --------------------------- Cross-platform audio (required) ---------------------------
HAS_SD = False
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SD = True
except Exception:
    HAS_SD = False

class _SDStreamer:
    """Persistent OutputStream using sounddevice; supports load/seek/pause without re-opening device.
       CHANGE: Output stream is always stereo; mono sources are duplicated to both channels."""
    def __init__(self):
        if not HAS_SD:
            raise RuntimeError("sounddevice/soundfile not installed")
        self._stream: sd.OutputStream | None = None
        self._sf: sf.SoundFile | None = None
        self._mutex = threading.Lock()
        self._sr = 0
        self._in_ch = 0
        self._paused = True
        self._start_frames = 0
        self._pos_frames = 0
        self._loaded_path: str | None = None
        self._ended_flag = False

    def _ensure_stream(self, sr: int, in_ch: int):
        recreate = False
        if self._stream is None:
            recreate = True
        else:
            if self._sr != sr:
                recreate = True

        if recreate:
            self._close_stream_only()
            self._sr, self._in_ch = int(sr), int(in_ch)

            def _cb(outdata, frames, time_info, status):
                if status:
                    pass
                with self._mutex:
                    if self._sf is None or self._paused:
                        outdata.fill(0); return
                    data = self._sf.read(frames, dtype="float32", always_2d=True)
                    n = data.shape[0]
                    if n == 0:
                        outdata.fill(0)
                        self._paused = True
                        self._ended_flag = True
                        return
                    # --- Upmix/downmix to STEREO (2 channels) ---
                    if data.shape[1] == 1:
                        # duplicate mono → stereo
                        if n < frames:
                            # pad with zeros if short
                            outdata[:n, 0] = data[:, 0]
                            outdata[:n, 1] = data[:, 0]
                            outdata[n:, :].fill(0)
                        else:
                            outdata[:frames, 0] = data[:frames, 0]
                            outdata[:frames, 1] = data[:frames, 0]
                    else:
                        # use first two channels
                        if n < frames:
                            outdata[:n, 0] = data[:, 0]
                            outdata[:n, 1] = data[:, 1]
                            outdata[n:, :].fill(0)
                        else:
                            outdata[:frames, 0] = data[:frames, 0]
                            outdata[:frames, 1] = data[:frames, 1]

                    self._pos_frames += min(frames, n)
                    if n < frames:
                        self._paused = True
                        self._ended_flag = True

            # Always create a 2-channel output stream so mono files play center
            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=2,
                dtype="float32",
                blocksize=2048,
                latency="high",
                callback=_cb,
            )
            self._stream.start()

    def _close_stream_only(self):
        try:
            if self._stream:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self._stream = None

    def load(self, wav_path: str):
        f = sf.SoundFile(wav_path, mode="r")
        sr, ch = int(f.samplerate), int(f.channels)
        with self._mutex:
            if self._sf:
                try: self._sf.close()
                except Exception: pass
            self._sf = f
            self._loaded_path = wav_path
            self._start_frames = 0
            self._pos_frames = 0
            self._paused = True
            self._ended_flag = False
        self._ensure_stream(sr, ch)

    def play(self, start_sec: float = 0.0):
        if self._sf is None:
            raise RuntimeError("No file loaded")
        with self._mutex:
            self._ended_flag = False
            frame = max(0, int(float(start_sec) * self._sr))
            try:
                self._sf.seek(frame)
            except Exception:
                self._sf.seek(0); frame = 0
            self._start_frames = frame
            self._pos_frames = 0
            self._paused = False

    def pause(self, on: bool):
        with self._mutex:
            self._paused = bool(on)

    def toggle_pause(self) -> bool:
        with self._mutex:
            self._paused = not self._paused
            return self._paused

    def stop(self):
        with self._mutex:
            self._paused = True
            self._pos_frames = 0
            self._start_frames = 0
            self._ended_flag = False
            try:
                if self._sf:
                    self._sf.seek(0)
            except Exception:
                pass

    def current_time(self) -> float:
        with self._mutex:
            return (self._start_frames + self._pos_frames) / float(self._sr or 1)

    def ended(self) -> bool:
        with self._mutex:
            flag = self._ended_flag
            self._ended_flag = False
            return flag

    def close(self):
        self.stop()
        try:
            if self._sf: self._sf.close()
        except Exception:
            pass
        self._sf = None
        self._close_stream_only()

# --------------------------- Memory helpers ---------------------------
def _torch_sync_and_free():
    try:
        import torch  # local import if present
        if hasattr(torch, "mps"):
            try: torch.mps.synchronize()
            except Exception: pass
        if hasattr(torch, "cuda"):
            try: torch.cuda.synchronize()
            except Exception: pass
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
    except Exception:
        pass

def _mlx_sync_and_free():
    try:
        import mlx.core as mx  # type: ignore
        if hasattr(mx, "clear_cache") and callable(getattr(mx, "clear_cache")):
            mx.clear_cache()
        else:
            if hasattr(mx, "metal") and hasattr(mx.metal, "synchronize"):
                try: mx.metal.synchronize()
                except Exception: pass
            for attr in ("empty_cache", "clear_cache", "reset_alloc", "reset_cache"):
                obj = getattr(getattr(mx, "metal", mx), attr, None)
                if callable(obj):
                    try: obj()
                    except Exception: pass
                    break
    except Exception:
        pass

def _free_all_memory(hint=""):
    gc.collect()
    _torch_sync_and_free()
    _mlx_sync_and_free()
    importlib.invalidate_caches()
    if hint:
        print(f"[Memory] GC + cache clear • {hint}")

# Peak RAM monitor (main + children)
class _MemPeak:
    def __init__(self, label: str = ""):
        self.label = label
        self._peak_bytes = 0
        self._evt = threading.Event()
        self._t = None
        try:
            import psutil  # noqa
            self._ok = True
        except Exception:
            self._ok = False

    def _sample_once(self):
        if not self._ok: return
        import psutil
        try:
            p = psutil.Process(os.getpid())
            rss = 0
            try:
                rss += p.memory_info().rss
            except Exception:
                pass
            for ch in p.children(recursive=True):
                try:
                    rss += ch.memory_info().rss
                except Exception:
                    pass
            if rss > self._peak_bytes:
                self._peak_bytes = rss
        except Exception:
            pass

    def _run(self):
        # sample immediately, then every 10s
        self._sample_once()
        while not self._evt.wait(10.0):
            self._sample_once()
        # final sample at stop
        self._sample_once()

    def start(self):
        if not self._ok: return
        self._evt.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> float | None:
        if not self._ok: return None
        try:
            self._evt.set()
            if self._t:
                self._t.join()
        except Exception:
            pass
        return self._peak_bytes / 1048576.0  # MB

# --------------------------- UI language list ---------------------------
LANG_MAP = {
    "Auto-detect":"auto","English":"en","Afrikaans (South Africa)":"af","Arabic (Egypt)":"ar","Armenian (Armenia)":"hy",
    "Azerbaijani (Azerbaijan)":"az","Belarusian (Belarus)":"be","Bosnian (Bosnia and Herzegovina)":"bs","Bulgarian (Bulgaria)":"bg",
    "Catalan (Spain)":"ca","Chinese (Mandarin, Simplified)":"zh","Croatian (Croatia)":"hr","Czech (Czech)":"cs","Danish (Denmark)":"da",
    "Dutch (Netherlands)":"nl","Estonian (Estonia)":"et","Finnish (Finland)":"fi","French (France)":"fr","Galician (Galician)":"gl",
    "German (Germany)":"de","Greek (Greece)":"el","Hebrew (Israel)":"he","Hindi (India)":"hi","Hungarian (Hungary)":"hu","Icelandic (Iceland)":"is",
    "Indonesian (Indonesia)":"id","Italian (Italy)":"it","Japanese (Japan)":"ja","Kannada (India)":"kn","Kazakh (Kazakhstan)":"kk","Korean (Korea)":"ko",
    "Latvian (Latvia)":"lv","Lithuanian (Lithuania)":"lt","Macedonian (North Macedonia)":"mk","Malay (Malaysia)":"ms","Marathi (India)":"mr",
    "Māori (New Zealand)":"mi","Nepali (Nepal)":"ne","Norwegian (Bokmål, Norway)":"no","Persian (Iran)":"fa","Polish (Poland)":"pl","Portuguese (Brazil)":"pt",
    "Romanian (Romania)":"ro","Russian (Russia)":"ru","Serbian (Serbia)":"sr","Slovak (Slovakia)":"sk","Slovenian (Slovenia)":"sl","Spanish (Spain)":"es",
    "Swahili (Kenya)":"sw","Swedish (Sweden)":"sv","Filipino (Philippines)":"tl","Tamil (India)":"ta","Thai (Thailand)":"th","Turkish (Türkiye)":"tr",
    "Ukrainian (Ukraine)":"uk","Urdu (Pakistan)":"ur","Vietnamese (Vietnam)":"vi","Welsh (United Kingdom)":"cy",
}
LANG_NAMES = list(LANG_MAP.keys())

# ===== LLM languages (target) =====
def _llm_langs():
    base = ["English (UK)", "English (US)"]
    rest = [n for n in LANG_NAMES if not n.startswith("English")]
    return base + rest
LLM_LANG_NAMES = _llm_langs()

def normalise_llm_lang(name: str):
    if name == "English (UK)": return "English (UK)", "Use BRITISH English spelling."
    if name == "English (US)": return "English (US)", "Use US English spelling."
    return name, ""

# --------------------------- ALIGN/DIAR HELPERS (shared) ---------------------------
def _segments_normalize_for_align(segments):
    def _get_time(d, long_key, short_key):
        if long_key in d: return float(d[long_key])
        if short_key in d: return float(d[short_key])
        return None
    fixed_segments = []
    for seg in (segments or []):
        seg = dict(seg)
        s_start = _get_time(seg, "start", "s")
        s_end = _get_time(seg, "end", "e")
        words = list(seg.get("words") or [])
        raw_starts = [ _get_time(w,"start","s") for w in words ]
        if s_start is None:
            s_start = next((st for st in raw_starts if st is not None), 0.0)
        if s_end is None:
            last_end = None
            for w in words:
                we = _get_time(w, "end", "e")
                if we is not None: last_end = we
            if last_end is None:
                last_start = next((st for st in reversed(raw_starts) if st is not None), s_start or 0.0)
                last_end = (last_start if last_start is not None else 0.0) + 0.02
            s_end = last_end
        if s_end < s_start: s_end = s_start
        seg["start"] = s_start; seg["end"] = s_end; seg["s"] = s_start; seg["e"] = s_end
        fixed_words = []
        n = len(words)
        for i, w in enumerate(words):
            w = dict(w)
            ws = _get_time(w, "start", "s")
            we = _get_time(w, "end", "e")
            if ws is None:
                ws = _get_time(fixed_words[-1], "end", "e") if fixed_words else s_start
            if we is None:
                nxt = None
                for j in range(i+1, n):
                    if raw_starts[j] is not None:
                        nxt = raw_starts[j]; break
                if nxt is None:
                    for j in range(i+1, n):
                        nxt = _get_time(words[j], "start", "s")
                        if nxt is not None: break
                we = nxt if nxt is not None else s_end
            if we < ws: we = ws
            w["start"] = ws; w["end"] = we; w["s"] = ws; w["e"] = we
            if "word" not in w and "text" in w: w["word"] = w["text"]
            fixed_words.append(w)
        seg["words"] = fixed_words
        fixed_segments.append(seg)
    return fixed_segments

def _segments_sanitize_word_ts(segments):
    out = []
    for seg in (segments or []):
        seg = dict(seg)
        words = seg.get("words") or []
        fixed_words = []
        last_t = float(seg.get("start", 0.0)) if seg.get("start") is not None else 0.0
        for w in words:
            if not isinstance(w, dict): continue
            ws = w.get("start", w.get("s")); we = w.get("end",   w.get("e"))
            if ws is None and "ts" in w: ws = w["ts"]
            if we is None and "te" in w: we = w["te"]
            try:
                ws = float(ws) if ws is not None else None
                we = float(we) if we is not None else None
            except Exception:
                ws = None; we = None
            if ws is None and we is None: continue
            if ws is None: ws = last_t
            if we is None: we = ws
            if we < ws: we = ws
            last_t = we
            w2 = dict(w); w2["start"] = ws; w2["end"] = we; w2["s"] = ws; w2["e"] = we
            if "word" not in w2 and "text" in w2: w2["word"] = w2["text"]
            fixed_words.append(w2)
        sst = seg.get("start", seg.get("s"))
        eet = seg.get("end", seg.get("e"))
        try: sst = float(sst) if sst is not None else (fixed_words[0]["start"] if fixed_words else 0.0)
        except Exception: sst = 0.0
        try: eet = float(eet) if eet is not None else (fixed_words[-1]["end"] if fixed_words else sst)
        except Exception: eet = sst
        if eet < sst: eet = sst
        seg["start"] = sst; seg["end"] = eet; seg["s"] = sst; seg["e"] = eet
        seg["words"] = fixed_words
        out.append(seg)
    return out

def _segment_level_assign_by_overlap(annotation, segments):
    diar_rows = []
    for (segment, _, speaker) in annotation.itertracks(yield_label=True):
        diar_rows.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
    for s in (segments or []):
        try:
            ss = float(s.get("start", s.get("s", 0.0)) or 0.0)
            se = float(s.get("end", s.get("e", ss)) or ss)
        except Exception:
            ss, se = 0.0, 0.0
        if se < ss: se = ss
        best_spk = None; best_inter = 0.0
        for d in diar_rows:
            inter = max(0.0, min(d["end"], se) - max(d["start"], ss))
            if inter > best_inter:
                best_inter = inter; best_spk = d["speaker"]
        if best_spk is None:
            best_spk = s.get("speaker") or "SPEAKER_00"
        s["speaker"] = best_spk
        for w in (s.get("words") or []):
            if not w.get("speaker"):
                w["speaker"] = best_spk
    return segments

def _smooth_word_speakers(segments, min_run_dur: float = 0.8) -> tuple[list, int]:
    """Reduce wrong-speaker 'islands' in word-level assignment by merging short runs to neighbors."""
    flips_total = 0
    for s in (segments or []):
        words = s.get("words") or []
        if len(words) < 2:
            continue

        def _build_runs():
            runs = []
            cur = None; start = 0
            for i, w in enumerate(words):
                spk = w.get("speaker")
                if spk != cur:
                    if cur is not None:
                        runs.append((cur, start, i-1))
                    cur = spk; start = i
            if cur is not None:
                runs.append((cur, start, len(words)-1))
            return runs

        def _run_dur(a, b):
            try:
                st = float(words[a].get("start", 0.0)); en = float(words[b].get("end", st))
            except Exception:
                return 0.0
            return max(0.0, en - st)

        runs = _build_runs()
        for r in range(1, len(runs)-1):
            spk, a, b = runs[r]
            dur = _run_dur(a, b)
            if dur < min_run_dur:
                left_spk = runs[r-1][0]; right_spk = runs[r+1][0]
                if left_spk == right_spk and left_spk != spk:
                    for i in range(a, b+1):
                        if words[i].get("speaker") != left_spk:
                            words[i]["speaker"] = left_spk; flips_total += 1

        runs = _build_runs()
        if len(runs) >= 2:
            spk,a,b = runs[0]
            if _run_dur(a,b) < min_run_dur:
                to_spk = runs[1][0]
                if to_spk != spk:
                    for i in range(a,b+1):
                        words[i]["speaker"] = to_spk; flips_total += 1
        runs = _build_runs()
        if len(runs) >= 2:
            spk,a,b = runs[-1]
            if _run_dur(a,b) < min_run_dur:
                to_spk = runs[-2][0]
                if to_spk != spk:
                    for i in range(a,b+1):
                        words[i]["speaker"] = to_spk; flips_total += 1

        totals={}
        for w in words:
            spk=w.get("speaker") or "SPEAKER_00"
            try:
                ws=float(w.get("start",0.0)); we=float(w.get("end",ws))
            except Exception:
                ws=0.0; we=0.0
            totals[spk]=totals.get(spk,0.0) + max(0.0,we-ws)
        if totals:
            s["speaker"]=max(totals.items(), key=lambda kv: kv[1])[0]

    return segments, flips_total

# --------------------------- CHILD PROCESS STAGES ---------------------------
def _ffmpeg_decode_f32_mono_16k_child(src_path: str):
    ff = ffmpeg_path()
    cmd = [ff, "-v", "error", "-nostdin", "-i", src_path, "-map", "0:a:0", "-vn", "-sn", "-dn",
           "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {err.decode('utf-8','ignore')}")
    arr = np.frombuffer(out, dtype=np.float32).copy()
    return arr, 16000

def _proc_transcribe_entry(job: dict, out_q):
    import traceback, time
    try:
        import mlx_whisper
        media_path = job["media_path"]
        whisper_dir = job["whisper_dir"]
        lang_code = job["lang_code"]
        t0 = time.perf_counter()
        out_q.put(("log", "Transcribe: start"))
        if lang_code == "auto":
            result = mlx_whisper.transcribe(media_path, path_or_hf_repo=whisper_dir)
            detected = result.get("language","en")
            lang_for_align = detected.split("-")[0].lower()
            out_q.put(("log", f"Language detected: {detected}"))
        else:
            result = mlx_whisper.transcribe(media_path, path_or_hf_repo=whisper_dir, language=lang_code)
            lang_for_align = lang_code
        segments = result.get("segments", [])
        out_q.put(("log", f"Segments: {len(segments)} • time={time.perf_counter()-t0:.2f}s"))
        out_q.put(("segments", {"segments": segments, "lang_for_align": lang_for_align}))
    except Exception:
        out_q.put(("error", traceback.format_exc()))

def _proc_align_entry(job: dict, out_q):
    import traceback, time
    try:
        import whisperx
        media_path = job["media_path"]
        segments = job["segments"]
        align_dir = job["align_dir"]

        t0 = time.perf_counter()
        out_q.put(("log", "Align: decode audio (mono16k)"))
        wav_f32, sr = _ffmpeg_decode_f32_mono_16k_child(media_path)

        out_q.put(("log", f"Align: loading model"))
        model_a, metadata = whisperx.load_align_model(language_code="en", device="mps", model_name=str(align_dir))

        out_q.put(("log", f"Align: run"))
        aligned = whisperx.align(segments, model_a, metadata, wav_f32, device="mps")
        raw_segments = aligned["segments"]

        nseg0 = len(raw_segments)
        nwords0 = sum(len(s.get("words") or []) for s in raw_segments)
        nwords_ts0 = sum(
            1 for s in raw_segments for w in (s.get("words") or [])
            if isinstance(w, dict) and isinstance(w.get("start"), (int,float)) and isinstance(w.get("end"), (int,float))
        )
        out_q.put(("log", f"Allignment: raw segs={nseg0}, words={nwords0}, words_with_ts={nwords_ts0}"))

        segments2 = _segments_sanitize_word_ts(_segments_normalize_for_align(raw_segments))
        nseg = len(segments2)
        nwords = sum(len(s.get("words") or []) for s in segments2)
        nwords_ts = sum(
            1 for s in segments2 for w in (s.get("words") or [])
            if isinstance(w, dict) and isinstance(w.get("start"), (int,float)) and isinstance(w.get("end"), (int,float))
        )
        out_q.put(("log", f"Normalised: words_ts={nwords_ts}/{nwords} across {nseg} segments"))
        out_q.put(("log", f"Alignment: done in {time.perf_counter()-t0:.2f}s"))
        out_q.put(("segments", segments2))
    except Exception:
        out_q.put(("error", traceback.format_exc()))

def _proc_diar_entry(job: dict, out_q):
    import traceback, tempfile, time
    try:
        # (unchanged; remains in second half)
        pass
    except Exception:
        out_q.put(("error", traceback.format_exc()))

# --------------------------- GUI ---------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1260x840"); self.minsize(1120, 720)

        self.media_files=[]; self.output_files=[]; self.msg_q=queue.Queue()

        # fixed state
        self.lang_var  = tk.StringVar(value="English")
        self.mode_var  = tk.StringVar(value="transcribe")
        self.align_var = tk.BooleanVar(value=True)   # English only
        self.diar_var  = tk.BooleanVar(value=True)
        self.maxdur_var = tk.StringVar(value="6.0")
        self.num_speakers_str = tk.StringVar(value="auto")

        self.save_json_var = tk.BooleanVar(value=False)
        self.save_txt_var  = tk.BooleanVar(value=True)
        self.save_csv_var  = tk.BooleanVar(value=False)
        self.save_srt_var  = tk.BooleanVar(value=False)
        self.save_vtt_var  = tk.BooleanVar(value=False)

        # LLM state (fixed model, auto-load/unload)
        self.llm_model = None; self.llm_tok = None
        self.llm_max_new = tk.StringVar(value="30000")
        self.llm_temp    = tk.StringVar(value="0.3")
        self.llm_top_p   = tk.StringVar(value="0.9")
        self.llm_top_k   = tk.StringVar(value="50")
        self.llm_rep_pen = tk.StringVar(value="1.12")
        self.llm_no_repeat = tk.StringVar(value="3")
        self.llm_target_lang_var = tk.StringVar(value="English (UK)")
        self.llm_thinking_var = tk.BooleanVar(value=False)  # NEW: Thinking toggle
        self._loaded_model_id = None; self._llm_loading=False
        self._llm_lock = threading.Lock()

        # Output/editor/player state
        self._current_preview_path = None
        self._preview_prog_update = False
        self._save_after_id = None
        self._segments = []
        self._active_seg_idx = None
        self._current_media_path = None
        self._play_source = None

        # Preview font controls
        self._preview_font_family = "Consolas"
        self._preview_font_size = tk.IntVar(value=10)
        self._preview_min_font = 8
        self._preview_max_font = 48

        # Edit/Speed UI state
        self._edit_btn_txt = tk.StringVar(value="Edit transcript…")
        self._speed_var = tk.StringVar(value="1.0")
        self._current_speed = 1.0

        # Player (sounddevice)
        if not HAS_SD:
            _dep_missing("sounddevice soundfile", "Audio playback")
        self._sd = _SDStreamer() if HAS_SD else None
        self._player_tick_id = None
        self._paused = False

        self._build_layout()
        self.after(120, self._poll_queue)
        self._update_align_availability()
        self.after(0, self._apply_min_sizes)  # enforce min heights after layout

        # Hotkeys
        self.bind_all("<Control-space>", self._on_ctrl_space)
        self.bind_all("<Control-Key-space>", self._on_ctrl_space)
        self.bind_all("<Control-Shift-space>", self._on_ctrl_space)

        # Zoom shortcuts
        self.bind_all("<Control-=>", lambda e: self._increase_preview_font())
        self.bind_all("<Control-plus>", lambda e: self._increase_preview_font())
        self.bind_all("<Control-minus>", lambda e: self._decrease_preview_font())
        self.bind_all("<Control-0>", lambda e: self._reset_preview_font())

    # --------------------------- About window ---------------------------
    def _open_about_window(self):
        win = tk.Toplevel(self)
        win.title("About")
        win.geometry("820x620")
        win.transient(self)
        win.grab_set()

        top = ttk.Frame(win); top.pack(fill="x", padx=8, pady=6)
        txt = tk.Text(win, wrap="word", font=("Consolas", 10), state="disabled")
        txt.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # simple hyperlink system for markdown-like [label](path)
        link_map = {}

        def _render_md(path: Path):
            try:
                data = path.read_text(encoding="utf-8")
            except Exception as e:
                data = f"Could not read file:\n{path}\n\n{e}"
            txt.config("normal")
            txt.delete("1.0", "end")
            # detect [label](target)
            patt = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
            pos = 0
            idx = 0
            for m in patt.finditer(data):
                before = data[pos:m.start()]
                txt.insert("end", before)
                label = m.group(1)
                target = m.group(2)
                start_idx = txt.index("end-1c")
                txt.insert("end", label)
                end_idx = txt.index("end-1c")
                tag = f"link_{idx}"
                link_map[tag] = target
                txt.tag_add(tag, start_idx, end_idx)
                txt.tag_config(tag, foreground="#0b61a4", underline=1)
                def _make_cb(tg):
                    return lambda e: _open_target(link_map.get(tg,""))
                txt.tag_bind(tag, "<Button-1>", _make_cb(tag))
                idx += 1
                pos = m.end()
            txt.insert("end", data[pos:])
            txt.config("disabled")

        def _open_target(target: str):
            # Try to resolve relative to app root
            base = _resource_base()
            p = (base / target).resolve()
            p2 = (CONTENT / target).resolve()
            if p.exists():
                _render_md(p)
            elif p2.exists():
                _render_md(p2)
            else:
                messagebox.showinfo("Link", f"External links are disabled in offline mode.\n\nTarget: {target}")

        # buttons
        def _load_about():
            link_map.clear()
            _render_md(CONTENT / "licences" / "README.md")
        def _load_licence():
            link_map.clear()
            _render_md(CONTENT / "licences" / "LICENCE.txt")
        def _load_thirdparty():
            link_map.clear()
            _render_md(CONTENT / "licences" / "Third-Party-Licences.txt")

        ttk.Button(top, text="About", command=_load_about).pack(side="left", padx=(0,8))
        ttk.Button(top, text="Licence", command=_load_licence).pack(side="left", padx=8)
        ttk.Button(top, text="Third Party Licences", command=_load_thirdparty).pack(side="left", padx=8)

        # context menu for text
        cm = tk.Menu(txt, tearoff=False)
        cm.add_command(label="Copy", command=lambda: txt.event_generate("<<Copy>>"))
        txt.bind("<Button-3>", lambda e: (txt.focus_set(), cm.tk_popup(e.x_root, e.y_root)))
        txt.bind("<Button-2>", lambda e: (txt.focus_set(), cm.tk_popup(e.x_root, e.y_root)))

        _load_about()

    def _on_ctrl_space(self, _evt=None):
        self._toggle_pause()

    def _update_align_availability(self, *_):
        ok = (self.lang_var.get() == "English")
        self.align_var.set(bool(ok))
        try:
            for w in getattr(self, "_align_widgets", []):
                w.state(["!disabled" if ok else "disabled"])
        except Exception: pass

    def _build_layout(self):
        pad={'padx': 6, 'pady': 4}
        root = ttk.Frame(self); root.pack(fill="both", expand=True)

        # --- Controls row
        ctrl1 = ttk.Frame(root); ctrl1.pack(fill="x", **pad)
        ttk.Label(ctrl1, text="Language:").pack(side="left", padx=(0,4))
        lang_cb = ttk.Combobox(ctrl1, textvariable=self.lang_var, width=28, values=LANG_NAMES, state="readonly")
        lang_cb.pack(side="left", padx=5); lang_cb.bind("<<ComboboxSelected>>", self._update_align_availability)
        ttk.Label(ctrl1, text="Mode:").pack(side="left", padx=(16,4))
        ttk.Radiobutton(ctrl1, text="Transcribe", variable=self.mode_var, value="transcribe", command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(ctrl1, text="Subtitles",  variable=self.mode_var, value="subs",        command=self._on_mode_change).pack(side="left", padx=5)
        ttk.Button(ctrl1, text="About…", command=self._open_about_window).pack(side="right")

        ctrl2 = ttk.Frame(root); ctrl2.pack(fill="x", **pad)
        align_chk = ttk.Checkbutton(ctrl2, text="Alignment (English only)", variable=self.align_var)
        align_chk.pack(side="left", padx=6); self._align_widgets = [align_chk]
        ttk.Label(ctrl2, text="Speakers:").pack(side="left", padx=(8,2))
        ttk.Entry(ctrl2, textvariable=self.num_speakers_str, width=6).pack(side="left", padx=(0,8))
        self.diar_chk = ttk.Checkbutton(ctrl2, text="Diarisation", variable=self.diar_var)
        self.diar_chk.pack(side="left", padx=6)
        ttk.Label(ctrl2, text="Max subtitle dur (s):").pack(side="left", padx=(12,4))
        ttk.Entry(ctrl2, textvariable=self.maxdur_var, width=6).pack(side="left")

        row_save = ttk.Frame(root); row_save.pack(fill="x", **pad)
        ttk.Label(row_save, text="Auto-save:").pack(side="left")
        ttk.Checkbutton(row_save, text="JSON", variable=self.save_json_var).pack(side="left")
        ttk.Checkbutton(row_save, text="TXT",  variable=self.save_txt_var ).pack(side="left")
        ttk.Checkbutton(row_save, text="CSV",  variable=self.save_csv_var ).pack(side="left")
        ttk.Checkbutton(row_save, text="SRT",  variable=self.save_srt_var ).pack(side="left")
        ttk.Checkbutton(row_save, text="VTT",  variable=self.save_vtt_var ).pack(side="left")

        # --- Top row: Media inputs & Output files (ABOVE the editor)
        lists_row = ttk.Panedwindow(root, orient="horizontal"); lists_row.pack(fill="x", **pad)

        left_small = ttk.Labelframe(lists_row, text="Media files (inputs)")
        self.media_list = tk.Listbox(left_small, height=6, selectmode="extended"); self.media_list.pack(fill="x", expand=True)
        row_media_buttons = ttk.Frame(left_small); row_media_buttons.pack(fill="x")
        ttk.Button(row_media_buttons, text="Add…", command=self._add_media).pack(side="left", padx=3)
        ttk.Button(row_media_buttons, text="Remove", command=self._remove_media).pack(side="left", padx=3)
        ttk.Button(row_media_buttons, text="Clear", command=self._clear_media).pack(side="left", padx=3)
        ttk.Button(row_media_buttons, text="Select all", command=lambda: self.media_list.selection_set(0, "end")).pack(side="left", padx=6)
        ttk.Button(row_media_buttons, text="Run Batch ▶", command=self._run_batch).pack(side="right", padx=3)
        lists_row.add(left_small, weight=1)

        right_small = ttk.Labelframe(lists_row, text="Output files (results)")
        self.output_list = tk.Listbox(right_small, height=6, selectmode="extended"); self.output_list.pack(fill="x", expand=True)
        self.output_list.bind("<<ListboxSelect>>", self._preview_selected)
        row_out_buttons = ttk.Frame(right_small); row_out_buttons.pack(fill="x")
        ttk.Button(row_out_buttons, text="Add…", command=self._add_outputs).pack(side="left", padx=3)
        ttk.Button(row_out_buttons, text="Remove", command=self._remove_outputs).pack(side="left", padx=3)
        ttk.Button(row_out_buttons, text="Clear", command=self._clear_outputs).pack(side="left", padx=3)
        ttk.Button(row_out_buttons, text="Select all", command=lambda: self.output_list.selection_set(0, "end")).pack(side="left", padx=6)
        lists_row.add(right_small, weight=1)

        # =================== Center panes: Output (top) + Log (bottom) ===================
        mid = ttk.Panedwindow(root, orient="vertical")
        mid.pack(fill="both", expand=True, **pad)
        self._mid = mid  # store for minsize updates

        # --- Output editor (merged LLM tools + Output)
        output_frame = ttk.Labelframe(mid, text="Output")
        self._output_frame = output_frame
        output_frame.columnconfigure(0, weight=1)

        # Row 0: LLM params
        bar1 = ttk.Frame(output_frame); bar1.grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        ttk.Label(bar1, text="Target language:").pack(side="left")
        ttk.Combobox(bar1, textvariable=self.llm_target_lang_var, width=22, values=LLM_LANG_NAMES, state="readonly").pack(side="left", padx=(2,10))
        ttk.Button(bar1, text="Model parameters…", command=self._open_llm_params_dialog).pack(side="left", padx=4)
        ttk.Checkbutton(bar1, text="Thinking", variable=self.llm_thinking_var).pack(side="left", padx=10)

        # Row 1: LLM actions
        bar3 = ttk.Frame(output_frame); bar3.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        self._bar3_widgets = [
            ttk.Button(bar3, text="Translate",  command=self._llm_translate_selected),
            ttk.Button(bar3, text="Summarise",  command=self._llm_summarise_selected),
            ttk.Button(bar3, text="Correct text", command=self._llm_correct_selected),
            ttk.Button(bar3, text="Custom Prompt ▶", command=self._llm_custom_on_files),
        ]
        for i,w in enumerate(self._bar3_widgets): w.grid(row=0, column=i, padx=6, pady=2, sticky="w")
        bar3.bind("<Configure>", lambda e: self._reflow_bar3(bar3, e.width))

        # Row 2: Zoom + Edit toggle
        zoom_bar = ttk.Frame(output_frame); zoom_bar.grid(row=2, column=0, sticky="ew", padx=4, pady=(0,2))
        ttk.Label(zoom_bar, text="Text size:").pack(side="left")
        ttk.Button(zoom_bar, text="A−", width=3, command=self._decrease_preview_font).pack(side="left", padx=(6,2))
        ttk.Button(zoom_bar, text="A+", width=3, command=self._increase_preview_font).pack(side="left", padx=2)
        ttk.Button(zoom_bar, text="Reset", command=self._reset_preview_font).pack(side="left", padx=6)
        ttk.Separator(zoom_bar, orient="vertical").pack(side="left", padx=10, fill="y")
        ttk.Button(zoom_bar, textvariable=self._edit_btn_txt, command=self._toggle_edit_mode).pack(side="left", padx=(6,0))

        # Row 3: Preview (grows)
        self.preview = tk.Text(
            output_frame,
            wrap="word",
            font=(self._preview_font_family, self._preview_font_size.get()),
            undo=True,
            autoseparators=True,
            maxundo=-1
        )
        self.preview.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0,0))
        output_frame.rowconfigure(3, weight=1)  # only preview grows
        self.preview.tag_configure("seg_active", background="#FFF59D")
        self.preview.bind("<<Modified>>", self._on_preview_modified)
        self.preview.bind("<Double-Button-1>", self._on_preview_double_click)

        # right-click menu
        self._ctx = tk.Menu(self.preview, tearoff=False)
        self._ctx.add_command(label="Cut", command=lambda: self.preview.event_generate("<<Cut>>"))
        self._ctx.add_command(label="Copy", command=lambda: self.preview.event_generate("<<Copy>>"))
        self._ctx.add_command(label="Paste", command=lambda: self.preview.event_generate("<<Paste>>"))
        self._ctx.add_separator()
        self._ctx.add_command(label="Select All", command=lambda: (self.preview.tag_add("sel","1.0","end-1c"), self.preview.focus_set()))
        self.preview.bind("<Button-3>", lambda e: (self.preview.focus_set(), self._ctx.tk_popup(e.x_root, e.y_root)))
        self.preview.bind("<Button-2>", lambda e: (self.preview.focus_set(), self._ctx.tk_popup(e.x_root, e.y_root)))
        # Default to VIEW-ONLY
        self.preview.config("disabled")

        # Row 4: Player bar (with speed on SAME LINE)
        player_bar = ttk.Frame(output_frame); player_bar.grid(row=4, column=0, sticky="ew", padx=4, pady=(4,0))
        # Speed (left)
        ttk.Label(player_bar, text="Speed ×").pack(side="left")
        sp_ent = ttk.Entry(player_bar, textvariable=self._speed_var, width=5, justify="center")
        sp_ent.pack(side="left", padx=(4,6))
        sp_ent.bind("<Return>", lambda e: self._apply_speed_change())
        sp_ent.bind("<FocusOut>", lambda e: self._apply_speed_change())
        ttk.Button(player_bar, text="Apply", command=self._apply_speed_change).pack(side="left", padx=(0,8))
        # Transport (right)
        ttk.Button(player_bar, text="Stop ⏹", command=self._stop_playback).pack(side="right", padx=4)
        self._pause_btn_txt = tk.StringVar(value="Pause ⏸")
        ttk.Button(player_bar, textvariable=self._pause_btn_txt, command=self._toggle_pause).pack(side="right", padx=4)
        ttk.Button(player_bar, text="Play All ▶", command=self._play_all).pack(side="right", padx=4)

        # --- Log (in lower pane)
        log_frame = ttk.Labelframe(mid, text="Log")
        self._log_frame = log_frame
        self.log = tk.Text(log_frame, wrap="word", font=("Consolas", 9), height=6)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        # Add panes with initial weights; minsize set in _apply_min_sizes()
        mid.add(output_frame, weight=3)
        mid.add(log_frame, weight=1)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", padx=6, pady=3)

        self._on_mode_change()

    # ---- min sizes for panes & rows so bars don't vanish on resize ----
    def _apply_min_sizes(self):
        try:
            # preview: ≥ 8 lines
            pf = tkfont.Font(font=(self._preview_font_family, self._preview_font_size.get()))
            line = max(1, pf.metrics("linespace"))
            preview_min = line * 8 + 8
            # keep the preview row above this min height
            try:
                self._output_frame.rowconfigure(3, minsize=preview_min)
            except Exception:
                pass
            # panes min size (approx add toolbar heights)
            out_min = preview_min + 140
            self._mid.paneconfigure(self._output_frame, minsize=out_min)
        except Exception:
            pass
        try:
            # log: ≥ 2 lines
            lf = tkfont.Font(font=("Consolas", 9))
            log_min = lf.metrics("linespace") * 2 + 16
            self._mid.paneconfigure(self._log_frame, minsize=log_min)
        except Exception:
            pass

    def _open_llm_params_dialog(self):
        win = tk.Toplevel(self); win.title("Model parameters"); win.transient(self); win.grab_set()
        frm = ttk.Frame(win, padding=10); frm.pack(fill="both", expand=True)
        rows = [
            ("Max new tokens", self.llm_max_new),
            ("Temperature",    self.llm_temp),
            ("Top-p",          self.llm_top_p),
            ("Top-k",          self.llm_top_k),
            ("Rep. penalty",   self.llm_rep_pen),
            ("No-repeat n-gram", self.llm_no_repeat),
        ]
        for i,(label,var) in enumerate(rows):
            ttk.Label(frm, text=label+":").grid(row=i, column=0, sticky="e", padx=6, pady=4)
            ttk.Entry(frm, textvariable=var, width=12).grid(row=i, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(frm, text="Close", command=win.destroy).grid(row=len(rows), column=0, columnspan=2, pady=(10,0))

    def _reflow_bar3(self, frame, width):
        frame.update_idletasks()
        per = 140
        cols = max(1, width // per)
        for i, w in enumerate(self._bar3_widgets):
            r = i // cols; c = i % cols; w.grid_configure(row=r, column=c)

    # --- UI helpers ---
    def _on_mode_change(self):
        if self.mode_var.get() == "subs":
            self.diar_var.set(False); self.diar_chk.state(["disabled"])
            self.save_txt_var.set(False); self.save_srt_var.set(True); self.save_vtt_var.set(False)
            self.save_csv_var.set(False); self.save_json_var.set(False)
        else:
            self.diar_chk.state(["!disabled"])
            self.save_txt_var.set(True); self.save_srt_var.set(False); self.save_vtt_var.set(False)
            self.save_csv_var.set(False); self.save_json_var.set(False)

    def _dest_base(self, audio_path):
        from os.path import splitext
        return splitext(audio_path)[0]

    # --- lists ---
    def _add_media(self):
        paths = filedialog.askopenfilenames(
            title="Add media files",
            filetypes=[("Audio/Video","*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac;*.wma;*.webm;*.mp4;*.mkv;*.mov;*.avi")]
        )
        for p in paths:
            if p not in self.media_files:
                self.media_files.append(p); self.media_list.insert("end", p)

    def _remove_media(self):
        for idx in reversed(self.media_list.curselection()):
            p = self.media_list.get(idx); self.media_list.delete(idx)
        self.media_files = [x for x in self.media_files if x in self.media_list.get(0,"end")]

    def _clear_media(self):
        self.media_list.delete(0,"end"); self.media_files.clear()

    def _add_outputs(self):
        paths = filedialog.askopenfilenames(
            title="Add output files",
            filetypes=[("Text/JSON/Subtitles","*.json;*.txt;*.csv;*.srt;*.vtt")]
        )
        for p in paths:
            if p not in self.output_files: self.output_files.append(p); self.output_list.insert("end", p)

    def _remove_outputs(self):
        for idx in reversed(self.output_list.curselection()):
            p = self.output_list.get(idx); self.output_list.delete(idx)
        self.output_files = [x for x in self.output_files if x in self.output_list.get(0,"end")]

    def _clear_outputs(self):
        self.output_list.delete(0,"end"); self.output_files.clear()

    def _preview_selected(self, _evt=None):
        sel = self.output_list.curselection()
        if not sel: return
        path = self.output_list.get(sel[0])
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self._stop_playback()
            self._current_preview_path = path
            # Always return to VIEW-ONLY on selection
            self._edit_btn_txt.set("Edit transcript…")
            self.preview.config("disabled")
            self._set_preview_text(content)
            self._parse_preview_segments()
            self._auto_add_and_bind_media_for_output(path)
        except Exception as e:
            self._post("log", f"Failed to load {path}: {e}")

    def _set_preview_text(self, text: str):
        self._preview_prog_update = True
        try:
            prev_state = self.preview.cget("state")
            self.preview.config("normal")
            self.preview.delete("1.0","end")
            self.preview.insert("1.0", text)
            self.preview.edit_modified(False)
            # restore previous state (default view-only)
            self.preview.config(prev_state)
        finally:
            self._preview_prog_update = False

    # ---- preview font helpers ----
    def _apply_preview_font(self):
        try:
            self.preview.configure(font=(self._preview_font_family, self._preview_font_size.get()))
            self._apply_min_sizes()
        except Exception as e:
            self._post("log", f"Font apply failed: {e}")

    def _increase_preview_font(self):
        s = min(self._preview_font_size.get() + 1, self._preview_max_font)
        self._preview_font_size.set(s)
        self._apply_preview_font()

    def _decrease_preview_font(self):
        s = max(self._preview_font_size.get() - 1, self._preview_min_font)
        self._preview_font_size.set(s)
        self._apply_preview_font()

    def _reset_preview_font(self):
        self._preview_font_size.set(10)
        self._apply_preview_font()

#BREAK PART1
    def _post(self, kind, payload): self.msg_q.put((kind, payload))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind=="log":
                    ts=time.strftime("%H:%M:%S"); self.log.insert("end", f"[{ts}] {payload}\n"); self.log.see("end")
                elif kind=="status":
                    self.status_var.set(str(payload))
                elif kind=="preview":
                    # force view-only when preview text is replaced programmatically
                    self._edit_btn_txt.set("Edit transcript…")
                    self.preview.config("disabled")
                    self._set_preview_text(str(payload))
                    try: self._parse_preview_segments()
                    except Exception: pass
                    self.preview.see("end")
                elif kind=="preview_from_file":
                    p=str(payload)
                    try:
                        with open(p,"r",encoding="utf-8") as f: txt=f.read()
                        self._current_preview_path = p
                        # force view-only on file load
                        self._edit_btn_txt.set("Edit transcript…")
                        self.preview.config("disabled")
                        self._set_preview_text(txt)
                        self._parse_preview_segments()
                        self._auto_add_and_bind_media_for_output(p)
                    except Exception as e:
                        self._post("log", f"Failed to open preview file: {e}")
                elif kind=="add_output":
                    p=str(payload)
                    if p not in self.output_files:
                        self.output_files.append(p); self.output_list.insert("end", p)
                    try: self._auto_add_and_bind_media_for_output(p)
                    except Exception: pass
        except queue.Empty: pass
        self.after(120, self._poll_queue)

    # --------------------------- Edit mode: only .edited* can be modified ---------------------------
    def _toggle_edit_mode(self):
        if not self._current_preview_path:
            messagebox.showinfo("No file", "Select a transcript in Output files first.")
            return

        path = Path(self._current_preview_path)
        is_edited = ".edited" in path.stem

        # If already editing an edited* file → switch back to view-only
        if self.preview.cget("state") == "normal":
            self.preview.config("disabled")
            self._edit_btn_txt.set("Edit transcript…")
            return

        # If current file is an edited* → enable editing directly
        if is_edited:
            self.preview.config("normal")
            self._edit_btn_txt.set("Done editing (lock)")
            return

        # Current file is original → search for existing edited siblings
        base = path.with_suffix("")  # remove extension
        ext = path.suffix
        folder = path.parent

        # Find name.edited{,2,3,...}.ext
        edited_candidates = []
        for p in folder.iterdir():
            if not p.is_file(): continue
            if p.suffix.lower() != ext.lower(): continue
            st = p.stem  # e.g., "name.edited2"
            if st == f"{base.stem}.edited" or (st.startswith(f"{base.stem}.edited") and st[len(f"{base.stem}.edited"):].isdigit()):
                edited_candidates.append(p)
        edited_candidates.sort(key=lambda p: (p.stem, p.suffix))

        def _open_path(pp: Path):
            try:
                with open(pp, "r", encoding="utf-8") as f: txt = f.read()
            except Exception as e:
                self._post("log", f"Open failed: {e}"); return
            self._current_preview_path = str(pp)
            if self._current_preview_path not in self.output_files:
                self.output_files.append(self._current_preview_path)
                self.output_list.insert("end", self._current_preview_path)
            self._set_preview_text(txt)
            self._parse_preview_segments()
            self.preview.config("normal")
            self._edit_btn_txt.set("Done editing (lock)")
            try: self._auto_add_and_bind_media_for_output(str(pp))
            except Exception: pass

        # If found at least one edited → ask user to pick or create new
        if edited_candidates:
            win = tk.Toplevel(self); win.title("Choose edited transcript"); win.transient(self); win.grab_set()
            ttk.Label(win, text="I found edited versions of this file. Choose one to continue editing or create a new edited copy:").pack(anchor="w", padx=10, pady=(10,6))
            lb = tk.Listbox(win, height=min(6, len(edited_candidates)))
            for pp in edited_candidates: lb.insert("end", str(pp.name))
            lb.pack(fill="x", expand=True, padx=10, pady=6)
            btns = ttk.Frame(win); btns.pack(fill="x", padx=10, pady=(0,10))

            def on_open():
                sel = lb.curselection()
                if not sel:
                    messagebox.showerror("Select", "Pick an edited file from the list.")
                    return
                pp = edited_candidates[sel[0]]
                win.destroy()
                _open_path(pp)

            def on_new():
                # next free suffix
                n = 1
                while True:
                    cand = folder / f"{base.stem}.edited{'' if n==1 else n}{ext}"
                    if not cand.exists(): break
                    n += 1
                try:
                    text = self.preview.get("1.0","end-1c")
                    with open(cand, "w", encoding="utf-8") as f: f.write(text)
                except Exception as e:
                    self._post("log", f"Create edited failed: {e}"); win.destroy(); return
                win.destroy()
                _open_path(cand)

            ttk.Button(btns, text="Open selected", command=on_open).pack(side="left")
            ttk.Button(btns, text="Create new edited copy", command=on_new).pack(side="right")
            return

        # No edited found → create first edited copy
        first = folder / f"{base.stem}.edited{ext}"
        try:
            text = self.preview.get("1.0","end-1c")
            with open(first, "w", encoding="utf-8") as f: f.write(text)
        except Exception as e:
            self._post("log", f"Create edited failed: {e}"); return
        _open_path(first)

    # --------------------------- Preview editing + autosave ---------------------------
    def _on_preview_modified(self, _evt=None):
        # Only autosave user edits when editing an .edited* file and state is 'normal'
        if self._preview_prog_update:
            self.preview.edit_modified(False); return
        if not self._current_preview_path:
            self.preview.edit_modified(False); return
        if self.preview.cget("state") != "normal":
            self.preview.edit_modified(False); return
        if ".edited" not in Path(self._current_preview_path).stem:
            # Do not autosave originals anymore
            self.preview.edit_modified(False); return
        if self._save_after_id:
            try: self.after_cancel(self._save_after_id)
            except Exception: pass
        self._save_after_id = self.after(700, self._save_preview_to_edited)

    def _save_preview_to_edited(self):
        self._save_after_id = None
        if not self._current_preview_path:
            self.preview.edit_modified(False); return
        # Write BACK to the currently-open edited file only
        cur = Path(self._current_preview_path)
        if ".edited" not in cur.stem:
            self.preview.edit_modified(False); return
        out_path = str(cur)
        try:
            content = self.preview.get("1.0","end-1c")
            with open(out_path, "w", encoding="utf-8") as f: f.write(content)
            if out_path not in self.output_files:
                self.output_files.append(out_path); self.output_list.insert("end", out_path)
            self._post("log", f"[Output] Autosaved edits → {out_path}")
            try: self._parse_preview_segments()
            except Exception: pass
        except Exception as e:
            self._post("log", f"[Output] Autosave failed: {e}")
        finally:
            self.preview.edit_modified(False)

    # --------------------------- Segment parsing & highlight (all formats) ---------------------------
    def _parse_time_any(self, s: str) -> float | None:
        s = s.strip()
        m = re.match(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?(?:[.,](\d{1,3}))?$', s)
        if m:
            if m.group(3) is None:
                mm, ss = int(m.group(1)), int(m.group(2)); ms = int((m.group(4) or "0").ljust(3,"0"))
                return mm*60 + ss + ms/1000.0
            else:
                hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3)); ms = int((m.group(4) or "0").ljust(3,"0"))
                return hh*3600 + mm*60 + ss + ms/1000.0
        try:
            return float(s.replace(",", "."))
        except Exception:
            return None

    def _parse_preview_segments(self):
        txt = self.preview.get("1.0","end-1c")
        self._segments.clear()
        self.preview.tag_remove("seg_active", "1.0", "end")
        ext = (Path(self._current_preview_path).suffix.lower() if self._current_preview_path else "")

        if ext == ".srt":
            self._parse_srt(txt)
        elif ext == ".vtt":
            self._parse_vtt(txt)
        elif ext == ".csv":
            self._parse_csv(txt)
        elif ext == ".json":
            self._parse_json(txt)
        else:
            self._parse_txt(txt)

        self._active_seg_idx = None

    def _parse_txt(self, txt: str):
        pat = re.compile(r'^\[(\d{2}):(\d{2})\s*[–-]\s*(\d{2}):(\d{2})\]', re.UNICODE)
        line_no = 1
        for line in txt.splitlines():
            m = pat.match(line)
            if m:
                s = int(m.group(1))*60 + int(m.group(2))
                e = int(m.group(3))*60 + int(m.group(4))
                self._segments.append({
                    "start": float(s),
                    "end": float(e),
                    "start_idx": f"{line_no}.0",
                    "end_idx": f"{line_no}.end",
                })
            line_no += 1

    def _parse_srt(self, txt: str):
        lines = txt.splitlines()
        i = 0; line_no = 1
        time_re = re.compile(r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})')
        while i < len(lines):
            start_ln = line_no
            if re.match(r'^\s*\d+\s*$', lines[i] if i < len(lines) else ""): i += 1; line_no += 1
            if i >= len(lines): break
            m = time_re.search(lines[i])
            if not m:
                i += 1; line_no += 1; continue
            s = self._parse_time_any(m.group(1)); e = self._parse_time_any(m.group(2))
            time_ln = line_no
            i += 1; line_no += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1; line_no += 1
            end_ln = line_no - 1
            self._segments.append({
                "start": float(s or 0.0),
                "end": float(e or (s or 0)+0.5),
                "start_idx": f"{time_ln}.0",
                "end_idx": f"{end_ln}.end",
            })
            if i < len(lines) and lines[i].strip()=="":
                i += 1; line_no += 1

    def _parse_vtt(self, txt: str):
        lines = txt.splitlines()
        i = 0; line_no = 1
        time_re = re.compile(r'(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})')
        if i < len(lines) and lines[i].strip().upper().startswith("WEBVTT"):
            i += 1; line_no += 1
        while i < len(lines):
            if i < len(lines) and lines[i].strip()=="":
                i += 1; line_no += 1; continue
            if i >= len(lines): break
            m = time_re.search(lines[i])
            if not m:
                i += 1; line_no += 1; continue
            s = self._parse_time_any(m.group(1)); e = self._parse_time_any(m.group(2))
            time_ln = line_no
            i += 1; line_no += 1
            while i < len(lines) and lines[i].strip() != "":
                i += 1; line_no += 1
            end_ln = line_no - 1
            self._segments.append({
                "start": float(s or 0.0),
                "end": float(e or (s or 0)+0.5),
                "start_idx": f"{time_ln}.0",
                "end_idx": f"{end_ln}.end",
            })
            if i < len(lines) and lines[i].strip()=="":
                i += 1; line_no += 1

    def _parse_csv(self, txt: str):
        line_no = 1
        header = None
        for raw in txt.splitlines():
            line = raw.strip()
            if line_no == 1:
                if re.search(r'\bstart\b', line, re.I) and re.search(r'\bend\b', line, re.I):
                    header = True
                    line_no += 1
                    continue
            parts = [p.strip().strip('"') for p in re.split(r',(?![^"]*"[^"]*(?:"[^"]*"[^"]*)*$)', line)]
            if len(parts) >= 2:
                s = self._parse_time_any(parts[0]); e = self._parse_time_any(parts[1])
                if s is not None and e is not None:
                    self._segments.append({
                        "start": float(s),
                        "end": float(e),
                        "start_idx": f"{line_no}.0",
                        "end_idx": f"{line_no}.end",
                    })
            line_no += 1

    def _parse_json(self, txt: str):
        segs = []
        try:
            data = json.loads(txt)
            if isinstance(data, dict) and "segments" in data:
                arr = data.get("segments") or []
            elif isinstance(data, list):
                arr = data
            else:
                arr = []
            for s in arr:
                try:
                    ss = float(s.get("start") if isinstance(s, dict) else s[0])
                    ee = float(s.get("end")   if isinstance(s, dict) else s[1])
                    segs.append((ss, ee))
                except Exception:
                    continue
        except Exception:
            return
        lines = txt.splitlines()
        start_lines = []
        for i, line in enumerate(lines, start=1):
            if '"start"' in line:
                start_lines.append(i)
        for i, (ss, ee) in enumerate(segs):
            start_ln = start_lines[i] if i < len(start_lines) else 1
            end_ln = start_ln
            j = start_ln
            while j <= len(lines):
                if j > start_ln and '"start"' in lines[j-1]:
                    end_ln = j-2
                    break
                if lines[j-1].strip().startswith("}"):
                    end_ln = j
                    if j < len(lines) and lines[j].strip()==",":
                        end_ln = j+1
                    break
                j += 1
            if j > len(lines):
                end_ln = len(lines)
            self._segments.append({
                "start": float(ss),
                "end": float(ee),
                "start_idx": f"{start_ln}.0",
                "end_idx": f"{end_ln}.end",
            })

    def _index_inside_range(self, idx: str, start_idx: str, end_idx: str) -> bool:
        return (self.preview.compare(idx, ">=", start_idx) and self.preview.compare(idx, "<=", end_idx))

    def _on_preview_double_click(self, event):
        idx = self.preview.index(f"@{event.x},{event.y}")
        seg_idx = None
        for i, seg in enumerate(self._segments):
            if self._index_inside_range(idx, seg["start_idx"], seg["end_idx"]):
                seg_idx = i; break
        if seg_idx is None:
            return
        if self._active_seg_idx == seg_idx and not self._paused and self._sd:
            self._stop_playback()
            return
        start = float(self._segments[seg_idx]["start"])
        self._set_active_highlight(seg_idx)
        self._play_from(start)

    def _set_active_highlight(self, seg_idx: int | None):
        self.preview.tag_remove("seg_active", "1.0", "end")
        self._active_seg_idx = seg_idx
        if seg_idx is None: return
        seg = self._segments[seg_idx]
        self.preview.tag_add("seg_active", seg["start_idx"], seg["end_idx"])
        self.preview.see(seg["start_idx"])

    def _highlight_for_time(self, t_sec: float):
        for i, seg in enumerate(self._segments):
            if seg["start"] <= t_sec < seg["end"]:
                if self._active_seg_idx != i:
                    self._set_active_highlight(i)
                return
        if self._active_seg_idx is not None:
            self._set_active_highlight(None)

    # --------------------------- Auto media binding + normalized/speed WAV ---------------------------
    _MEDIA_EXTS = (".mp3",".wav",".m4a",".flac",".ogg",".aac",".wma",".webm",".mp4",".mkv",".mov",".avi")
    _OUT_EXTS = (".json",".txt",".csv",".srt",".vtt")

    def _strip_our_suffixes(self, stem: str) -> str:
        return re.sub(r"(\.(translated|summarised|corrected)(\.[A-Za-z_]+)?|\.(custom|edited))$", "", stem)

    def _auto_add_and_bind_media_for_output(self, out_path: str):
        out_p = Path(out_path)
        stem = self._strip_our_suffixes(out_p.stem)
        for m in self.media_files:
            if Path(m).stem == stem:
                self._current_media_path = m
                try:
                    # Preload normalized base; actual playback loads speed-specific
                    _ = self._prepare_normalised_wav(m)
                except Exception as e: self._post("log", f"[Player] Normalise failed: {e}")
                return
        try:
            for cand in out_p.parent.iterdir():
                if not cand.is_file(): continue
                if cand.suffix.lower() in self._MEDIA_EXTS and self._strip_our_suffixes(cand.stem) == stem:
                    self._current_media_path = str(cand)
                    if self._current_media_path not in self.media_files:
                        self.media_files.append(self._current_media_path)
                        self.media_list.insert("end", self._current_media_path)
                    try:
                        _ = self._prepare_normalised_wav(self._current_media_path)
                    except Exception as e: self._post("log", f"[Player] Normalise failed: {e}")
                    return
        except Exception:
            pass

    def _normalized_wav_path(self, media_path: str) -> str:
        p = Path(media_path)
        return str(p.with_suffix(".norm.wav"))

    def _speed_wav_path(self, media_path: str, speed: float) -> str:
        p = Path(media_path)
        # Use clean suffix like .norm.x1.25.wav
        return str(p.with_suffix(f".norm.x{speed:.2f}.wav"))

    def _prepare_normalised_wav(self, media_path: str) -> str:
        """Create/refresh a PCM 16-bit 48kHz stereo WAV next to source."""
        out = self._normalized_wav_path(media_path)
        src = Path(media_path); dst = Path(out)
        need = True
        try:
            if dst.exists():
                need = (dst.stat().st_mtime < src.stat().st_mtime) or (dst.stat().st_size < 16_000)
        except Exception:
            need = True
        if need:
            ff = ffmpeg_path()
            cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", media_path,
                   "-vn", "-ac", "2", "-ar", "48000",
                   "-c:a", "pcm_s16le",
                   out]
            self._post("log", f"[Player] Creating normalized WAV → {out}")
            subprocess.run(cmd, check=True)
        return out

    def _prepare_playback_wav(self, media_path: str, speed: float) -> str:
        """Return path to WAV at requested speed. Uses normalized base + atempo filters."""
        base = self._prepare_normalised_wav(media_path)
        s = float(speed)
        if abs(s - 1.0) < 1e-3:
            return base
        # Clamp to ffmpeg atempo supported range per filter (0.5..2.0); we chain if needed, but we clamp UI to 0.5..2.0 anyway.
        s = max(0.5, min(2.0, s))
        out = self._speed_wav_path(media_path, s)
        src = Path(base); dst = Path(out)
        need = True
        try:
            if dst.exists():
                need = (dst.stat().st_mtime < src.stat().st_mtime) or (dst.stat().st_size < 16_000)
        except Exception:
            need = True
        if need:
            ff = ffmpeg_path()
            # Build atempo chain (just one stage because s is clamped)
            filt = f"atempo={s:.6f}"
            cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", base,
                   "-filter:a", filt,
                   "-vn", "-c:a", "pcm_s16le", out]
            self._post("log", f"[Player] Preparing speed {s:.2f}× → {out}")
            subprocess.run(cmd, check=True)
        return out

    # --------------------------- Media playback (speed-aware) ---------------------------
    def _apply_speed_change(self):
        # Parse + clamp; update field to effective value
        try:
            new_s = float(self._speed_var.get().strip().lower().replace("x",""))
        except Exception:
            messagebox.showerror("Invalid speed", "Enter a number like 0.75, 1.0, 1.25, 1.5, or 2.0")
            self._speed_var.set(f"{self._current_speed:.2f}")
            return
        new_s = max(0.5, min(2.0, new_s))
        self._speed_var.set(f"{new_s:.2f}")
        old_s = self._current_speed
        self._current_speed = new_s

        if not (HAS_SD and self._sd and self._ensure_media_bound()):
            return

        try:
            # Map to equivalent original time then re-load new speed file
            t_play = float(self._sd.current_time())
            t_orig = t_play * max(0.001, old_s)
            new_start = t_orig / max(0.001, new_s)

            wav = self._prepare_playback_wav(self._current_media_path, self._current_speed)
            self._sd.load(wav)

            # Resume at mapped position if we were playing; otherwise stay paused
            if not self._paused:
                self._post("log", f"[Player] speed {new_s:.2f}× • {Path(wav).name} @ {new_start:.3f}s")
                self._sd.play(new_start)
                self._pause_btn_txt.set("Pause CTRL+SPACE ⏸")
                self._tick_player()
            else:
                self._post("log", f"[Player] speed ready {new_s:.2f}×; press Play")
        except Exception as e:
            self._post("log", f"[Player] speed change failed: {e}")

    def _play_all(self):
        if not HAS_SD or not self._sd:
            _dep_missing("sounddevice soundfile", "Audio playback"); return
        if not self._ensure_media_bound(): return
        try:
            wav = self._prepare_playback_wav(self._current_media_path, self._current_speed)
            self._sd.load(wav)
        except Exception as e:
            self._post("log", f"[Player] Prepare/load failed: {e}"); return
        self._post("log", f"[Player] play • {wav} @ 0.000s")
        try:
            self._sd.play(0.0)
        except Exception as e:
            self._post("log", f"[Player] start failed: {e}"); return
        self._paused = False
        self._pause_btn_txt.set("Pause CTRL+SPACE ⏸")
        self._tick_player()

    def _play_from(self, start_sec: float):
        if not HAS_SD or not self._sd:
            _dep_missing("sounddevice soundfile", "Audio playback"); return
        if not self._ensure_media_bound(): return
        try:
            wav = self._prepare_playback_wav(self._current_media_path, self._current_speed)
            self._sd.load(wav)
        except Exception as e:
            self._post("log", f"[Player] Prepare/load failed: {e}"); return
        # Map original-start to playback-start by dividing by speed
        pb_start = float(start_sec) / max(0.001, self._current_speed)
        self._post("log", f"[Player] play • {Path(wav).name} @ {pb_start:.3f}s (orig {start_sec:.3f}s)")
        try:
            self._sd.play(pb_start)
        except Exception as e:
            self._post("log", f"[Player] start failed: {e}"); return
        self._paused = False
        self._pause_btn_txt.set("Pause CTRL+SPACE ⏸")
        self._tick_player()

    def _toggle_pause(self):
        if not (HAS_SD and self._sd): return
        try:
            paused = self._sd.toggle_pause()
            self._paused = paused
            self._pause_btn_txt.set("Resume CTRL+SPACE ▶" if paused else "Pause CTRL+SPACE ⏸")
            if not paused:
                self._tick_player()
        except Exception as e:
            self._post("log", f"[Player] Pause/Resume error: {e}")

    def _stop_playback(self):
        if self._player_tick_id:
            try: self.after_cancel(self._player_tick_id)
            except Exception: pass
            self._player_tick_id = None
        try:
            if self._sd: self._sd.pause(True)
        except Exception:
            pass
        self._paused = True
        self._pause_btn_txt.set("Pause ⏸")

    def _ensure_media_bound(self) -> bool:
        if self._current_media_path and os.path.exists(self._current_media_path):
            return True
        if self._current_preview_path:
            try: self._auto_add_and_bind_media_for_output(self._current_preview_path)
            except Exception: pass
        ok = bool(self._current_media_path and os.path.exists(self._current_media_path))
        if not ok:
            self._post("log", "No media file bound to the selected output.")
        return ok

    def _tick_player(self):
        if not (self._sd and not self._paused):
            return
        try:
            t_play = float(self._sd.current_time())
            # Convert playback timeline → original timeline for highlighting
            t_orig = t_play * max(0.001, self._current_speed)
            self._highlight_for_time(t_orig)
            if self._sd.ended():
                self._stop_playback(); return
            self._player_tick_id = self.after(100, self._tick_player)
        except Exception:
            self._stop_playback()

    # --------------------------- Heavy work orchestration (split subprocesses) ---------------------------
    def _stage_run(self, target_fn, job: dict):
        import time
        q = mp.Queue()
        proc = mp.Process(target=target_fn, args=(job, q), daemon=True)

        # Start peak RAM monitor for the duration of this stage
        mem = _MemPeak("stage")
        mem.start()

        t0 = time.perf_counter()
        proc.start()
        result = None
        errtxt = None

        # Poll messages while the child is alive
        while proc.is_alive():
            try:
                kind, payload = q.get(timeout=0.1)
                if kind == "log":
                    self._post("log", payload)
                elif kind == "segments":
                    result = payload
                elif kind == "error":
                    errtxt = payload
                    self._post("log", payload)
            except queue.Empty:
                pass

        # Child exited: drain any remaining messages without timeout
        while True:
            try:
                kind, payload = q.get_nowait()
                if kind == "log":
                    self._post("log", payload)
                elif kind == "segments":
                    result = payload
                elif kind == "error":
                    errtxt = payload
                    self._post("log", payload)
            except queue.Empty:
                break

        proc.join()  # no timeout — wait as long as needed

        # Stop memory monitor and log peak
        peak = mem.stop()
        self._post("log", f"[Stage] done in {time.perf_counter()-t0:.2f}s")
        if peak is not None:
            self._post("log", f"[Stage] peak RAM ≈ {peak:.0f} MB")
        return result, errtxt

    def _run_pipeline_subproc(self, media_path: str, local_whisper_dir: str) -> list:
        # 1) Transcribe (child)
        self._post("log", "[Stage] Transcribe → start")
        job1 = {"media_path": media_path, "whisper_dir": local_whisper_dir, "lang_code": LANG_MAP.get(self.lang_var.get(),"en")}
        res1, err1 = self._stage_run(_proc_transcribe_entry, job1)
        if not res1:
            self._post("log", "Transcribe failed; aborting this file.")
            return []
        segments = res1["segments"]
        lang_for_align = (res1.get("lang_for_align") or job1["lang_code"] or "en").lower()

        # 2) Align (child) if requested and English
        alignment_on = False
        if self.align_var.get() and lang_for_align == "en":
            align_dir = _model_dir(ALIGN_EN_LABEL)
            if align_dir:
                alignment_on = True
                self._post("log", "[Stage] Align → start")
                job2 = {"media_path": media_path, "segments": segments, "align_dir": str(align_dir)}
                res2, err2 = self._stage_run(_proc_align_entry, job2)
                if res2:
                    segments = res2
                else:
                    self._post("log", "[Align] Failed or skipped; using unaligned segments.")
                    alignment_on = False
            else:
                self._post("log", f"[Align] Skipped: alignment model not found.")
        elif self.align_var.get() and lang_for_align != "en":
            self._post("log", f"[Align] Skipped: language is '{lang_for_align}', English-only model bundled.")

        # 3) Diarization (child) if requested
        if self.diar_var.get():
            seg_dir = _model_dir(PYA_SEG_LABEL)
            emb_dir = _model_dir(PYA_EMB_LABEL)
            if seg_dir and emb_dir:
                self._post("log", "[Stage] Diarise → start")
                job3 = {
                    "media_path": media_path,
                    "segments": segments,
                    "pya_seg_dir": str(seg_dir),
                    "pya_emb_dir": str(emb_dir),
                    "num_speakers": self.num_speakers_str.get(),
                    "alignment_on": alignment_on,  # only do word-level assign when True
                }
                res3, err3 = self._stage_run(_proc_diar_entry, job3)
                if res3:
                    segments = res3
                else:
                    self._post("log", "[Diar] Failed or skipped; continuing without diarisation.")
            else:
                self._post("log", "[Diar] Skipped: Local model not found.")
        return segments

    # --------------------------- Processing ---------------------------
    def _run_batch(self):
        if self.media_list.curselection():
            paths=[self.media_list.get(i) for i in self.media_list.curselection()]
        else:
            paths=list(self.media_files)
        if not paths:
            messagebox.showerror("Empty","Add media files first"); return

        whisper_dir = _model_dir(WHISPER_DIR_LABEL)
        if not whisper_dir:
            messagebox.showerror("Model missing", f"Model folder not found.")
            return
        self._post("log", f"Model loaded")
        threading.Thread(target=self._worker_batch, args=(paths, str(whisper_dir)), daemon=True).start()

    def _worker_batch(self, paths, local_whisper_dir):
        try:
            for idx, p in enumerate(paths, 1):
                self._post("status", f"{idx}/{len(paths)} Processing: {os.path.basename(p)}")

                # === Transcribe → Align → Diarize as SEPARATE CHILD PROCESSES ===
                segments = self._run_pipeline_subproc(p, local_whisper_dir)

                # -------- Build preview / export --------
                base=self._dest_base(p)

                if self.mode_var.get()=="subs":
                    data_for_export=segments
                    preview_set = False
                    if self.save_srt_var.get():
                        path=base+".srt"
                        try: self._save_srt(path, segments)
                        except Exception: pass
                        self._post("log", f"Saved: {path}"); self._post("add_output", path)
                        self._post("preview_from_file", path); preview_set = True
                    if self.save_vtt_var.get():
                        path=base+".vtt"
                        try: self._save_vtt(path, segments)
                        except Exception: pass
                        self._post("log", f"Saved: {path}"); self._post("add_output", path)
                        if not preview_set: self._post("preview_from_file", path)
                else:
                    data_for_export=segments
                    path = base + ".txt"
                    try:
                        if any((seg.get("words") or []) for seg in segments):
                            lines = self._merge_by_speaker_word_level(segments)
                        else:
                            lines = self._merge_by_speaker_segment_level(segments)
                        with open(path, "w", encoding="utf-8") as ftxt: ftxt.write("\n".join(lines) + "\n")
                        self._post("log", f"Saved: {path}"); self._post("add_output", path)
                        self._post("preview_from_file", path)
                    except Exception:
                        with open(path, "w", encoding="utf-8") as ftxt:
                            ftxt.write("\n".join((s.get("text") or "").strip() for s in data_for_export))
                        self._post("log", f"Saved (fallback): {path}"); self._post("add_output", path)
                        self._post("preview_from_file", path)

                if self.save_json_var.get():
                    jpath=base+".json"; open(jpath,"w",encoding="utf-8").write(json.dumps(data_for_export, ensure_ascii=False, indent=2))
                    self._post("log", f"Saved: {jpath}"); self._post("add_output", jpath)
                if self.save_csv_var.get():
                    cpath=base+".csv"
                    f=open(cpath,"w",newline="",encoding="utf-8"); w=csv.writer(f); w.writerow(["start","end","text"])
                    for s in data_for_export: w.writerow([s.get("start"), s.get("end"), (s.get("text") or "").strip()])
                    f.close(); self._post("log", f"Saved: {cpath}"); self._post("add_output", cpath)

                try: del segments, data_for_export
                except Exception: pass

            self._post("status","Batch complete."); self._post("log","All done.")
        except Exception:
            self._post("status","Error"); self._post("log", traceback.format_exc())

    # ------- merge helpers for TXT -------
    def _speaker_id_map(self, speakers_in_order):
        mapping = {}; next_id = 1
        for spk in speakers_in_order:
            if spk not in mapping: mapping[spk] = next_id; next_id += 1
        return mapping

    def _format_ts_seconds(self, t):
        try: ts = float(t); m = int(ts // 60); s = int(ts % 60); return f"{m:02d}:{s:02d}"
        except Exception: return "00:00"

    def _merge_by_speaker_word_level(self, segments):
        utterances = []; curr = None; speakers_seen = []
        for seg in (segments or []):
            seg_spk = seg.get("speaker") or "SPEAKER_00"
            words = seg.get("words") or []
            seg_txt = (seg.get("text") or "").strip()
            seg_start = seg.get("start", seg.get("s"))
            seg_end   = seg.get("end", seg.get("e"))

            if words:
                for w in words:
                    spk = w.get("speaker") or seg_spk
                    ws = w.get("start", w.get("s", seg_start)); we = w.get("end",   w.get("e", ws))
                    token = (w.get("word") or w.get("text") or "").strip()
                    if token == "": continue
                    if curr is None or spk != curr["speaker"]:
                        if curr is not None: utterances.append(curr)
                        curr = {"speaker": spk, "start": ws, "end": we, "tokens": [token]}
                        speakers_seen.append(spk)
                    else:
                        if we is not None: curr["end"] = we
                        if token and token not in [",", ".", "!", "?", ":", ";"] and not token.startswith("-"):
                            curr["tokens"].append(" " + token)
                        else:
                            curr["tokens"].append(token)
            else:
                if seg_txt == "":
                    continue
                spk = seg_spk
                if curr is None or spk != curr["speaker"]:
                    if curr is not None: utterances.append(curr)
                    curr = {"speaker": spk, "start": seg_start, "end": seg_end, "tokens": [seg_txt]}
                    speakers_seen.append(spk)
                else:
                    if seg_end is not None: curr["end"] = seg_end
                    curr["tokens"].append(" " + seg_txt)

        if curr is not None: utterances.append(curr)
        spk_map = self._speaker_id_map(speakers_seen)
        out = []
        for u in utterances:
            sid = spk_map.get(u["speaker"], 0)
            label = f"Speaker{sid:02d}"
            start = self._format_ts_seconds(u.get("start", 0.0))
            end   = self._format_ts_seconds(u.get("end", u.get("start", 0.0)))
            text = "".join(u["tokens"]).strip()
            out.append(f"[{start}–{end}] {label}: {text}")
        return out

    def _merge_by_speaker_segment_level(self, segments):
        utterances = []; curr = None; speakers_seen = []
        for seg in (segments or []):
            spk = seg.get("speaker") or "SPEAKER_00"
            st = seg.get("start", seg.get("s")); en = seg.get("end",   seg.get("e"))
            txt = (seg.get("text") or "").strip()
            if txt == "": continue
            if curr is None or spk != curr["speaker"]:
                if curr is not None: utterances.append(curr)
                curr = {"speaker": spk, "start": st, "end": en, "text": [txt]}
                speakers_seen.append(spk)
            else:
                if en is not None: curr["end"] = en
                curr["text"].append(" " + txt)
        if curr is not None: utterances.append(curr)
        spk_map = self._speaker_id_map(speakers_seen)
        out = []
        for u in utterances:
            sid = spk_map.get(u["speaker"], 0)
            label = f"Speaker{sid:02d}"
            start = self._format_ts_seconds(u.get("start", 0.0))
            end   = self._format_ts_seconds(u.get("end", u.get("start", 0.0)))
            text = "".join(u["text"]).strip()
            out.append(f"[{start}–{end}] {label}: {text}")
        return out

    # ------- subtitle writers -------
    def _fmt_srt(self,t):
        h=int(t//3600); t-=h*3600; m=int(t//60); t-=m*60; s=int(t); ms=int(round((t-s)*1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _save_srt(self, path, segs):
        with open(path,"w",encoding="utf-8") as f:
            for i,s in enumerate(segs,1):
                st=float(s.get("start",0.0)); en=float(s.get("end",st+0.5)); txt=(s.get("text") or "").strip()
                f.write(f"{i}\n{self._fmt_srt(st)} --> {self._fmt_srt(en)}\n{txt}\n\n")

    def _save_vtt(self, path, segs):
        with open(path,"w",encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for s in segs:
                st=float(s.get("start",0.0)); en=float(s.get("end",st+0.5)); txt=(s.get("text") or "").strip()
                f.write(f"{self._fmt_srt(st).replace(',','.') } --> {self._fmt_srt(en).replace(',','.') }\n{txt}\n\n")

    # --------------------------- LLM helpers ---------------------------
    def _llm_log(self, msg): self._post("log", f"[LLM] {msg}")

    def _format_chat(self, user_text: str):
        if not self.llm_tok: return user_text
        msgs = [{"role":"user","content":user_text}]
        try:
            if hasattr(self.llm_tok, "apply_chat_template"):
                return self.llm_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        return f"<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"

    def _maybe_strip_think(self, text: str) -> str:
        if self.llm_thinking_var.get():
            return text
        # strip <think>...</think> (case-insensitive), including newlines
        return re.sub(r'(?is)<\s*think\s*>.*?<\s*/\s*think\s*>', '', text).strip()

    def _llm_load_model(self):
        if self._llm_loading: return
        if self.llm_model is not None and self._loaded_model_id == QWEN_DIR_LABEL:
            return
        self._llm_loading = True
        try:
            if not HAS_MLX_LM:
                raise RuntimeError("mlx-lm not installed. pip install mlx-lm")
            local_dir = _model_dir(QWEN_DIR_LABEL)
            if not local_dir:
                raise RuntimeError(f"Model not found locally")
            self._llm_log(f"Model loading")
            self.llm_model, self.llm_tok = _mlx_load(str(local_dir))
            self._loaded_model_id = QWEN_DIR_LABEL
            self._llm_log("Ready.")
        except Exception as e:
            self.llm_model = None; self.llm_tok = None; self._loaded_model_id = None
            self._llm_log(f"Load failed: {e}")
        finally:
            self._llm_loading = False

    def _llm_unload_model(self):
        self.llm_model = None; self.llm_tok = None; self._loaded_model_id = None
        _free_all_memory("LLM unload")
        self._llm_log("Unloaded.")

    def _build_sampler(self):
        def parse_num(s, fallback):
            try:
                if isinstance(fallback, int): return int(s)
                return float(s)
            except Exception:
                return fallback
        max_new    = min(parse_num(self.llm_max_new.get(), 30000), 32768)
        temperature= parse_num(self.llm_temp.get(), 0.3)
        top_p      = parse_num(self.llm_top_p.get(), 0.9)
        top_k      = parse_num(self.llm_top_k.get(), 50)
        if _make_sampler is None:
            return None, max_new
        try:
            sampler = _make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        except TypeError:
            try:
                sampler = _make_sampler(temperature=temperature, top_p=top_p, top_k=top_k)
            except Exception:
                sampler = None
        return sampler, max_new

    def _llm_generate_blocking(self, prompt: str) -> str:
        if self.llm_model is None or self.llm_tok is None:
            raise RuntimeError("LLM not loaded.")
        sampler, max_new = self._build_sampler()
        try:
            if sampler is None:
                out = _mlx_generate(self.llm_model, self.llm_tok,
                                    prompt=self._format_chat(prompt),
                                    max_tokens=max_new, verbose=False)
            else:
                out = _mlx_generate(self.llm_model, self.llm_tok,
                                    prompt=self._format_chat(prompt),
                                    max_tokens=max_new, sampler=sampler, verbose=False)
            if isinstance(out, str): return out
            try:
                return "".join(list(out))
            except Exception:
                return str(out)
        except TypeError:
            try:
                if sampler is None:
                    out = _mlx_generate(self.llm_model, self.llm_tok, prompt=self._format_chat(prompt), max_tokens=max_new)
                else:
                    out = _mlx_generate(self.llm_model, self.llm_tok, prompt=self._format_chat(prompt), max_tokens=max_new, sampler=sampler)
                return out if isinstance(out, str) else "".join(list(out))
            except Exception as e2:
                return f"[Error] {e2}"
        except Exception as e:
            return f"[Error] {e}"

    def _prompt_translate(self, text: str) -> str:
        tgt_name, style = normalise_llm_lang(self.llm_target_lang_var.get())
        return (
            f"Translate the transcript into {tgt_name}. {style}\n"
            "RULES:\n"
            "1) KEEP ALL TIMESTAMPS AND SPEAKER LABELS EXACTLY AS IN THE INPUT do not change the format of them and do not change the timestamp numbers, do not round them, just copy as they are.\n"
            "2) Preserve line breaks/blocks; do not merge/split.\n"
            "3) Do not summarise or omit.\n\nINPUT:\n"
            f"{text}\n\nOUTPUT:\n"
        )

    def _prompt_summarise(self, text: str) -> str:
        tgt_name, style = normalise_llm_lang(self.llm_target_lang_var.get())
        return (
            f"Summarise the following transcript into paragraphs on key topics and details, followed by concise bullet points in. Write in {tgt_name}. {style}\n"
            "Keep key facts, names and claims. Ignore timestamps if you like.\n\n"
            f"{text}\n\nBullet points:\n"
        )

    def _prompt_correct(self, text: str) -> str:
        tgt_name, style = normalise_llm_lang(self.llm_target_lang_var.get())
        return (
            f"Correct grammar/spelling/punctuation in {tgt_name}. {style}\n"
            "RULES:\n1) KEEP TIMESTAMPS & SPEAKER LABELS UNCHANGED, do not change the format of them and do not change the timestamp numbers, do not round them, just copy as they are.\n"
            "2) Preserve line breaks; no summarising, no praraphrasing.\n\nINPUT:\n"
            f"{text}\n\nOUTPUT:\n"
        )

    def _llm_translate_selected(self): self._llm_run_over_outputs("translate")
    def _llm_summarise_selected(self): self._llm_run_over_outputs("summarise")
    def _llm_correct_selected(self):   self._llm_run_over_outputs("correct")

    def _llm_run_over_outputs(self, task: str):
        sel = self.output_list.curselection()
        if not sel:
            return self._llm_log("Select outputs (right list) first.")
        files = [self.output_list.get(i) for i in sel]
        if not self._llm_lock.acquire(blocking=False):
            return self._llm_log("Busy: another LLM task is running.")
        threading.Thread(target=self._llm_worker_queue, args=(task, files), daemon=True).start()

    def _llm_worker_queue(self, task: str, files: list):
        # Per-run peak RAM tracker for LLM step
        mem = _MemPeak("LLM")
        mem.start()
        try:
            self._llm_load_model()
            for idx, path in enumerate(files, 1):
                self._llm_log(f"{task.title()} {idx}/{len(files)}: {os.path.basename(path)}")
                try:
                    with open(path, "r", encoding="utf-8") as f: text = f.read()
                except Exception as e:
                    self._llm_log(f"Read failed: {e}"); continue
                if task == "translate": prompt = self._prompt_translate(text)
                elif task == "summarise": prompt = self._prompt_summarise(text)
                else: prompt = self._prompt_correct(text)
                out = self._llm_generate_blocking(prompt)
                out2 = self._maybe_strip_think(out)
                self._set_preview_text(out2)
                self._autosave_llm_output(path, out2, task)
        except Exception as e:
            self._llm_log(f"LLM queue error: {e}")
        finally:
            try:
                self._llm_unload_model()
            finally:
                if self._llm_lock.locked():
                    self._llm_lock.release()
                peak = mem.stop()
                if peak is not None:
                    self._llm_log(f"Peak RAM ≈ {peak:.0f} MB")

    def _llm_custom_on_files(self):
        sel = self.output_list.curselection()
        if not sel:
            return self._llm_log("Select outputs (right list) first, then click Custom Prompt.")
        files = [self.output_list.get(i) for i in sel]
        win = tk.Toplevel(self)
        win.title("Custom Prompt on selected files")
        win.transient(self); win.grab_set()
        frm = ttk.Frame(win, padding=10); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Enter your instruction (it will be applied to each selected file):").pack(anchor="w")
        txt = tk.Text(frm, width=70, height=10); txt.pack(fill="both", expand=True, pady=(6,8))

        btns = ttk.Frame(frm); btns.pack(fill="x")
        def on_cancel():
            try: win.destroy()
            except Exception: pass
        def on_run():
            prompt_body = txt.get("1.0","end").strip()
            if not prompt_body:
                messagebox.showerror("Empty prompt", "Type what you want the model to do.")
                return
            if not self._llm_lock.acquire(blocking=False):
                self._llm_log("Busy: another LLM task is running.")
                win.destroy()
                return
            def _run():
                mem = _MemPeak("LLM-custom"); mem.start()
                try:
                    self._llm_load_model()
                    for idx, path in enumerate(files, 1):
                        self._llm_log(f"Custom {idx}/{len(files)}: {os.path.basename(path)}")
                        try:
                            with open(path, "r", encoding="utf-8") as f: text = f.read()
                        except Exception as e:
                            self._llm_log(f"Read failed: {e}"); continue
                        prompt = (
                            f"{prompt_body}\n\n"
                            "RULES:\n- Write in {tgt_name}. {style}\n"
                            "INPUT:\n"
                            f"{text}\n\nOUTPUT:\n"
                        )
                        out = self._llm_generate_blocking(prompt)
                        out2 = self._maybe_strip_think(out)
                        self._set_preview_text(out2)
                        base, ext = os.path.splitext(path)
                        ext = ext if ext.lower() in (".txt",".srt",".vtt",".csv",".json") else ".txt"
                        out_path = f"{base}.custom{ext}"
                        try:
                            with open(out_path,"w",encoding="utf-8") as fo: fo.write(out2)
                            self._post("add_output", out_path)
                        except Exception as e:
                            self._llm_log(f"Autosave failed: {e}")
                except Exception as e:
                    self._llm_log(f"Custom task error: {e}")
                finally:
                    try:
                        self._llm_unload_model()
                    finally:
                        if self._llm_lock.locked():
                            self._llm_lock.release()
                        peak = mem.stop()
                        if peak is not None:
                            self._llm_log(f"Peak RAM ≈ {peak:.0f} MB")
            threading.Thread(target=_run, daemon=True).start()
            win.destroy()
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=6)
        ttk.Button(btns, text="Run ▶", command=on_run).pack(side="right")

    def _autosave_llm_output(self, input_path: str, content: str, task: str):
        base, ext = os.path.splitext(input_path)
        suffix = {"translate":"translated","summarise":"summarised","correct":"corrected"}.get(task,"output")
        if ext.lower() not in (".txt",".srt",".vtt",".csv",".json"): ext = ".txt"
        tgt_name, _ = normalise_llm_lang(self.llm_target_lang_var.get())
        safe_lang = re.sub(r"[^A-Za-z0-9\-]+","_", tgt_name)
        out_path = f"{base}.{suffix}.{safe_lang}{ext}"
        try:
            with open(out_path, "w", encoding="utf-8") as f: f.write(content)
            self._post("add_output", out_path)
        except Exception as e:
            self._llm_log(f"Autosave failed: {e}")

# --------------------------- Diarization process (alignment-aware) ---------------------------
def _proc_diar_entry(job: dict, out_q):
    import traceback, tempfile, time
    try:
        # ---------- Lightning __version__ runtime shim ----------
        try:
            import pytorch_lightning as pl
            if not hasattr(pl, "__version__"):
                try:
                    try:
                        from importlib.metadata import version as _pl_version
                    except Exception:
                        from importlib_metadata import version as _pl_version
                except Exception:
                    _pl_version = None
                if _pl_version is not None:
                    try:
                        pl.__version__ = _pl_version("pytorch-lightning")
                    except Exception:
                        pl.__version__ = "0.0.0"
                else:
                    pl.__version__ = "0.0.0"
        except Exception:
            pass
        # -------------------------------------------------------

        import torch
        from pyannote.audio import Pipeline
        try:
            import whisperx
        except Exception:
            whisperx = None
        try:
            import pandas as pd
        except Exception:
            pd = None

        media_path   = job["media_path"]
        segments     = job["segments"]
        pya_seg_dir  = job["pya_seg_dir"]
        pya_emb_dir  = job["pya_emb_dir"]
        num_speakers = job["num_speakers"]
        alignment_on = bool(job.get("alignment_on", False))

        t0 = time.perf_counter()
        out_q.put(("log", "Diarisation: decode audio (mono16k)"))
        wav_f32, sr = _ffmpeg_decode_f32_mono_16k_child(media_path)

        tmp_yaml = Path(tempfile.mkdtemp(prefix="pya_")) / "diar_local.yaml"
        yaml_text = f"""version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: "{(Path(pya_emb_dir) / 'pytorch_model.bin').as_posix()}"
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: "{(Path(pya_seg_dir) / 'pytorch_model.bin').as_posix()}"
    segmentation_batch_size: 32
"""
        tmp_yaml.write_text(yaml_text, encoding="utf-8")
        out_q.put(("log", "Diarisation: load pipeline"))
        pipe = Pipeline.from_pretrained(str(tmp_yaml))
        try:
            pipe.instantiate({"clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7046},
                              "segmentation": {"min_duration_off": 0.0}})
        except Exception:
            pass
        pipe.to(torch.device("mps"))
        waveform = torch.from_numpy(np.ascontiguousarray(wav_f32)).unsqueeze(0)

        diar_kwargs = {}
        ns = (str(num_speakers) or "auto").strip().lower()
        if ns not in ("", "auto"):
            try: diar_kwargs["num_speakers"] = int(ns)
            except Exception: pass

        out_q.put(("log", "Diarisation: running"))
        with torch.no_grad():
            diarization = pipe({"waveform": waveform, "sample_rate": 16000}, **diar_kwargs)
        out_q.put(("log", "Disarisation: assigning speakers"))

        did_word_level = False
        has_word_ts = any((s.get("words") for s in (segments or []))) and any(
            any(isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None for w in (s.get("words") or []))
            for s in (segments or [])
        )

        # Only attempt word-level assignment if BOTH alignment and diarization are in use
        if alignment_on and whisperx is not None and pd is not None and has_word_ts:
            rows = []
            for (segment, _, speaker) in diarization.itertracks(yield_label=True):
                rows.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
            diar_df = pd.DataFrame(rows)

            def _to_minimal_segments_dual(segs):
                out=[]
                for s in (segs or []):
                    try: s_start=float(s.get("start", s.get("s", 0.0)) or 0.0)
                    except Exception: s_start=0.0
                    try: s_end=float(s.get("end", s.get("e", s_start)) or s_start)
                    except Exception: s_end=s_start
                    if s_end < s_start: s_end = s_start
                    ms={"start": s_start, "end": s_end, "text": (s.get("text") or ""), "words":[]}
                    for w in (s.get("words") or []):
                        if not isinstance(w, dict): continue
                        ws=w.get("start", w.get("s")); we=w.get("end", w.get("e"))
                        try:
                            ws=float(ws) if ws is not None else None
                            we=float(we) if we is not None else None
                        except Exception:
                            ws=None; we=None
                        if ws is None or we is None: continue
                        if we < ws: we = ws
                        token=(w.get("word") or w.get("text") or "").strip()
                        ms["words"].append({"start": ws, "end": we, "word": token})
                    out.append(ms)
                return out

            try:
                final = whisperx.assign_word_speakers(diar_df, {"segments": _to_minimal_segments_dual(segments)})
                segments = final["segments"]
                total_words = sum(len(s.get("words") or []) for s in segments)
                with_spk = sum(1 for s in segments for w in (s.get("words") or []) if w.get("speaker"))
                out_q.put(("log", f"Diarisation: word-level speakers {with_spk}/{total_words} words"))

                segments, flips = _smooth_word_speakers(segments, min_run_dur=0.8)
                out_q.put(("log", f"Diarisation: smoothing flips={flips}"))
                did_word_level = True
            except Exception as e:
                out_q.put(("log", f"Diarisation: whisperx.assign_word_speakers failed: {e}"))

        if not did_word_level:
            segments = _segment_level_assign_by_overlap(diarization, segments)
            out_q.put(("log", "Diarisation: segment-level assignment"))

        out_q.put(("log", f"Diarisation: done in {time.perf_counter()-t0:.2f}s"))
        out_q.put(("segments", segments))
    except Exception:
        out_q.put(("error", traceback.format_exc()))

# --------------------------- Tkinter compatibility patch (minor) ---------------------------
def _patch_tk_config():
    """Allow calling Text.config('disabled') or Text.config('normal') as a shorthand."""
    try:
        _orig_conf = tk.Text.config
        def _wrapped(self, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], str) and not kwargs and args[0] in ("disabled","normal"):
                kwargs = {"state": args[0]}
                args = ()
            return _orig_conf(self, *args, **kwargs)
        tk.Text.config = _wrapped
        tk.Text.configure = _wrapped
    except Exception:
        pass

# --------------------------- Entrypoint ---------------------------
def main():
    try:
        ensure_ffmpeg_on_path()
    except SystemExit:
        return
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    _patch_tk_config()  # apply minor compatibility shim
    App().mainloop()

if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.freeze_support()   # <-- prevents child processes from running main()
    except Exception:
        pass
    main()
