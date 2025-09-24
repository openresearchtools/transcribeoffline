# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, csv, time, queue, shutil, traceback, threading, re, subprocess, tempfile, importlib, gc
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    import tkinter.font as tkfont
except Exception:
    tkfont = None

import numpy as np
import multiprocessing as mp

# --------------------------- Strict offline sockets ---------------------------
def _offline_enable():
    import socket
    if getattr(_offline_enable, "_ON", False): return
    _offline_enable._ON = True
    class _NoNetSock(socket.socket):
        def connect(self, *a, **k): raise OSError("Network blocked (Transcribe Offline)")
        def connect_ex(self, *a, **k): raise OSError("Network blocked (Transcribe Offline)")
    def _raise(*a, **k): raise OSError("Network blocked (Transcribe Offline)")
    socket.socket = _NoNetSock
    socket.create_connection = _raise
    socket.getaddrinfo = _raise
    if hasattr(socket, "create_server"): socket.create_server = _raise
_offline_enable()

# Caches → temp so nothing lands in user home
TMP_CACHE = Path(tempfile.mkdtemp(prefix="transcribe_off_cache_"))
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

APP_NAME = "Transcribe Offline"

# --------------------------- Max performance caps (no modes) ---------------------------
def _cpu_logical_cores() -> int:
    try: return max(1, os.cpu_count() or 1)
    except Exception: return 1

def _recommended_threads() -> tuple[int, int]:
    """
    Always MAX performance: returns (threads, interop_threads).
    No UI, no modes. Downstream code uses this.
    """
    n = _cpu_logical_cores()
    threads = max(1, n)
    interop = max(1, n // 2)
    # Optional env overrides
    try: threads = int(os.getenv("TRANSOFF_THREADS", threads))
    except Exception: pass
    try: interop = int(os.getenv("TRANSOFF_INTEROP", interop))
    except Exception: pass
    return threads, interop

def _apply_perf_env_caps():
    try:
        thr, _ = _recommended_threads()
        os.environ["OMP_NUM_THREADS"]       = str(thr)
        os.environ["MKL_NUM_THREADS"]       = str(thr)
        os.environ["OPENBLAS_NUM_THREADS"]  = str(thr)
        os.environ["NUMEXPR_NUM_THREADS"]   = str(thr)
    except Exception:
        pass

_apply_perf_env_caps()

# --------------------------- Resources ---------------------------
def _resource_base() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

CONTENT = _resource_base() / "content"
MODELS  = CONTENT / "models"
VENDOR  = CONTENT / "vendor"
LLAMA_VENDOR_DIR = VENDOR / "llama.cpp"  # kept for optional local binaries; Part 2 prefers PATH on Ubuntu

# --------------------------- Models / labels ---------------------------
FASTER_WHISPER_DIR_LABEL = "mobiuslabsgmbh__faster-whisper-large-v3-turbo"
ALIGN_EN_LABEL           = "facebook__wav2vec2-base-960h"
PYA_SEG_LABEL            = "pyannote__segmentation-3.0"
PYA_EMB_LABEL            = "pyannote__wespeaker-voxceleb-resnet34-LM"
QWEN_GGUF_DIR_LABEL      = "unsloth__Qwen3-4B-Instruct-2507-GGUF"
QWEN_GGUF_FILENAME       = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

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

# --------------------------- FFmpeg helper (Ubuntu: use system ffmpeg) ---------------------------
def ensure_ffmpeg_on_path() -> str:
    """
    Ubuntu/Linux: require a user-installed ffmpeg available on PATH.
    We do NOT ship or vendor ffmpeg. If not found, show a helpful error.
    """
    ff = shutil.which("ffmpeg")
    if not ff:
        messagebox.showerror(
            "FFmpeg not found",
            "System 'ffmpeg' was not found in $PATH.\n\n"
            "Install it on Ubuntu with:\n\n"
            "  sudo apt update && sudo apt install -y ffmpeg\n\n"
            "Then restart the app."
        )
        raise SystemExit(1)
    return ff

def ffmpeg_path() -> str: return ensure_ffmpeg_on_path()

# --------------------------- Playback (sounddevice) ---------------------------
HAS_SD = False
try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SD=True
except Exception:
    HAS_SD=False

class _SDStreamer:
    def __init__(self):
        if not HAS_SD: raise RuntimeError("sounddevice/soundfile not installed")
        self._stream=None; self._sf=None; self._mutex=threading.Lock()
        self._sr=0; self._paused=True; self._start_frames=0; self._pos_frames=0; self._ended=False
    def _ensure_stream(self, sr: int):
        if self._stream and self._sr==sr: return
        self._close_stream_only()
        self._sr=int(sr)
        def _cb(outdata, frames, time_info, status):
            with self._mutex:
                if self._sf is None or self._paused:
                    outdata.fill(0); return
                data = self._sf.read(frames, dtype="float32", always_2d=True)
                if data.shape[1]==1: data=np.repeat(data,2,axis=1)
                n=data.shape[0]
                outdata[:n,:]=data[:n,:2]
                if n<frames: outdata[n:,:].fill(0)
                self._pos_frames+=min(n,frames)
                if n<frames:
                    self._paused=True; self._ended=True
        self._stream=sd.OutputStream(samplerate=self._sr, channels=2, dtype="float32", blocksize=2048, latency="high", callback=_cb)
        self._stream.start()
    def _close_stream_only(self):
        try:
            if self._stream:
                self._stream.stop(); self._stream.close()
        except Exception: pass
        self._stream=None
    def load(self, wav_path:str):
        f=sf.SoundFile(wav_path, mode="r"); self._ensure_stream(int(f.samplerate))
        with self._mutex:
            if self._sf:
                try: self._sf.close()
                except Exception: pass
            self._sf=f; self._paused=True; self._ended=False; self._start_frames=0; self._pos_frames=0
    def play(self, start_sec:float=0.0):
        if self._sf is None: raise RuntimeError("No file loaded")
        with self._mutex:
            frame=max(0,int(float(start_sec)*self._sr))
            try: self._sf.seek(frame)
            except Exception: self._sf.seek(0); frame=0
            self._start_frames=frame; self._pos_frames=0; self._paused=False; self._ended=False
    def toggle_pause(self)->bool:
        with self._mutex:
            self._paused=not self._paused
            return self._paused
    def pause(self,on:bool):
        with self._mutex: self._paused=bool(on)
    def stop(self):
        with self._mutex:
            self._paused=True; self._pos_frames=0; self._start_frames=0; self._ended=False
            try:
                if self._sf: self._sf.seek(0)
            except Exception: pass
    def current_time(self)->float:
        with self._mutex:
            return (self._start_frames+self._pos_frames)/float(self._sr or 1)
    def ended(self)->bool:
        with self._mutex:
            f=self._ended; self._ended=False; return f
    def close(self):
        self.stop()
        try:
            if self._sf: self._sf.close()
        except Exception: pass
        self._sf=None; self._close_stream_only()

# --------------------------- Memory helpers ---------------------------
def _torch_sync_and_free():
    try:
        import torch
        if hasattr(torch,"cuda"):
            try: torch.cuda.synchronize()
            except Exception: pass
            try: torch.cuda.empty_cache()
            except Exception: pass
    except Exception: pass

def _free_all_memory(hint=""):
    gc.collect(); _torch_sync_and_free(); importlib.invalidate_caches()

class _MemPeak:
    def __init__(self, label=""): self.label=label; self._peak=0; self._evt=threading.Event(); self._t=None
    def _sample_once(self):
        try:
            import psutil
            p=psutil.Process(os.getpid()); rss=p.memory_info().rss
            for ch in p.children(recursive=True):
                try: rss+=ch.memory_info().rss
                except Exception: pass
            self._peak=max(self._peak,rss)
        except Exception: pass
    def _run(self):
        self._sample_once()
        while not self._evt.wait(10.0): self._sample_once()
        self._sample_once()
    def start(self):
        try:
            import psutil  # noqa
            self._evt.clear(); self._t=threading.Thread(target=self._run, daemon=True); self._t.start()
        except Exception: pass
    def stop(self)->float|None:
        try:
            self._evt.set()
            if self._t: self._t.join()
            return self._peak/1048576.0
        except Exception: return None

# --------------------------- Languages ---------------------------
LANG_MAP = {
    "Auto-detect":"auto","English":"en","Afrikaans":"af","Arabic":"ar","Armenian":"hy","Azerbaijani":"az","Belarusian":"be",
    "Bosnian":"bs","Bulgarian":"bg","Catalan":"ca","Chinese":"zh","Croatian":"hr","Czech":"cs","Danish":"da","Dutch":"nl",
    "Estonian":"et","Finnish":"fi","French":"fr","Galician":"gl","German":"de","Greek":"el","Hebrew":"he","Hindi":"hi",
    "Hungarian":"hu","Icelandic":"is","Indonesian":"id","Italian":"it","Japanese":"ja","Kannada":"kn","Kazakh":"kk",
    "Korean":"ko","Latvian":"lv","Lithuanian":"lt","Macedonian":"mk","Malay":"ms","Marathi":"mr","Māori":"mi","Nepali":"ne",
    "Norwegian":"no","Persian":"fa","Polish":"pl","Portuguese":"pt","Romanian":"ro","Russian":"ru","Serbian":"sr","Slovak":"sk",
    "Slovenian":"sl","Spanish":"es","Swahili":"sw","Swedish":"sv","Filipino":"tl","Tamil":"ta","Thai":"th","Turkish":"tr",
    "Ukrainian":"uk","Urdu":"ur","Vietnamese":"vi","Welsh":"cy",
}
LANG_NAMES = list(LANG_MAP.keys())

def _llm_langs():
    base=["English (UK)","English (US)"]
    rest=[n for n in LANG_NAMES if not n.startswith("English")]
    return base+rest
LLM_LANG_NAMES=_llm_langs()

def normalise_llm_lang(name: str):
    if name=="English (UK)": return "English (UK)", "Use BRITISH English spelling."
    if name=="English (US)": return "English (US)", "Use US English spelling."
    return name, ""

# --------------------------- Icon helper ---------------------------
def _apply_icon(win: tk.Tk):
    try:
        icns = CONTENT / "AppIcon.icns"
        ico  = CONTENT / "AppIcon.ico"
        if sys.platform.startswith("win") and ico.exists():
            win.iconbitmap(default=str(ico))
        elif sys.platform == "darwin" and icns.exists():
            pass
        else:
            png = CONTENT / "AppIcon.png"
            if png.exists():
                img = tk.PhotoImage(file=str(png))
                win.iconphoto(True, img)
    except Exception:
        pass

# ============================ GUI APP ============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # --- NEW: set WM_CLASS so desktop launchers bind the icon on Ubuntu ---
        try:
            # only set if still default
            if self.winfo_class() == "Tk":
                self.wm_class("TranscribeOffline")
        except Exception:
            pass
        # ----------------------------------------------------------------------

        self.title(APP_NAME)
        self.geometry("1260x900"); self.minsize(1120,760)
        _apply_icon(self)

        # queues/state
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

        # LLM state
        self.llm_target_lang_var = tk.StringVar(value="English (UK)")
        self.llm_max_new = tk.StringVar(value="30000")
        self.llm_temp    = tk.StringVar(value="0.3")
        self.llm_top_p   = tk.StringVar(value="0.9")
        self.llm_top_k   = tk.StringVar(value="50")
        self.llm_rep_pen = tk.StringVar(value="1.12")
        self._llm_lock   = threading.Lock()
        self._llama      = None  # real runner in PART 2/2

        # Output/editor/player state
        self._current_preview_path=None
        self._save_after_id=None
        self._segments=[]
        self._active_seg_idx=None
        self._current_media_path=None
        self._play_source=None
        self._play_speed_var = tk.StringVar(value="1.0")
        self._play_speed = 1.0
        self._player_tick_id=None
        self._paused=True

        # Editing state (NEW)
        self._editing_active = False
        self._editing_path: str | None = None
        self._edit_btn_txt = tk.StringVar(value="Edit transcript…")

        # Player (sounddevice)
        if not HAS_SD:
            self._dep_missing("sounddevice soundfile", "Audio playback")
        self._sd = _SDStreamer() if HAS_SD else None

        self._build_layout()
        self.after(120, self._poll_queue)
        self._update_align_availability()

        # Log chosen perf (max) so users see thread count
        try:
            t, io = _recommended_threads()
            self._post("log", f"[Perf] MAX threads={t} • torch interop={io}")
        except Exception:
            pass

        # Hotkeys
        self.bind_all("<Control-space>", lambda e: self._toggle_pause())

    # ---------- helpers ----------
    def _dep_missing(self, pkg: str, feature: str = ""):
        msg = f"Missing dependency: {pkg}."
        if feature: msg += f"\nFeature: {feature}"
        messagebox.showerror("Missing dependency", msg)

    def _post(self, kind, payload): self.msg_q.put((kind, payload))

    # ---- ttk.Panedwindow helpers (Windows-safe) ----
    def _pw_add(self, pw, child, **kw):
        """Safely add a pane to a ttk.Panedwindow, ignoring unsupported options like 'minsize'."""
        try:
            pw.add(child, **kw)
        except tk.TclError:
            cleaned = {k: v for k, v in kw.items() if k in ("weight", "sticky", "padding")}
            try:
                pw.add(child, **cleaned)
            except tk.TclError:
                pw.add(child)

    def _px_from_lines(self, font_obj, lines, extra=0):
        try:
            lh = font_obj.metrics("linespace")
        except Exception:
            lh = 14
        return int(lines * lh + extra)

    def _bind_min_sash(self, pw, top_min_px=160, bottom_min_px=48, sash_index=0):
        """Emulate a minsize for vertical ttk.Panedwindow by clamping the sash on resize."""
        def _enforce(_evt=None):
            try:
                pw.update_idletasks()
                total = pw.winfo_height()
                if total <= 0:
                    return
                pos = pw.sashpos(sash_index)
                if pos is None:
                    return
                pos = max(top_min_px, min(pos, total - bottom_min_px))
                pw.sashpos(sash_index, pos)
            except Exception:
                pass
        pw.bind("<Configure>", _enforce)
        self.after(120, _enforce)

    # ---------- UI ----------
    def _build_layout(self):
        pad={'padx': 6, 'pady': 4}
        root = ttk.Frame(self); root.pack(fill="both", expand=True)

        # Controls row
        ctrl1 = ttk.Frame(root); ctrl1.pack(fill="x", **pad)
        ttk.Label(ctrl1, text="Language:").pack(side="left", padx=(0,4))
        lang_cb = ttk.Combobox(ctrl1, textvariable=self.lang_var, width=28, values=LANG_NAMES, state="readonly")
        lang_cb.pack(side="left", padx=5); lang_cb.bind("<<ComboboxSelected>>", self._update_align_availability)

        ttk.Label(ctrl1, text="Mode:").pack(side="left", padx=(16,4))
        ttk.Radiobutton(ctrl1, text="Transcribe", variable=self.mode_var, value="transcribe", command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(ctrl1, text="Subtitles",  variable=self.mode_var, value="subs",        command=self._on_mode_change).pack(side="left", padx=5)

        ttk.Button(ctrl1, text="About…", command=self._open_about_window).pack(side="right", padx=(8,0))

        ctrl2 = ttk.Frame(root); ctrl2.pack(fill="x", **pad)
        self._align_chk = ttk.Checkbutton(ctrl2, text="Alignment (English only)", variable=self.align_var)
        self._align_chk.pack(side="left", padx=6)
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

        # Top row: Media inputs & Output files
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

        # Output + Log split (vertical panedwindow; toolbars won't disappear)
        center_split = ttk.Panedwindow(root, orient="vertical")
        center_split.pack(fill="both", expand=True, padx=6, pady=4)

        # ----- Output/editor + controls -----
        output_frame = ttk.Labelframe(center_split, text="Output")

        # LLM controls ROW 1
        top_tools1 = ttk.Frame(output_frame); top_tools1.pack(fill="x", padx=4, pady=(6,4))
        ttk.Label(top_tools1, text="Target:").pack(side="left")
        ttk.Combobox(top_tools1, textvariable=self.llm_target_lang_var, width=18, state="readonly",
                     values=LLM_LANG_NAMES).pack(side="left", padx=(4,12))
        ttk.Button(top_tools1, text="Model parameters…", command=self._open_llm_params).pack(side="left", padx=(0,10))
        ttk.Button(top_tools1, text="Translate", command=self._llm_translate_selected).pack(side="left", padx=4)
        ttk.Button(top_tools1, text="Summarise", command=self._llm_summarise_selected).pack(side="left", padx=4)
        ttk.Button(top_tools1, text="Correct text", command=self._llm_correct_selected).pack(side="left", padx=4)
        ttk.Button(top_tools1, text="Custom prompt…", command=self._llm_custom_prompt_dialog).pack(side="left", padx=8)

        # LLM/Text size + Editing ROW 2
        top_tools2 = ttk.Frame(output_frame); top_tools2.pack(fill="x", padx=4, pady=(0,4))
        ttk.Label(top_tools2, text="Text size:").pack(side="left", padx=(0,6))
        ttk.Button(top_tools2, text="A−", width=3, command=lambda: self._bump_text_size(-1)).pack(side="left")
        ttk.Button(top_tools2, text="A+", width=3, command=lambda: self._bump_text_size(+1)).pack(side="left", padx=(4,0))
        ttk.Button(top_tools2, text="Reset", command=lambda: self._set_text_size(10)).pack(side="left", padx=(4,12))

        ttk.Button(top_tools2, textvariable=self._edit_btn_txt, command=self._toggle_editing).pack(side="left", padx=(2,8))
        ttk.Label(top_tools2, text="(view-only unless editing an .edited file)").pack(side="left")

        # The editor (default: READ-ONLY)
        self.preview_font = tkfont.Font(family="Consolas", size=10) if tkfont else None
        self.preview = tk.Text(output_frame, wrap="word",
                               font=self.preview_font if self.preview_font else ("Consolas", 10),
                               undo=True, state="disabled")
        self.preview.pack(fill="both", expand=True, padx=4, pady=(0,0))
        self.preview.tag_configure("seg_active", background="#FFF59D")
        self.preview.bind("<<Modified>>", self._on_preview_modified)
        self.preview.bind("<Double-Button-1>", self._on_preview_double_click)

        # Player bar (SAME ROW includes speed)
        player_bar = ttk.Frame(output_frame); player_bar.pack(fill="x", padx=4, pady=(4,6))
        ttk.Button(player_bar, text="Play All ▶", command=self._play_all).pack(side="left", padx=4)
        self._pause_btn_txt = tk.StringVar(value="Pause ⏸ (Ctrl+Space)")
        ttk.Button(player_bar, textvariable=self._pause_btn_txt, command=self._toggle_pause).pack(side="left", padx=4)
        ttk.Button(player_bar, text="Stop ⏹", command=self._stop_playback).pack(side="left", padx=4)

        ttk.Label(player_bar, text="Speed ×").pack(side="left", padx=(12,4))
        ttk.Entry(player_bar, textvariable=self._play_speed_var, width=6).pack(side="left")
        ttk.Button(player_bar, text="Apply to next play",
                   command=lambda: self._post("log", f"[Player] next play speed ×{self._safe_speed():.3f}")).pack(side="left", padx=(6,0))

        # ----- Log -----
        log_frame = ttk.Labelframe(center_split, text="Log")
        self.log = tk.Text(log_frame, wrap="word", font=("Consolas", 9), height=6)
        self.log.pack(fill="both", expand=True)

        # Add panes (Windows ttk has no 'minsize'; emulate via sash clamp)
        self._pw_add(center_split, output_frame, weight=5)
        self._pw_add(center_split, log_frame,    weight=1)

        # Clamp sash so toolbars don't disappear on vertical resize
        # Output minimum ≈ 8 lines + padding, Log minimum ≈ 2 lines + padding
        base_font = self.preview_font or (tkfont and tkfont.nametofont("TkFixedFont")) or tkfont.Font(family="Consolas", size=10)
        top_min = self._px_from_lines(base_font, 8, extra=48)
        bot_min = self._px_from_lines(base_font, 2, extra=24)
        self._bind_min_sash(center_split, top_min_px=top_min, bottom_min_px=bot_min)

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w", padx=6, pady=3)

        self._on_mode_change()

    # ---------- Text size ----------
    def _get_font_size(self)->int:
        try:
            return self.preview_font.cget("size") if self.preview_font else 10
        except Exception:
            return 10
    def _set_text_size(self, size:int):
        try:
            if self.preview_font:
                self.preview_font.configure(size=max(7, min(36, int(size))))
        except Exception:
            pass
    def _bump_text_size(self, delta:int):
        self._set_text_size(self._get_font_size()+delta)

    # ---------- About ----------
    def _open_about_window(self):
        win=tk.Toplevel(self); win.title("About"); win.geometry("740x560"); win.transient(self); win.grab_set()
        outer=ttk.Frame(win); outer.pack(fill="both", expand=True, padx=8, pady=8)
        row = ttk.Frame(outer); row.pack(fill="x", pady=(0,6))
        txt=tk.Text(outer, wrap="word", font=("Consolas",10)); txt.pack(fill="both", expand=True)
        def _load(path:Path, fallback:str):
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                return fallback
        def show_about():
            p=CONTENT/"licenses"/"README.MD"
            txt.configure(state="normal"); txt.delete("1.0","end")
            txt.insert("1.0", _load(p, "README.MD not found in content/licenses/")); txt.configure(state="disabled")
        def show_license():
            p=CONTENT/"licenses"/"LICENSE.txt"
            txt.configure(state="normal"); txt.delete("1.0","end")
            txt.insert("1.0", _load(p, "LICENSE.txt not found in content/licenses/")); txt.configure(state="disabled")
        def show_third():
            p=CONTENT/"licenses"/"Third-Party-Licenses.txt"
            txt.configure(state="normal"); txt.delete("1.0","end")
            txt.insert("1.0", _load(p, "Third-Party-Licenses.txt not found in content/licenses/")); txt.configure(state="disabled")
        ttk.Button(row, text="About", command=show_about).pack(side="left")
        ttk.Button(row, text="License", command=show_license).pack(side="left", padx=(6,0))
        ttk.Button(row, text="Third Party Licenses", command=show_third).pack(side="left", padx=(6,0))
        ttk.Button(row, text="Close", command=win.destroy).pack(side="right")
        show_about()

    # ---------- Lists ----------
    def _add_media(self):
        # FIX: use a tuple of patterns (Linux Tk hates semicolons); add "All files"
        paths = filedialog.askopenfilenames(
            title="Add media files",
            filetypes=[
                ("Audio/Video", ("*.mp3","*.wav","*.m4a","*.flac","*.ogg","*.aac","*.wma","*.webm","*.mp4","*.mkv","*.mov","*.avi")),
                ("All files", "*"),
            ]
        )
        for p in paths:
            if p not in self.media_files:
                self.media_files.append(p); self.media_list.insert("end", p)

    def _remove_media(self):
        for idx in reversed(self.media_list.curselection()):
            self.media_list.delete(idx)
        self.media_files = [x for x in self.media_files if x in self.media_list.get(0,"end")]

    def _clear_media(self):
        self.media_list.delete(0,"end"); self.media_files.clear()

    def _add_outputs(self):
        # FIX: use a tuple of patterns + "All files"
        paths = filedialog.askopenfilenames(
            title="Add output files",
            filetypes=[
                ("Text/JSON/Subtitles", ("*.json","*.txt","*.csv","*.srt","*.vtt")),
                ("All files", "*"),
            ]
        )
        for p in paths:
            if p not in self.output_files: self.output_files.append(p); self.output_list.insert("end", p)

    def _remove_outputs(self):
        for idx in reversed(self.output_list.curselection()):
            self.output_list.delete(idx)
        self.output_files = [x for x in self.output_files if x in self.output_list.get(0,"end")]

    def _clear_outputs(self):
        self.output_list.delete(0,"end"); self.output_files.clear()

    # ---------- Read-only / Editing logic ----------
    _EDITED_RE = re.compile(r"\.edited(\d+)?$", re.I)

    def _is_edited_path(self, path: str) -> bool:
        stem, _ext = os.path.splitext(path)
        return bool(self._EDITED_RE.search(stem))

    def _make_edited_candidate(self, original: str, n: int) -> str:
        base, ext = os.path.splitext(original)
        return f"{base}.edited{'' if n == 0 else n}{ext}"

    def _next_edited_path(self, original: str) -> str:
        n = 0
        while True:
            cand = self._make_edited_candidate(original, n)
            if not os.path.exists(cand):
                return cand
            n += 1

    def _find_existing_edited_versions(self, original: str) -> list[str]:
        base, ext = os.path.splitext(original)
        parent = Path(original).parent
        pattern = re.compile(rf"^{re.escape(Path(base).name)}\.edited(\d+)?{re.escape(ext)}$", re.I)
        out=[]
        try:
            for p in parent.iterdir():
                if not p.is_file(): continue
                if pattern.match(p.name):
                    out.append(str(p))
        except Exception:
            pass
        out.sort(key=lambda s: (len(s), s.lower()))
        return out

    def _set_preview_text(self, text: str):
        # always write with state=normal then restore desired state
        self.preview.configure(state="normal")
        self.preview.delete("1.0","end")
        self.preview.insert("1.0", text)
        self.preview.edit_modified(False)
        # restore read-only unless actively editing this file
        if not self._editing_active or (self._editing_path and self._current_preview_path != self._editing_path):
            self.preview.configure(state="disabled")

    def _apply_readonly_state(self):
        # Called after selection changes or when toggling edit
        if self._editing_active and self._editing_path and self._current_preview_path == self._editing_path:
            self.preview.configure(state="normal")
            self._edit_btn_txt.set("Stop editing")
        else:
            self.preview.configure(state="disabled")
            self._edit_btn_txt.set("Edit transcript…")

    def _prompt_pick_edited_version(self, original: str) -> str | None:
        """Return selected edited path, empty string to create new, or None to cancel."""
        existing = self._find_existing_edited_versions(original)
        if not existing:
            return ""  # create new
        dlg = tk.Toplevel(self); dlg.title("Choose edited version"); dlg.transient(self); dlg.grab_set()
        ttk.Label(dlg, text="Found edited versions of this file. Choose one to continue editing, or create a new edited file:").pack(anchor="w", padx=10, pady=(10,6))
        lb = tk.Listbox(dlg, height=min(8, max(3, len(existing))), selectmode="browse")
        for p in existing: lb.insert("end", p)
        lb.pack(fill="both", expand=True, padx=10, pady=(0,8))
        choice = {"path": None}
        def use_selected():
            sel = lb.curselection()
            if not sel:
                messagebox.showerror("Select a version", "Select an edited version from the list or click 'Create new'.", parent=dlg)
                return
            choice["path"] = existing[sel[0]]
            dlg.destroy()
        def create_new():
            choice["path"] = ""  # sentinel for create new
            dlg.destroy()
        def cancel():
            choice["path"] = None
            dlg.destroy()
        btns = ttk.Frame(dlg); btns.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(btns, text="Use selected", command=use_selected).pack(side="right")
        ttk.Button(btns, text="Create new", command=create_new).pack(side="right", padx=(0,8))
        ttk.Button(btns, text="Cancel", command=cancel).pack(side="left")
        dlg.wait_window()
        return choice["path"]

    def _toggle_editing(self):
        # If already editing and current file is that edited file -> turn OFF
        if self._editing_active and self._editing_path and self._current_preview_path == self._editing_path:
            self._editing_active = False
            self._editing_path = None
            self._apply_readonly_state()
            self._post("log", "[Edit] Read-only")
            return

        # We want to start editing whatever is selected in Output files or shown
        if not self._current_preview_path:
            messagebox.showerror("No file", "Select a file in 'Output files (results)' first.")
            return

        cur = self._current_preview_path
        if self._is_edited_path(cur):
            # Editing an edited file directly
            self._editing_active = True
            self._editing_path = cur
            self._apply_readonly_state()
            self._post("log", f"[Edit] Editing: {cur}")
            return

        # Selected is an ORIGINAL -> offer existing edited versions or create new
        choice = self._prompt_pick_edited_version(cur)
        if choice is None:
            return  # cancelled
        if choice == "":
            # create new edited (next available)
            newp = self._next_edited_path(cur)
            try:
                text = self.preview.get("1.0","end-1c") if self.preview else open(cur,"r",encoding="utf-8").read()
            except Exception:
                try: text = open(cur,"r",encoding="utf-8").read()
                except Exception: text = ""
            try:
                with open(newp, "w", encoding="utf-8") as f: f.write(text)
            except Exception as e:
                messagebox.showerror("Create failed", f"Could not create:\n{newp}\n\n{e}")
                return
            if newp not in self.output_files:
                self.output_files.append(newp); self.output_list.insert("end", newp)
            self._post("log", f"[Edit] Created: {newp}")
            self._current_preview_path = newp
            self._set_preview_text(text)
            self._editing_active = True
            self._editing_path = newp
            self._apply_readonly_state()
            return
        else:
            # use existing edited path
            try:
                with open(choice,"r",encoding="utf-8") as f: txt=f.read()
            except Exception as e:
                messagebox.showerror("Open failed", f"Could not open:\n{choice}\n\n{e}")
                return
            if choice not in self.output_files:
                self.output_files.append(choice); self.output_list.insert("end", choice)
            self._current_preview_path = choice
            self._set_preview_text(txt)
            self._editing_active = True
            self._editing_path = choice
            self._apply_readonly_state()
            self._post("log", f"[Edit] Editing existing: {choice}")

    # ---------- Preview / autosave ----------
    def _on_preview_modified(self, _evt=None):
        if not (self._editing_active and self._editing_path and self._current_preview_path == self._editing_path):
            # ignore stray modified flags while read-only
            self.preview.edit_modified(False); return
        if self._save_after_id:
            try: self.after_cancel(self._save_after_id)
            except Exception: pass
        self._save_after_id = self.after(700, self._autosave_editing)

    def _autosave_editing(self):
        self._save_after_id=None
        if not (self._editing_active and self._editing_path and self._current_preview_path == self._editing_path):
            self.preview.edit_modified(False); return
        try:
            content = self.preview.get("1.0","end-1c")
            with open(self._editing_path, "w", encoding="utf-8") as f: f.write(content)
            if self._editing_path not in self.output_files:
                self.output_files.append(self._editing_path); self.output_list.insert("end", self._editing_path)
            self._post("log", f"[Output] Autosaved edits → {self._editing_path}")
            self._parse_preview_segments()
        except Exception as e:
            self._post("log", f"[Output] Autosave failed: {e}")
        finally:
            self.preview.edit_modified(False)

    # ---------- Segment parsing ----------
    def _parse_time_any(self, s: str) -> float | None:
        s=s.strip()
        m=re.match(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?(?:[.,](\d{1,3}))?$', s)
        if m:
            if m.group(3) is None:
                mm, ss = int(m.group(1)), int(m.group(2)); ms = int((m.group(4) or "0").ljust(3,"0"))
                return mm*60 + ss + ms/1000.0
            else:
                hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3)); ms = int((m.group(4) or "0").ljust(3,"0"))
                return hh*3600 + mm*60 + ss + ms/1000.0
        try: return float(s.replace(",", "."))  # seconds
        except Exception: return None

    def _parse_preview_segments(self):
        txt = self.preview.get("1.0","end-1c")
        self._segments.clear()
        self.preview.tag_remove("seg_active", "1.0", "end")
        ext = (Path(self._current_preview_path).suffix.lower() if self._current_preview_path else "")
        if ext == ".srt":   self._parse_srt(txt)
        elif ext == ".vtt": self._parse_vtt(txt)
        elif ext == ".csv": self._parse_csv(txt)
        elif ext == ".json":self._parse_json(txt)
        else:               self._parse_txt(txt)
        self._active_seg_idx=None

    def _parse_txt(self, txt: str):
        pat = re.compile(r'^\[(\d{2}):(\d{2})\s*[–-]\s*(\d{2}):(\d{2})\]')
        line_no=1
        for line in txt.splitlines():
            m=pat.match(line)
            if m:
                s=int(m.group(1))*60 + int(m.group(2))
                e=int(m.group(3))*60 + int(m.group(4))
                self._segments.append({"start": float(s), "end": float(e), "start_idx": f"{line_no}.0", "end_idx": f"{line_no}.end"})
            line_no+=1

    def _parse_srt(self, txt: str):
        lines=txt.splitlines(); i=0; line_no=1
        time_re = re.compile(r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})')
        while i < len(lines):
            if re.match(r'^\s*\d+\s*$', lines[i] if i < len(lines) else ""): i += 1; line_no += 1
            if i >= len(lines): break
            m=time_re.search(lines[i])
            if not m: i+=1; line_no+=1; continue
            s=self._parse_time_any(m.group(1)); e=self._parse_time_any(m.group(2))
            time_ln=line_no; i+=1; line_no+=1
            while i < len(lines) and lines[i].strip() != "": i+=1; line_no+=1
            end_ln=line_no-1
            self._segments.append({"start": float(s or 0.0),"end": float(e or (s or 0)+0.5),"start_idx": f"{time_ln}.0","end_idx": f"{end_ln}.end"})
            if i < len(lines) and lines[i].strip()=="": i+=1; line_no+=1

    def _parse_vtt(self, txt: str):
        lines=txt.splitlines(); i=0; line_no=1
        time_re = re.compile(r'(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})')
        if i < len(lines) and lines[i].strip().upper().startswith("WEBVTT"): i+=1; line_no+=1
        while i < len(lines):
            if i < len(lines) and lines[i].strip()=="": i+=1; line_no+=1; continue
            if i >= len(lines): break
            m=time_re.search(lines[i])
            if not m: i+=1; line_no+=1; continue
            s=self._parse_time_any(m.group(1)); e=self._parse_time_any(m.group(2))
            time_ln=line_no; i+=1; line_no+=1
            while i < len(lines) and lines[i].strip() != "": i+=1; line_no+=1
            end_ln=line_no-1
            self._segments.append({"start": float(s or 0.0),"end": float(e or (s or 0)+0.5),"start_idx": f"{time_ln}.0","end_idx": f"{end_ln}.end"})
            if i < len(lines) and lines[i].strip()=="": i+=1; line_no+=1

    def _parse_csv(self, txt: str):
        line_no=1
        for raw in txt.splitlines():
            line=raw.strip()
            parts=[p.strip().strip('"') for p in re.split(r',(?![^"]*"[^"]*(?:"[^"]*"[^"]*)*$)', line)]
            if len(parts)>=2:
                s=self._parse_time_any(parts[0]); e=self._parse_time_any(parts[1])
                if s is not None and e is not None:
                    self._segments.append({"start": float(s),"end": float(e),"start_idx": f"{line_no}.0","end_idx": f"{line_no}.end"})
            line_no+=1

    def _parse_json(self, txt: str):
        segs=[]
        try:
            data=json.loads(txt)
            if isinstance(data, dict) and "segments" in data: arr=data.get("segments") or []
            elif isinstance(data, list): arr=data
            else: arr=[]
            for s in arr:
                try:
                    ss=float(s.get("start") if isinstance(s, dict) else s[0])
                    ee=float(s.get("end")   if isinstance(s, dict) else s[1])
                    segs.append((ss, ee))
                except Exception: continue
        except Exception: return
        lines=txt.splitlines(); start_lines=[]
        for i,line in enumerate(lines, start=1):
            if '"start"' in line: start_lines.append(i)
        for i,(ss,ee) in enumerate(segs):
            start_ln = start_lines[i] if i < len(start_lines) else 1
            end_ln = start_ln
            j=start_ln
            while j <= len(lines):
                if j > start_ln and '"start"' in lines[j-1]: end_ln=j-2; break
                if lines[j-1].strip().startswith("}"):
                    end_ln=j
                    if j < len(lines) and lines[j].strip()==",": end_ln=j+1
                    break
                j+=1
            if j > len(lines): end_ln=len(lines)
            self._segments.append({"start": float(ss),"end": float(ee),"start_idx": f"{start_ln}.0","end_idx": f"{end_ln}.end"})

    # ---------- Selection & highlight ----------
    def _index_inside_range(self, idx: str, start_idx: str, end_idx: str) -> bool:
        return (self.preview.compare(idx, ">=", start_idx) and self.preview.compare(idx, "<=", end_idx))

    def _on_preview_double_click(self, event):
        idx=self.preview.index(f"@{event.x},{event.y}")
        seg_idx=None
        for i,seg in enumerate(self._segments):
            if self._index_inside_range(idx, seg["start_idx"], seg["end_idx"]):
                seg_idx=i; break
        if seg_idx is None: return
        if self._active_seg_idx == seg_idx and not self._paused and self._sd:
            self._stop_playback(); return
        start=float(self._segments[seg_idx]["start"])
        self._set_active_highlight(seg_idx)
        self._play_from(start)

    def _set_active_highlight(self, seg_idx: int | None):
        self.preview.tag_remove("seg_active", "1.0", "end")
        self._active_seg_idx = seg_idx
        if seg_idx is None: return
        seg=self._segments[seg_idx]
        self.preview.tag_add("seg_active", seg["start_idx"], seg["end_idx"])
        self.preview.see(seg["start_idx"])

    def _highlight_for_time(self, t_source_sec: float):
        for i, seg in enumerate(self._segments):
            if seg["start"] <= t_source_sec < seg["end"]:
                if self._active_seg_idx != i:
                    self._set_active_highlight(i)
                return
        if self._active_seg_idx is not None:
            self._set_active_highlight(None)

    # ---------- Media binding & player ----------
    _MEDIA_EXTS = (".mp3",".wav",".m4a",".flac",".ogg",".aac",".wma",".webm",".mp4",".mkv",".mov",".avi")

    def _strip_our_suffixes(self, stem: str) -> str:
        return re.sub(r"(\.(translated|summarised|corrected)(\.[A-Za-z_]+)?|\.(custom|edited)(\d+)?)$", "", stem, flags=re.I)

    def _auto_add_and_bind_media_for_output(self, out_path: str):
        out_p=Path(out_path); stem=self._strip_our_suffixes(out_p.stem)
        for m in self.media_files:
            if Path(m).stem == stem:
                self._current_media_path = m
                try:
                    self._play_source = self._prepare_playback_wav(m, self._safe_speed())
                    if self._sd: self._sd.load(self._play_source)
                except Exception as e: self._post("log", f"[Player] Prepare failed: {e}")
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
                        self._play_source = self._prepare_playback_wav(self._current_media_path, self._safe_speed())
                        if self._sd: self._sd.load(self._play_source)
                    except Exception as e: self._post("log", f"[Player] Prepare failed: {e}")
                    return
        except Exception:
            pass

    def _safe_speed(self)->float:
        try:
            s=float(self._play_speed_var.get().strip())
            if not np.isfinite(s) or s<=0: raise ValueError
            return max(0.1, min(8.0, s))
        except Exception:
            return 1.0

    def _normalized_wav_path(self, media_path: str, speed: float) -> str:
        p = Path(media_path)
        speed_tag = f"{speed:.3f}".replace(".", "_")
        fname = f"{p.stem}.norm_x{speed_tag}.wav"
        return str(p.with_name(fname))

    def ffmpeg_path_cached(self) -> str:
        try:
            if not hasattr(self, "_ffcached"):
                self._ffcached = ffmpeg_path()
            return self._ffcached
        except Exception:
            return ffmpeg_path()

    def _prepare_playback_wav(self, media_path: str, speed: float) -> str:
        out = self._normalized_wav_path(media_path, speed)
        src = Path(media_path); dst = Path(out)
        need = True
        try:
            if dst.exists():
                need = (dst.stat().st_mtime < src.stat().st_mtime) or (dst.stat().st_size < 16000)
        except Exception:
            need = True
        if need:
            ff = self.ffmpeg_path_cached()
            def _atempo_chain(f):
                chain=[]
                if f>=1.0:
                    remaining=f
                    while remaining>2.0:
                        chain.append("atempo=2.0"); remaining/=2.0
                    chain.append(f"atempo={remaining:.6g}")
                else:
                    inv=1.0/f
                    while inv>2.0:
                        chain.append("atempo=0.5"); inv/=2.0
                    chain.append(f"atempo={1.0/inv:.6g}")
                return ",".join(chain)
            atempo = _atempo_chain(speed)
            cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", media_path, "-vn", "-ac", "2", "-ar", "48000",
                   "-af", atempo, "-c:a", "pcm_s16le", out]
            subprocess.run(cmd, check=True)
        return out

    def _ensure_media_bound(self)->bool:
        if self._current_media_path and os.path.exists(self._current_media_path):
            return True
        if self._current_preview_path:
            try: self._auto_add_and_bind_media_for_output(self._current_preview_path)
            except Exception: pass
        ok = bool(self._current_media_path and os.path.exists(self._current_media_path))
        if not ok: self._post("log", "No media file bound to the selected output.")
        return ok

    def _play_all(self):
        if not HAS_SD or not self._sd:
            self._dep_missing("sounddevice soundfile", "Audio playback"); return
        if not self._ensure_media_bound(): return
        try:
            self._play_speed = self._safe_speed()
            self._play_source = self._prepare_playback_wav(self._current_media_path, self._play_speed)
            self._sd.load(self._play_source)
        except Exception as e:
            self._post("log", f"[Player] Prepare/load failed: {e}"); return
        self._post("log", f"[Player] play • {self._play_source} @ 0.000s • speed ×{self._play_speed:.3f}")
        try: self._sd.play(0.0)
        except Exception as e:
            self._post("log", f"[Player] start failed: {e}"); return
        self._paused=False; self._pause_btn_txt.set("Pause ⏸ (Ctrl+Space)"); self._tick_player()

    def _play_from(self, start_sec: float):
        if not HAS_SD or not self._sd:
            self._dep_missing("sounddevice soundfile", "Audio playback"); return
        if not self._ensure_media_bound(): return
        try:
            self._play_speed = self._safe_speed()
            self._play_source = self._prepare_playback_wav(self._current_media_path, self._play_speed)
            self._sd.load(self._play_source)
        except Exception as e:
            self._post("log", f"[Player] Prepare/load failed: {e}"); return
        pb_start = float(start_sec)/float(self._play_speed or 1.0)
        self._post("log", f"[Player] play • {self._play_source} @ {pb_start:.3f}s (orig {start_sec:.3f}s) • speed ×{self._play_speed:.3f}")
        try: self._sd.play(pb_start)
        except Exception as e:
            self._post("log", f"[Player] start failed: {e}"); return
        self._paused=False; self._pause_btn_txt.set("Pause ⏸ (Ctrl+Space)"); self._tick_player()

    def _toggle_pause(self):
        if not (HAS_SD and self._sd): return
        try:
            paused=self._sd.toggle_pause(); self._paused=paused
            self._pause_btn_txt.set("Resume ▶ (Ctrl+Space)" if paused else "Pause ⏸ (Ctrl+Space)")
            if not paused: self._tick_player()
        except Exception as e:
            self._post("log", f"[Player] Pause/Resume error: {e}")

    def _stop_playback(self):
        if self._player_tick_id:
            try: self.after_cancel(self._player_tick_id)
            except Exception: pass
            self._player_tick_id=None
        try:
            if self._sd: self._sd.pause(True)
        except Exception: pass
        self._paused=True; self._pause_btn_txt.set("Pause ⏸ (Ctrl+Space)")

    def _tick_player(self):
        if not (self._sd and not self._paused): return
        try:
            t_pb = float(self._sd.current_time())
            t_src = t_pb * float(self._play_speed or 1.0)
            self._highlight_for_time(t_src)
            if self._sd.ended():
                self._stop_playback(); return
            self._player_tick_id = self.after(100, self._tick_player)
        except Exception:
            self._stop_playback()

    # ---------- Queue pump ----------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind=="log":
                    ts=time.strftime("%H:%M:%S"); self.log.insert("end", f"[{ts}] {payload}\n"); self.log.see("end")
                elif kind=="status":
                    self.status_var.set(str(payload))
                elif kind=="preview":
                    self._set_preview_text(str(payload))
                    try: self._parse_preview_segments()
                    except Exception: pass
                    self.preview.see("end")
                elif kind=="preview_from_file":
                    p=str(payload)
                    try:
                        with open(p,"r",encoding="utf-8") as f: txt=f.read()
                        self._current_preview_path = p
                        self._set_preview_text(txt)
                        self._parse_preview_segments()
                        self._auto_add_and_bind_media_for_output(p)
                        # leaving edit mode unless this IS our editing file
                        if not (self._editing_active and self._editing_path and p == self._editing_path):
                            self._editing_active = False
                            self._editing_path = None
                        self._apply_readonly_state()
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

    # ---------- Mode & align availability ----------
    def _update_align_availability(self, *_):
        ok = (self.lang_var.get() == "English")
        try:
            if ok: self._align_chk.state(["!disabled"])
            else:  self._align_chk.state(["disabled"])
            self.align_var.set(bool(ok))
        except Exception:
            pass

    def _on_mode_change(self):
        if self.mode_var.get() == "subs":
            self.diar_var.set(False); self.diar_chk.state(["disabled"])
            self.save_txt_var.set(False); self.save_srt_var.set(True); self.save_vtt_var.set(False)
            self.save_csv_var.set(False); self.save_json_var.set(False)
        else:
            self.diar_chk.state(["!disabled"])
            self.save_txt_var.set(True); self.save_srt_var.set(False); self.save_vtt_var.set(False)
            self.save_csv_var.set(False); self.save_json_var.set(False)

    # ---------- LLM UI ----------
    def _open_llm_params(self):
        dlg=tk.Toplevel(self); dlg.title("Model Parameters"); dlg.geometry("420x300"); dlg.transient(self); dlg.grab_set()
        g=ttk.Frame(dlg); g.pack(fill="both", expand=True, padx=12, pady=12)
        def row(lbl, var, w=10):
            r=ttk.Frame(g); r.pack(fill="x", pady=4)
            ttk.Label(r, text=lbl, width=14).pack(side="left")
            ttk.Entry(r, textvariable=var, width=w).pack(side="left")
        row("Max new tokens:", self.llm_max_new)
        row("Temperature:", self.llm_temp)
        row("Top-p:", self.llm_top_p)
        row("Top-k:", self.llm_top_k)
        row("Repeat penalty:", self.llm_rep_pen)
        ttk.Button(g, text="Close", command=dlg.destroy).pack(side="right", pady=(8,0))

    # Prompts (Chat content is preserved; real run in PART 2)
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

    # LLM actions — actual implementation is in PART 2/2
    def _llm_translate_selected(self): self._llm_run_over_outputs("translate")
    def _llm_summarise_selected(self): self._llm_run_over_outputs("summarise")
    def _llm_correct_selected(self):   self._llm_run_over_outputs("correct")

    def _llm_custom_prompt_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Custom Prompt")
        dlg.geometry("720x420")
        dlg.transient(self)
        dlg.grab_set()

        editor = ttk.Frame(dlg)
        editor.pack(fill="both", expand=True, padx=8, pady=8)

        ybar = ttk.Scrollbar(editor, orient="vertical")
        txt = tk.Text(editor, wrap="word", font=("Consolas", 10), undo=True, yscrollcommand=ybar.set)
        ybar.config(command=txt.yview)
        txt.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(0,  weight=1)

        HIDDEN_PREAMBLE = "I will provide you the transcript that you will need to do the following:"

        row = ttk.Frame(dlg)
        row.pack(fill="x", pady=6, padx=8)

        def _submit(_evt=None):
            user_instr = txt.get("1.0", "end-1c").strip()
            final_prompt = HIDDEN_PREAMBLE
            if user_instr:
                final_prompt += "\n\n" + user_instr
            final_prompt += "\n\nHere is the transcript:\n"
            dlg.destroy()
            self._llm_run_over_outputs("custom", custom_prompt=final_prompt)

        ttk.Button(row, text="Run  (Ctrl+Enter)", command=_submit).pack(side="right")
        ttk.Button(row, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 6))
        ttk.Button(row, text="Clear", command=lambda: txt.delete("1.0", "end")).pack(side="left")

        dlg.bind("<Control-Return>", _submit)
        txt.focus_set()

    def _llm_run_over_outputs(self, task: str, custom_prompt: str | None = None):
        # Stub — implemented in PART 2/2
        self._post("log", "[LLM] model loading")
        self._post("log", "[LLM] model loaded")
        self._post("log", "Paste PART 2/2 to enable LLM processing.")

    # Listbox callback
    def _preview_selected(self, event=None):
        sel = self.output_list.curselection()
        if not sel: return
        p = self.output_list.get(sel[0])
        self._post("preview_from_file", p)

# ====== Part 2/2 starts below (pipeline, alignment, diarization, LLM runner, main()) ======
# ============================ PART 2 — Pipeline + llama-cli (non-blocking, robust decoding, save→preview) ============================

import os, sys, re, time, json, csv, queue, traceback, threading, subprocess, tempfile, importlib, gc, shutil
from pathlib import Path
import numpy as np
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------- light mem helpers ----------
def _torch_sync_and_free():
    try:
        import torch
        if hasattr(torch, "cuda"):
            try: torch.cuda.synchronize()
            except Exception: pass
            try: torch.cuda.empty_cache()
            except Exception: pass
    except Exception:
        pass

def _free_all_memory(_=""):
    gc.collect(); _torch_sync_and_free(); importlib.invalidate_caches()

# ---------- Torch thread caps ----------
def _set_torch_threads(threads: int, interop: int):
    try:
        import torch
        torch.set_num_threads(max(1, int(threads)))
        torch.set_num_interop_threads(max(1, int(interop)))
    except Exception:
        pass

# ---------- ffmpeg (child) ----------
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

# =================== CHILD STAGES ===================
def _proc_transcribe_entry(job: dict, out_q):
    """
    Single-process, single-stream transcription.
    - faster-whisper on CPU with int8
    - cpu_threads from _recommended_threads()
    - num_workers = 1 (NO parallel worker pool)
    """
    try:
        from faster_whisper import WhisperModel
        media_path = job["media_path"]; model_dir = job["whisper_dir"]; lang_code = job["lang_code"]

        threads, interop = _recommended_threads()
        # mirror BLAS/OMP caps in child
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

        out_q.put(("log", f"[Transcribe] loading (threads={threads}, workers=1)"))

        model = WhisperModel(
            str(model_dir),
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            num_workers=1
        )

        out_q.put(("log", "[Transcribe] start"))
        # IMPORTANT: word_timestamps=True to prevent downstream word-loss
        segments, info = model.transcribe(
            media_path,
            language=None if lang_code == "auto" else lang_code,
            word_timestamps=True,
            vad_filter=False,
        )

        seg_list = []
        for s in segments:
            words = []
            if s.words:
                for w in s.words:
                    words.append({
                        "start": float(w.start if w.start is not None else (s.start or 0.0)),
                        "end":   float(w.end   if w.end   is not None else (w.start or s.start or 0.0)),
                        "word": (w.word or "").strip()
                    })

            text = (s.text or "").strip()
            if not text and words:
                buff = []
                for tok in words:
                    t = tok.get("word","")
                    if not t: continue
                    if t in [",",".","!","?",":",";"]:
                        buff.append(t)
                    elif not buff:
                        buff.append(t)
                    else:
                        buff.append(" " + t)
                text = "".join(buff).strip()

            seg_list.append({
                "start": float(s.start or 0.0),
                "end":   float(s.end   if s.end is not None else (s.start or 0.0) + 0.02),
                "text":  text,
                "words": words
            })

        lang = (getattr(info, "language", None) or (lang_code if lang_code != "auto" else "en"))
        out_q.put(("log", f"[Transcribe] segments={len(seg_list)} • lang={lang}"))
        out_q.put(("segments", {"segments": seg_list, "lang_for_align": (lang.split('-')[0] if lang else 'en')}))
    except Exception:
        out_q.put(("error", traceback.format_exc()))
    finally:
        try: del model  # noqa
        except Exception: pass
        gc.collect()

# ---------- alignment helpers ----------
def _segments_normalize_for_align(segments):
    def _gt(d, L, S):
        if L in d: return float(d[L])
        if S in d: return float(d[S])
        return None

    out = []
    for seg in (segments or []):
        seg = dict(seg)
        s = _gt(seg, "start", "s"); e = _gt(seg, "end", "e")
        words = list(seg.get("words") or [])

        raw_st = []
        for w in words:
            raw_st.append(_gt(w, "start", "s"))

        if s is None:
            s = next((st for st in raw_st if st is not None), 0.0)

        if e is None:
            last = None
            for w in words:
                we = _gt(w, "end", "e")
                if we is not None: last = we
            if last is None:
                last_start = next((st for st in reversed(raw_st) if st is not None), s or 0.0)
                last = (last_start if last_start is not None else 0.0) + 0.02
            e = last

        if e < s: e = s

        seg["start"] = s; seg["end"] = e; seg["s"] = s; seg["e"] = e

        fixed = []
        n = len(words)
        for i, w in enumerate(words):
            w = dict(w)
            ws = _gt(w, "start", "s"); we = _gt(w, "end", "e")
            if ws is None:
                ws = _gt(fixed[-1], "end", "e") if fixed else s
            if we is None:
                nxt = None
                for j in range(i+1, n):
                    if raw_st[j] is not None: nxt = raw_st[j]; break
                if nxt is None:
                    for j in range(i+1, n):
                        nxt = _gt(words[j], "start", "s")
                        if nxt is not None: break
                we = nxt if nxt is not None else e
            if we < ws: we = ws
            w["start"] = ws; w["end"] = we; w["s"] = ws; w["e"] = we
            if "word" not in w and "text" in w: w["word"] = w["text"]
            fixed.append(w)

        seg["words"] = fixed
        if not (seg.get("text") or "").strip() and fixed:
            buff = []
            for tok in fixed:
                t = (tok.get("word") or tok.get("text") or "").strip()
                if not t: continue
                if t in [",",".","!","?",":",";"]:
                    buff.append(t)
                elif not buff:
                    buff.append(t)
                else:
                    buff.append(" " + t)
            seg["text"] = "".join(buff).strip()

        out.append(seg)
    return out

def _segments_sanitize_word_ts(segments):
    out = []
    for seg in (segments or []):
        seg = dict(seg); words = seg.get("words") or []
        fixed = []; last = float(seg.get("start", 0.0)) if seg.get("start") is not None else 0.0
        for w in words:
            if not isinstance(w, dict): continue
            ws = w.get("start", w.get("s")); we = w.get("end", w.get("e"))
            if ws is None and "ts" in w: ws = w["ts"]
            if we is None and "te" in w: we = w["te"]
            try:
                ws = float(ws) if ws is not None else None
                we = float(we) if we is not None else None
            except Exception:
                ws = None; we = None
            if ws is None and we is None: continue
            if ws is None: ws = last
            if we is None: we = ws
            if we < ws: we = ws
            last = we
            w2 = dict(w); w2["start"] = ws; w2["end"] = we; w2["s"] = ws; w2["e"] = we
            if "word" not in w2 and "text" in w2: w2["word"] = w2["text"]
            fixed.append(w2)

        sst = seg.get("start", seg.get("s"))
        eet = seg.get("end", seg.get("e"))
        try: sst = float(sst) if sst is not None else (fixed[0]["start"] if fixed else 0.0)
        except Exception: sst = 0.0
        try: eet = float(eet) if eet is not None else (fixed[-1]["end"] if fixed else sst)
        except Exception: eet = sst
        if eet < sst: eet = sst

        seg["start"] = sst; seg["end"] = eet; seg["s"] = sst; seg["e"] = eet; seg["words"] = fixed

        if not (seg.get("text") or "").strip() and fixed:
            buff = []
            for tok in fixed:
                t = (tok.get("word") or tok.get("text") or "").strip()
                if not t: continue
                if t in [",",".","!","?",":",";"]:
                    buff.append(t)
                elif not buff:
                    buff.append(t)
                else:
                    buff.append(" " + t)
            seg["text"] = "".join(buff).strip()

        out.append(seg)
    return out

def _proc_align_entry(job: dict, out_q):
    """
    WhisperX alignment on CPU, single process.
    """
    try:
        t, io = _recommended_threads()
        _set_torch_threads(t, io)
        os.environ["OMP_NUM_THREADS"] = str(t)
        os.environ["MKL_NUM_THREADS"] = str(t)
        os.environ["OPENBLAS_NUM_THREADS"] = str(t)
        os.environ["NUMEXPR_NUM_THREADS"] = str(t)

        import whisperx, time as _t
        media_path = job["media_path"]; segments = job["segments"]; align_dir = job["align_dir"]
        t0 = _t.perf_counter()
        out_q.put(("log", f"[Align] threads={t}, interop={io}"))
        out_q.put(("log", "[Align] decode"))
        wav_f32, sr = _ffmpeg_decode_f32_mono_16k_child(media_path)
        out_q.put(("log", "[Align] load"))
        model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu", model_name=str(align_dir))
        out_q.put(("log", "[Align] run"))
        aligned = whisperx.align(segments, model_a, metadata, wav_f32, device="cpu")

        raw = aligned.get("segments") or []
        total_words = sum(len(s.get("words") or []) for s in raw)
        seg2 = _segments_sanitize_word_ts(_segments_normalize_for_align(raw))
        words_ts_after = sum(
            1 for s in seg2 for w in (s.get("words") or [])
            if isinstance(w, dict) and (w.get("start") is not None and w.get("end") is not None)
        )
        out_q.put(("log", f"[Align] aligned words: {words_ts_after}/{total_words} • {(_t.perf_counter()-t0):.2f}s"))
        out_q.put(("segments", seg2))
    except Exception:
        out_q.put(("error", traceback.format_exc()))
    finally:
        try: del model_a, metadata  # noqa
        except Exception: pass
        gc.collect()

# ---------- diarization ----------
def _segment_level_assign_by_overlap(annotation, segments):
    diar_rows = []
    for (segment, _, speaker) in annotation.itertracks(yield_label=True):
        diar_rows.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})

    for s in (segments or []):
        try:
            ss = float(s.get("start", s.get("s", 0.0)) or 0.0)
            se = float(s.get("end",   s.get("e", ss))    or ss)
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
            if not w.get("speaker"): w["speaker"] = best_spk
    return segments

def _proc_diar_entry(job: dict, out_q):
    """
    Pyannote diarization on CPU, single process.
    """
    try:
        t, io = _recommended_threads()
        _set_torch_threads(t, io)
        os.environ["OMP_NUM_THREADS"] = str(t)
        os.environ["MKL_NUM_THREADS"] = str(t)
        os.environ["OPENBLAS_NUM_THREADS"] = str(t)
        os.environ["NUMEXPR_NUM_THREADS"] = str(t)

        import torch
        from pyannote.audio import Pipeline
        import whisperx as _wx
        import pandas as pd

        media_path = job["media_path"]; segments = job["segments"]
        pya_seg_dir = job["pya_seg_dir"]; pya_emb_dir = job["pya_emb_dir"]
        num_speakers = job["num_speakers"]; alignment_on = bool(job.get("alignment_on", False))

        out_q.put(("log", f"[Diarise] threads={t}, interop={io}"))
        out_q.put(("log", "[Diarise] decode"))
        wav_f32, sr = _ffmpeg_decode_f32_mono_16k_child(media_path)

        # local pipeline config with provided local model bins
        tmp_yaml = Path(tempfile.mkdtemp(prefix="pya_")) / "diar_local.yaml"
        tmp_yaml.write_text(f"""version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: "{(Path(pya_emb_dir) / 'pytorch_model.bin').as_posix()}"
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: "{(Path(pya_seg_dir) / 'pytorch_model.bin').as_posix()}"
    segmentation_batch_size: 32
""", "utf-8")

        out_q.put(("log", "[Diarise] load"))
        pipe = Pipeline.from_pretrained(str(tmp_yaml))
        try:
            pipe.instantiate({
                "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7046},
                "segmentation": {"min_duration_off": 0.0}
            })
        except Exception:
            pass

        waveform = torch.from_numpy(np.ascontiguousarray(wav_f32)).unsqueeze(0)
        diar_kwargs = {}
        ns = (str(num_speakers) or "auto").strip().lower()
        if ns not in ("", "auto"):
            try: diar_kwargs["num_speakers"] = int(ns)
            except Exception: pass

        out_q.put(("log", "[Diarise] run"))
        with torch.no_grad():
            diarization = pipe({"waveform": waveform, "sample_rate": 16000}, **diar_kwargs)

        out_q.put(("log", "[Diarise] assign"))

        # Determine if we have word-level timestamps
        has_word_ts = any((s.get("words") or []) for s in (segments or [])) and any(
            any(isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None for w in (s.get("words") or []))
            for s in (segments or [])
        )

        did_word = False
        if alignment_on and has_word_ts:
            raw_rows = []
            for (segment, _, speaker) in diarization.itertracks(yield_label=True):
                raw_rows.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
            raw_rows = sorted(raw_rows, key=lambda r: (r["start"], r["end"]))
            diar_df = pd.DataFrame(raw_rows)

            def _miniseg(segs):
                outm = []
                for s in (segs or []):
                    try: ss = float(s.get("start", s.get("s", 0.0)) or 0.0)
                    except Exception: ss = 0.0
                    try: ee = float(s.get("end",   s.get("e", ss))    or ss)
                    except Exception: ee = ss
                    if ee < ss: ee = ss
                    ms = {"start": ss, "end": ee, "text": (s.get("text") or ""), "words": []}
                    for w in (s.get("words") or []):
                        if not isinstance(w, dict): continue
                        ws = w.get("start", w.get("s")); we = w.get("end", w.get("e"))
                        try:
                            ws = float(ws) if ws is not None else None
                            we = float(we) if we is not None else None
                        except Exception:
                            ws = None; we = None
                        if ws is None or we is None: continue
                        if we < ws: we = ws
                        ms["words"].append({"start": ws, "end": we, "word": (w.get("word") or w.get("text") or "").strip()})
                    outm.append(ms)
                return outm

            try:
                final = _wx.assign_word_speakers(diar_df, {"segments": _miniseg(segments)})
                segments = final["segments"]
                out_q.put(("log", "[Diarise] word-level (raw)"))
                did_word = True
            except Exception as e:
                out_q.put(("log", f"[Diarise] word-level failed: {e}"))

        if not did_word:
            segments = _segment_level_assign_by_overlap(diarization, segments)
            out_q.put(("log", "[Diarise] segment-level"))

        for s in (segments or []):
            if not (s.get("text") or "").strip():
                buff = []
                for w in (s.get("words") or []):
                    t = (w.get("word") or w.get("text") or "").strip()
                    if not t: continue
                    if t in [",",".","!","?",":",";"]:
                        buff.append(t)
                    elif not buff:
                        buff.append(t)
                    else:
                        buff.append(" " + t)
                s["text"] = "".join(buff).strip()

        out_q.put(("segments", segments))
    except Exception:
        out_q.put(("error", traceback.format_exc()))
    finally:
        gc.collect()

# ---------- tiny stage runner ----------
def _stage_run(target_fn, job: dict):
    """
    Runs exactly ONE child process for the stage.
    Parent remains responsive; no parallel jobs are spawned.
    """
    q = mp.Queue(); proc = mp.Process(target=target_fn, args=(job, q), daemon=True)
    t0 = time.perf_counter(); proc.start()
    result = None; errtxt = None
    while proc.is_alive():
        try:
            kind, payload = q.get(timeout=0.1)
            yield (kind, payload)
            if kind == "segments": result = payload
            elif kind == "error": errtxt = payload
        except queue.Empty:
            pass
    while True:
        try:
            kind, payload = q.get_nowait()
            yield (kind, payload)
            if kind == "segments": result = payload
            elif kind == "error": errtxt = payload
        except queue.Empty:
            break
    proc.join(timeout=2.0)
    try:
        if proc.is_alive(): proc.kill()
    except Exception:
        pass
    yield ("_result", (result, errtxt))
    yield ("log", f"[Stage] done in {time.perf_counter()-t0:.2f}s")

# =================== App helpers (merge/save/export) ===================

def _dest_base(self, audio_path): from os.path import splitext; return splitext(audio_path)[0]

def _speaker_id_map(self, speakers_in_order):
    mapping = {}; next_id = 1
    for spk in speakers_in_order:
        if spk not in mapping:
            mapping[spk] = next_id; next_id += 1
    return mapping

def _format_ts_seconds(self, t):
    try:
        ts = float(t); m = int(ts // 60); s = int(ts % 60); return f"{m:02d}:{s:02d}"
    except Exception:
        return "00:00"

def _merge_by_speaker_word_level(self, segments):
    utterances = []; curr = None; speakers_seen = []
    for seg in (segments or []):
        seg_spk = seg.get("speaker") or "SPEAKER_00"
        words = seg.get("words") or []
        seg_txt = (seg.get("text") or "").strip()
        seg_start = seg.get("start", seg.get("s")); seg_end = seg.get("end", seg.get("e"))

        if words:
            for w in words:
                spk = w.get("speaker") or seg_spk
                ws = w.get("start", w.get("s", seg_start))
                we = w.get("end",   w.get("e", ws))
                token = (w.get("word") or w.get("text") or "").strip()
                if token == "": continue
                if curr is None or spk != curr["speaker"]:
                    if curr is not None: utterances.append(curr)
                    curr = {"speaker": spk, "start": ws, "end": we, "tokens": [token]}
                    speakers_seen.append(spk)
                else:
                    if we is not None: curr["end"] = we
                    if token in [",",".","!","?",":",";"]:
                        curr["tokens"].append(token)
                    else:
                        curr["tokens"].append(" " + token)
        else:
            if seg_txt == "": continue
            spk = seg_spk
            if curr is None or spk != curr["speaker"]:
                if curr is not None: utterances.append(curr)
                curr = {"speaker": spk, "start": seg_start, "end": seg_end, "tokens": [seg_txt]}
                speakers_seen.append(spk)
            else:
                if seg_end is not None: curr["end"] = seg_end
                curr["tokens"].append(" " + seg_txt)

    if curr is not None: utterances.append(curr)

    spk_map = self._speaker_id_map(speakers_in_order=speakers_seen)
    out = []
    for u in utterances:
        sid = spk_map.get(u["speaker"], 0); label = f"Speaker{sid:02d}"
        start = self._format_ts_seconds(u.get("start", 0.0))
        end   = self._format_ts_seconds(u.get("end",   u.get("start", 0.0)))
        text  = "".join(u["tokens"]).strip()
        out.append(f"[{start}–{end}] {label}: {text}")
    return out

def _merge_by_speaker_segment_level(self, segments):
    utterances = []; curr = None; speakers_seen = []
    for seg in (segments or []):
        spk = seg.get("speaker") or "SPEAKER_00"
        st  = seg.get("start", seg.get("s")); en = seg.get("end", seg.get("e"))
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

    spk_map = self._speaker_id_map(speakers_in_order=speakers_seen)
    out = []
    for u in utterances:
        sid = spk_map.get(u["speaker"], 0); label = f"Speaker{sid:02d}"
        start = self._format_ts_seconds(u.get("start", 0.0))
        end   = self._format_ts_seconds(u.get("end",   u.get("start", 0.0)))
        text  = "".join(u["text"]).strip()
        out.append(f"[{start}–{end}] {label}: {text}")
    return out

def _fmt_srt(self, t):
    h = int(t // 3600); t -= h*3600
    m = int(t // 60);   t -= m*60
    s = int(t)
    ms = int(round((t - s) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _save_srt(self, path, segs):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            st = float(s.get("start", 0.0))
            en = float(s.get("end",   st + 0.5))
            txt = (s.get("text") or "").strip()
            try:
                md = float(self.maxdur_var.get() or "0")
                if md > 0 and en - st > md: en = st + md
            except Exception:
                pass
            f.write(f"{i}\n{self._fmt_srt(st)} --> {self._fmt_srt(en)}\n{txt}\n\n")

def _save_vtt(self, path, segs):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            st = float(s.get("start", 0.0))
            en = float(s.get("end",   st + 0.5))
            txt = (s.get("text") or "").strip()
            try:
                md = float(self.maxdur_var.get() or "0")
                if md > 0 and en - st > md: en = st + md
            except Exception:
                pass
            f.write(f"{self._fmt_srt(st).replace(',','.')} --> {self._fmt_srt(en).replace(',','.')}\n{txt}\n\n")

# =================== Pipeline orchestrator ===================

def _run_pipeline_subproc(self, media_path: str, local_whisper_dir: str) -> list:
    # 1) Transcribe
    self._post("log", "[Transcribe] start")
    job1 = {"media_path": media_path, "whisper_dir": local_whisper_dir, "lang_code": LANG_MAP.get(self.lang_var.get(), "en")}
    res1 = None; err1 = None
    for kind, payload in _stage_run(_proc_transcribe_entry, job1):
        if kind in ("log","status","preview","preview_from_file","add_output"): self._post(kind, payload)
        elif kind == "_result": res1, err1 = payload
    if not res1:
        if err1 and "No module named 'faster_whisper'" in err1:
            self._post("log", "[Transcribe] missing faster-whisper (pip install faster-whisper)")
        self._post("log", "[Transcribe] failed; aborting.")
        _free_all_memory("after transcribe")
        return []
    segments = res1["segments"]; lang_for_align = (res1.get("lang_for_align") or job1["lang_code"] or "en").lower()
    _free_all_memory("after transcribe")

    # 2) Align (English only)
    alignment_on = False
    if self.align_var.get() and lang_for_align == "en":
        align_dir = _model_dir(ALIGN_EN_LABEL)
        if align_dir:
            self._post("log", "[Align] start")
            job2 = {"media_path": media_path, "segments": segments, "align_dir": str(align_dir)}
            res2 = None; err2 = None
            for kind, payload in _stage_run(_proc_align_entry, job2):
                if kind in ("log","status"): self._post(kind, payload)
                elif kind == "_result": res2, err2 = payload
            if res2:
                segments = res2; alignment_on = True
            else:
                if err2 and "No module named 'whisperx'" in err2:
                    self._post("log", "[Align] missing whisperx (pip install whisperx==3.1)")
                self._post("log", "[Align] failed/skipped; using unaligned.")
        else:
            self._post("log", "[Align] skipped (model not found)")
    elif self.align_var.get() and lang_for_align != "en":
        self._post("log", f"[Align] skipped (lang={lang_for_align})")
    _free_all_memory("after align")

    # 3) Diarise
    if self.diar_var.get():
        seg_dir = _model_dir(PYA_SEG_LABEL); emb_dir = _model_dir(PYA_EMB_LABEL)
        if seg_dir and emb_dir:
            self._post("log", "[Diarise] start")
            job3 = {"media_path": media_path, "segments": segments,
                    "pya_seg_dir": str(seg_dir), "pya_emb_dir": str(emb_dir),
                    "num_speakers": self.num_speakers_str.get(), "alignment_on": alignment_on}
            res3 = None; err3 = None
            for kind, payload in _stage_run(_proc_diar_entry, job3):
                if kind in ("log","status"): self._post(kind, payload)
                elif kind == "_result": res3, err3 = payload
            if res3:
                segments = res3
            else:
                self._post("log", "[Diarise] failed/skipped; continuing.")
        else:
            self._post("log", "[Diarise] skipped (models not found)")
    _free_all_memory("after diarise")

    return segments

# ---------- Batch worker (save → preview actual files) ----------
def _worker_batch(self, paths, local_whisper_dir):
    try:
        for idx, p in enumerate(paths, 1):
            self._post("status", f"{idx}/{len(paths)} {os.path.basename(p)}")

            segments = self._run_pipeline_subproc(p, local_whisper_dir)
            base = self._dest_base(p)

            if self.mode_var.get() == "subs":
                data_for_export = segments
                preview_set = False
                if self.save_srt_var.get():
                    path = base + ".srt"
                    try: self._save_srt(path, segments)
                    except Exception: pass
                    self._post("log", f"Saved: {path}"); self._post("add_output", path)
                    self._post("preview_from_file", path); preview_set = True
                if self.save_vtt_var.get():
                    path = base + ".vtt"
                    try: self._save_vtt(path, segments)
                    except Exception: pass
                    self._post("log", f"Saved: {path}"); self._post("add_output", path)
                    if not preview_set: self._post("preview_from_file", path)
            else:
                data_for_export = segments
                path = base + ".txt"
                try:
                    if any((seg.get("words") or []) for seg in segments):
                        lines = self._merge_by_speaker_word_level(segments)
                    else:
                        lines = self._merge_by_speaker_segment_level(segments)
                    with open(path, "w", encoding="utf-8") as ftxt:
                        ftxt.write("\n".join(lines) + "\n")
                    self._post("log", f"Saved: {path}"); self._post("add_output", path)
                    self._post("preview_from_file", path)
                except Exception:
                    with open(path, "w", encoding="utf-8") as ftxt:
                        ftxt.write("\n".join((s.get("text") or "").strip() for s in data_for_export))
                    self._post("log", f"Saved (fallback): {path}"); self._post("add_output", path)
                    self._post("preview_from_file", path)

            if self.save_json_var.get():
                jpath = base + ".json"
                open(jpath, "w", encoding="utf-8").write(json.dumps(data_for_export, ensure_ascii=False, indent=2))
                self._post("log", f"Saved: {jpath}"); self._post("add_output", jpath)
            if self.save_csv_var.get():
                cpath = base + ".csv"
                f = open(cpath, "w", newline="", encoding="utf-8")
                w = csv.writer(f); w.writerow(["start", "end", "text"])
                for s in data_for_export:
                    w.writerow([s.get("start"), s.get("end"), (s.get("text") or "").strip()])
                f.close(); self._post("log", f"Saved: {cpath}"); self._post("add_output", cpath)

            try: del segments, data_for_export
            except Exception: pass
            _free_all_memory("after save")

        self._post("status", "Batch complete."); self._post("log", "All done.")
    except Exception:
        self._post("status", "Error"); self._post("log", traceback.format_exc())
    finally:
        _free_all_memory("after batch")

def _run_batch(self):
    if self.media_list.curselection():
        paths = [self.media_list.get(i) for i in self.media_list.curselection()]
    else:
        paths = list(self.media_files)
    if not paths:
        messagebox.showerror("Empty", "Add media files first"); return
    whisper_dir = _model_dir(FASTER_WHISPER_DIR_LABEL)
    if not whisper_dir:
        messagebox.showerror("Model missing", f"Faster-Whisper model folder not found:\n{FASTER_WHISPER_DIR_LABEL}")
        return
    self._post("log", "[Transcribe] using local model")
    threading.Thread(target=self._worker_batch, args=(paths, str(whisper_dir)), daemon=True).start()

# =================== LLM (llama-cli via PATH, non-blocking, robust decoding) ===================

def _llama_vendor_dir():
    p = VENDOR / "llama.cpp"
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: pass
    return p

def _find_llama_binary():
    # Prefer system-installed llama-cli on Ubuntu
    for name in ("llama-cli", "llama"):
        p = shutil.which(name)
        if p: return p
    # Optional fallback to bundled/vendor directory if user drops one there
    d = _llama_vendor_dir()
    for c in [d/"llama-cli", d/"main", d/"llama", d/"bin/llama-cli", d/"bin/main", d/"bin/llama"]:
        if Path(c).exists(): return str(c)
    return None

def _resolve_gguf_model():
    pref = MODELS / QWEN_GGUF_DIR_LABEL / QWEN_GGUF_FILENAME
    if pref.exists(): return str(pref)
    for g in MODELS.rglob("*.gguf"): return str(g)
    return None

def _offline_env_for_llama():
    env = os.environ.copy()
    for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","NO_PROXY","no_proxy"): env[k] = ""
    env["GGML_NO_RPC"] = "1"; env["LLAMA_NO_RPC"] = "1"; env["GGML_RPC"] = "0"; env["GGML_RPC_SERVERS"] = ""
    return env

_NOISE_RE = re.compile(r'^(load_backend|warning|build|main|llama_|print_info|load_tensors|system_info|sampler|generate|== Running in interactive mode\.)', re.I)
def _is_noise(line: str) -> bool: return bool(_NOISE_RE.match(line.strip()))

_TS_FIX_RE = re.compile(r'(\[\d{2}:\d{2})[^\d](\d{2}:\d{2}\])')
def _normalise_model_line(s: str) -> str:
    s = _TS_FIX_RE.sub(r'\1–\2', s)
    s = s.replace("\uFFFD", "–")
    return s

# Plain prompts
def _plain_translate(tgt: str, text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a precise translator. Follow instructions exactly.\n"
        "If unsure, output 'I don't know.' Do not add commentary.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Translate this transcript into {tgt}.\n"
        "Rules:\n"
        "- KEEP TIMESTAMPS and SPEAKER LABELS exactly as in the input.\n"
        "- Do not change numbers or formatting.\n"
        "- Preserve line breaks.\n"
        "- Output ONLY the translation.\n"
        "END WITH <<<END>>>.\n\n"
        f"{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _plain_correct(tgt: str, text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a careful copy editor. Correct grammar and spelling only.\n"
        "Do not paraphrase or summarise.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Correct this transcript in {tgt}.\n"
        "Rules:\n"
        "- KEEP TIMESTAMPS and SPEAKER LABELS unchanged.\n"
        "- Preserve line breaks and numbers.\n"
        "- Do not add commentary.\n"
        "END WITH <<<END>>>.\n\n"
        f"{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _plain_summarise(tgt: str, text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a concise summariser. Keep only key facts.\n"
        "If not sure, output 'I don't know.'\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Summarise this transcript into {tgt}.\n"
        "Rules:\n"
        "- Write 1–2 short paragraphs followed by bullet points.\n"
        "- Do not include timestamps.\n"
        "- No extra commentary.\n"
        "END WITH <<<END>>>.\n\n"
        f"{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _plain_custom(instruction: str, text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Follow the instruction exactly.\n"
        "If unsure, say 'I don't know.' Do not invent details.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Instruction: {instruction}\n"
        "Apply this instruction to the transcript.\n"
        "END WITH <<<END>>>.\n\n"
        f"{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _read_text_multi(path: str) -> str:
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception:
        try: return open(path, "r", encoding="cp1252").read()
        except Exception: return Path(path).read_text(errors="ignore")

class _LlamaOnceMinimal:
    """Run llama-cli once. Minimal flags, robust decoding, kill after done."""
    def __init__(self):
        exe = _find_llama_binary(); model = _resolve_gguf_model()
        if not exe or not os.path.exists(exe):
            raise RuntimeError("llama-cli not found on PATH (install/build llama.cpp and ensure 'llama-cli' is accessible)")
        if not model or not os.path.exists(model):
            raise RuntimeError("GGUF model not found under content/models/")
        self.exe = exe
        self.model = model
        self._lock = threading.Lock()

    def run_to_file(
        self,
        prompt: str,
        out_path: str,
        max_new: int = 1024,
        timeout_sec: int = 900,
        temp: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
    ):
        threads, _ = _recommended_threads()

        args = [
            self.exe, "-m", self.model,
            "-n", str(int(max_new)),
            "-t", str(int(threads)),
            "-no-cnv", "--no-display-prompt",
            "-p", prompt,
        ]
        if temp is not None:            args += ["--temp", str(float(temp))]
        if top_p is not None:           args += ["--top-p", str(float(top_p))]
        if top_k is not None:           args += ["--top-k", str(int(top_k))]
        if repeat_penalty is not None:  args += ["--repeat-penalty", str(float(repeat_penalty))]

        env = _offline_env_for_llama()
        cwd = None  # use current working directory on Ubuntu

        with self._lock:
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            wrote = False
            start = time.time()
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                try:
                    while True:
                        if proc.stdout is None:
                            break
                        line_b = proc.stdout.readline()
                        if not line_b:
                            if proc.poll() is not None:
                                break
                            if time.time() - start > timeout_sec:
                                break
                            continue
                        try:
                            line = line_b.decode("utf-8")
                        except Exception:
                            try: line = line_b.decode("cp1252")
                            except Exception: line = line_b.decode("latin-1", "ignore")

                        if "<<<END>>>" in line:
                            line = line.split("<<<END>>>", 1)[0]
                            line = _normalise_model_line(line)
                            if line:
                                f.write(line); f.flush(); wrote = True
                            break

                        if not _is_noise(line):
                            line = _normalise_model_line(line)
                            if line:
                                f.write(line); f.flush(); wrote = True

                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                finally:
                    try:
                        if proc.poll() is None: proc.kill()
                    except Exception:
                        pass

            if not wrote:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("")

# ---------- LLM UI hook (NON-BLOCKING) ----------
def _llm_run_over_outputs(self, task: str, custom_prompt: str | None = None):
    if getattr(self, "_llm_busy", False):
        self._post("log", "[LLM] already running"); return

    sel = self.output_list.curselection()
    if not sel:
        messagebox.showerror("Select a file", "Select a saved output file (right list) first."); return
    in_path = self.output_list.get(sel[0])
    src_txt = _read_text_multi(in_path)

    tgt_name, _ = normalise_llm_lang(self.llm_target_lang_var.get())
    if task == "translate":
        prompt = _plain_translate(tgt_name, src_txt); suffix = ".translated.txt"
    elif task == "summarise":
        prompt = _plain_summarise(tgt_name, src_txt); suffix = ".summarised.txt"
    elif task == "correct":
        prompt = _plain_correct(tgt_name, src_txt); suffix = ".corrected.txt"
    elif task == "custom":
        prompt = _plain_custom(custom_prompt, src_txt); suffix = ".custom.txt"
    else:
        self._post("log", f"[LLM] unknown task: {task}"); return

    base = Path(in_path); out_path = str(base.with_name(base.stem + suffix))
    try:
        mx = int(self.llm_max_new.get().strip())
    except Exception:
        mx = 2048
    mx = max(64, min(mx, 30000))

    def _worker():
        setattr(self, "_llm_busy", True)
        threads, _ = _recommended_threads()
        self._post("log", f"[LLM] starting (threads={threads})")
        try:
            runner = _LlamaOnceMinimal()

            try:    mx_l = int(str(self.llm_max_new.get()).strip())
            except: mx_l = 2048
            mx_l = max(64, min(mx_l, 30000))

            try:    temp = float(str(self.llm_temp.get()).strip())
            except: temp = 0.3
            try:    top_p = float(str(self.llm_top_p.get()).strip())
            except: top_p = 0.9
            try:    top_k = int(str(self.llm_top_k.get()).strip())
            except: top_k = 50
            try:    rep_pen = float(str(self.llm_rep_pen.get()).strip())
            except: rep_pen = 1.12

            runner.run_to_file(
                prompt, out_path,
                max_new=mx_l, timeout_sec=900,
                temp=temp, top_p=top_p, top_k=top_k, repeat_penalty=rep_pen,
            )

            self._post("add_output", out_path)
            self._post("preview_from_file", out_path)
            self._post("log", f"[LLM] saved: {os.path.basename(out_path)}")
        except Exception as e:
            self._post("log", f"[LLM] error: {e}")
        finally:
            setattr(self, "_llm_busy", False)

    threading.Thread(target=_worker, daemon=True).start()

# ---------- Patch onto App ----------
App._dest_base = _dest_base
App._run_pipeline_subproc = _run_pipeline_subproc
App._speaker_id_map = _speaker_id_map
App._format_ts_seconds = _format_ts_seconds
App._merge_by_speaker_word_level = _merge_by_speaker_word_level
App._merge_by_speaker_segment_level = _merge_by_speaker_segment_level
App._fmt_srt = _fmt_srt
App._save_srt = _save_srt
App._save_vtt = _save_vtt
App._worker_batch = _worker_batch
App._run_batch = _run_batch
App._llm_run_over_outputs = _llm_run_over_outputs

# --------------------------- Entrypoint ---------------------------
def _load_icon_if_present(app: tk.Tk):
    try:
        ico = CONTENT / "AppIcon.ico"
        if ico.exists():
            app.iconbitmap(default=str(ico))
        else:
            png = CONTENT / "AppIcon.png"
            if png.exists():
                img = tk.PhotoImage(file=str(png))
                app.iconphoto(True, img)
    except Exception:
        pass

def main():
    # Lazy FFmpeg: don't block UI if missing — playback/decoding paths will surface scoped errors later.
    try:
        _ = ffmpeg_path()
    except SystemExit:
        pass
    except Exception:
        pass

    # Use 'spawn' for safety with PyTorch/whisperx; works on Ubuntu too.
    try:
        mp.set_start_method("spawn", force=True)
        mp.set_executable(sys.executable)
    except Exception:
        pass

    app = App()
    _load_icon_if_present(app)
    app.mainloop()

if __name__ == "__main__":
    try:
        mp.freeze_support()
    except Exception:
        pass

    try:
        main()
    except Exception:
        try:
            from tkinter import messagebox
            import traceback
            messagebox.showerror("Fatal error", traceback.format_exc())
        except Exception:
            import traceback
            print(traceback.format_exc(), file=sys.stderr)

