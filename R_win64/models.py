import os
import sys
import threading
import subprocess
import shlex
import queue
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------- Paths & caches ----------------
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "content" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)

ICON_PATH = APP_ROOT / "content" / "AppIcon.ico"

# ---------------- Model list ----------------
# Added 'markers' for robust "installed" checks.
MODEL_ITEMS = [
    {
        "ui": "Faster-Whisper Large V3 Turbo — Speech-to-Text (fast ASR)",
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "subdir": "mobiuslabsgmbh__faster-whisper-large-v3-turbo",
        "gated": False,
        "include": [],
        "exclude": [],
        "markers": ["model.bin"],  # primary artifact in that repo
        "blurb": "Fast, accurate Whisper variant for transcription."
    },
    {
        "ui": "wav2vec2 Base 960h — ASR backbone/features (Meta)",
        "repo": "facebook/wav2vec2-base-960h",
        "subdir": "facebook__wav2vec2-base-960h",
        "gated": False,
        "include": [
            "pytorch_model.bin", "*.bin",
            "vocab.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json", "feature_extractor_config.json",
            "*.json", "*.txt"
        ],
        "exclude": ["*.safetensors", "*.h5", "*.ckpt"],
        "markers": ["pytorch_model.bin", "tokenizer_config.json"],
        "blurb": "LibriSpeech 960h model; uses .bin weights + tokenizer/config files."
    },
    {
        "ui": "WeSpeaker VoxCeleb ResNet34 + LM — Speaker Embeddings",
        "repo": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "subdir": "pyannote__wespeaker-voxceleb-resnet34-LM",
        "gated": False,
        "include": [],
        "exclude": [],
        "markers": ["pytorch_model.bin", "config.yaml"],
        "blurb": "Speaker representation model (helps tell who is speaking)."
    },
    {
        "ui": "Segmentation 3.0 (pyannote) — GATED",
        "repo": "pyannote/segmentation-3.0",
        "subdir": "pyannote__segmentation-3.0",
        "gated": True,
        "include": [],
        "exclude": [],
        "markers": ["pytorch_model.bin", "config.yaml"],
        "blurb": "Detects speech segments. Requires accepting terms and using a token."
    },
    {
        "ui": "Speaker Diarization 3.1 (pyannote) — GATED",
        "repo": "pyannote/speaker-diarization-3.1",
        "subdir": "pyannote__speaker-diarization-3.1",
        "gated": True,
        "include": [],
        "exclude": [],
        "markers": ["config.yaml"],  # pipeline file
        "blurb": "Who-spoke-when pipeline. Requires access approval and token."
    },
    {
        "ui": "Qwen3-4B Instruct (GGUF Q4_K_M) — Local LLM for notes/QA",
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "subdir": "unsloth__Qwen3-4B-Instruct-2507-GGUF",
        "gated": False,
        "include": ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"],
        "exclude": [],
        "markers": ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"],
        "blurb": "Compact assistant model for llama.cpp in GGUF format."
    }
]

TOP_NOTE = (
    "Some models below (the pyannote ones) are free/open for research & community use but are "
    "gated so the authors can track usage and report impact in grants. Paste your token only if "
    "you select those gated models; everything else downloads anonymously."
)

HELP_STEPS = [
    ("1) Create a free Hugging Face account", "https://huggingface.co/join"),
    ("2) Request access: pyannote/segmentation-3.0", "https://huggingface.co/pyannote/segmentation-3.0"),
    ("2) Request access: pyannote/speaker-diarization-3.1", "https://huggingface.co/pyannote/speaker-diarization-3.1"),
    ("3) Create / view your access tokens", "https://huggingface.co/settings/tokens"),
]

HF_CMD = ["hf"]  # modern CLI

# --------------- Utility: installed check & cleanup ----------------
def is_installed(local_dir: Path, markers: list[str]) -> bool:
    if not local_dir.exists():
        return False
    # If markers are provided, require at least one present (or all? choose pragmatic "any two"?)
    # We'll require ALL listed markers if they are specific filenames (no wildcards).
    for m in markers or []:
        if "*" in m or "?" in m:
            # wildcard marker: any match is fine
            if not list(local_dir.glob(m)):
                return False
        else:
            if not (local_dir / m).exists():
                return False
    # If no markers at all, consider installed if directory has any non-hidden file
    if not markers:
        for p in local_dir.rglob("*"):
            if p.is_file() and p.name not in {".gitignore"}:
                return True
        return False
    return True

def cleanup_model_dir(local_dir: Path):
    # Remove helper folders produced during download
    for p in [local_dir / ".cache", local_dir / "refs"]:
        if p.exists():
            for root, dirs, files in os.walk(p, topdown=False):
                for f in files:
                    try: os.remove(Path(root) / f)
                    except Exception: pass
                for d in dirs:
                    try: os.rmdir(Path(root) / d)
                    except Exception: pass
            try: os.rmdir(p)
            except Exception: pass

# --------------- GUI ----------------
class ModelRow:
    def __init__(self, parent, item, token_var):
        self.item = item
        self.token_var = token_var
        self.var_sel = tk.BooleanVar(value=True)
        self.status = "unknown"
        self.spinner_idx = 0
        self.spinner_running = False

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", padx=8, pady=6)

        # Left: checkbox + name + blurb
        self.chk = ttk.Checkbutton(self.frame, variable=self.var_sel)
        self.chk.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0,8), pady=(2,0))

        name = item["ui"] + ("  🔒" if item["gated"] else "")
        self.lbl_name = ttk.Label(self.frame, text=name, font=("Segoe UI", 10, "bold"))
        self.lbl_name.grid(row=0, column=1, sticky="w")
        self.lbl_blurb = ttk.Label(self.frame, text=item["blurb"], foreground="#555", wraplength=740, justify="left")
        self.lbl_blurb.grid(row=1, column=1, sticky="w", pady=(2,0))

        # Right: status + retry
        self.lbl_status = ttk.Label(self.frame, text="", width=22, anchor="e")
        self.lbl_status.grid(row=0, column=2, sticky="e", padx=(12, 8))
        self.btn_retry = ttk.Button(self.frame, text="Retry", width=10, command=self.retry)
        self.btn_retry.grid(row=1, column=2, sticky="e", padx=(12, 8))
        self.btn_retry.state(["disabled"])

        self.frame.columnconfigure(1, weight=1)

        # compute local dir now
        self.local_dir = MODELS_DIR / item["subdir"]
        self.refresh_status()

    def refresh_status(self):
        if is_installed(self.local_dir, self.item.get("markers", [])):
            self.set_status("installed")
        else:
            self.set_status("not_installed")

    def set_status(self, state: str):
        self.status = state
        colors = {
            "installed": "#1a7f37",
            "not_installed": "#666666",
            "downloading": "#0a53a3",
            "failed": "#b00020",
        }
        texts = {
            "installed": "✓ Installed",
            "not_installed": "Not installed",
            "downloading": "Downloading…",
            "failed": "✗ Failed",
        }
        self.lbl_status.config(text=texts[state], foreground=colors[state])
        if state in ("failed", "not_installed"):
            self.btn_retry.state(["!disabled"])
        else:
            self.btn_retry.state(["disabled"])

    def start_spinner(self, root):
        if self.spinner_running:
            return
        self.spinner_running = True
        sequence = ["–", "\\", "|", "/"]

        def tick():
            if not self.spinner_running:
                return
            self.spinner_idx = (self.spinner_idx + 1) % len(sequence)
            self.lbl_status.config(text=f"Downloading… {sequence[self.spinner_idx]}")
            root.after(150, tick)

        tick()

    def stop_spinner(self):
        self.spinner_running = False

    def retry(self):
        # Enqueue a single-item download via the parent app
        self.frame.event_generate("<<RetryRequested>>", when="tail")

class ModelDownloaderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcribe Offline — Model Downloader")
        try:
            if ICON_PATH.exists():
                self.iconbitmap(default=str(ICON_PATH))
        except Exception:
            pass

        # Fixed layout so footer is always visible
        self.geometry("1060x720")
        self.minsize(1060, 720)
        self.configure(padx=10, pady=8)

        self.queue = queue.Queue()
        self.downloading = False

        self._build_header()
        self._build_notice_and_links()
        self._build_token()
        self._build_model_list()
        self._build_actions()
        self._build_footer()

        self._ui_set_status("Ready.")
        self.bind("<<RetryRequested>>", self._handle_retry_event)

    # ---- UI sections
    def _build_header(self):
        frm = ttk.Frame(self); frm.pack(fill="x", pady=(4, 8))
        left = ttk.Frame(frm); left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="Download Required Models", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(left, text=f"Destination: {MODELS_DIR}", foreground="#444").pack(anchor="w", pady=(3,0))

    def _build_notice_and_links(self):
        box = ttk.LabelFrame(self, text="Before you start")
        box.pack(fill="x", pady=6)
        ttk.Label(box, text=TOP_NOTE, wraplength=1000, justify="left").pack(anchor="w", padx=10, pady=(8,6))

        links = ttk.Frame(box); links.pack(fill="x", padx=6, pady=(0,8))
        for text, url in HELP_STEPS:
            lbl = ttk.Label(links, text=f"• {text}", foreground="#0a53a3", cursor="hand2")
            lbl.pack(anchor="w", padx=8, pady=2)
            lbl.bind("<Button-1>", lambda e, u=url: os.startfile(u) if os.name == "nt" else webbrowser.open(u))

    def _build_token(self):
        frm = ttk.Frame(self); frm.pack(fill="x", pady=(2, 8))
        ttk.Label(frm, text="Hugging Face Token (only for 🔒 gated models):").grid(row=0, column=0, sticky="w")
        self.token_var = tk.StringVar()
        self._token_entry = ttk.Entry(frm, textvariable=self.token_var, width=58, show="•")
        self._token_entry.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self._show_token = tk.BooleanVar(value=False)

        def toggle_show():
            self._token_entry.config(show="" if self._show_token.get() else "•")
        ttk.Checkbutton(frm, text="Show", variable=self._show_token, command=toggle_show)\
            .grid(row=0, column=2, sticky="w", padx=8)

    def _build_model_list(self):
        box = ttk.LabelFrame(self, text="Models")
        box.pack(fill="both", pady=6)

        self.rows: list[ModelRow] = []
        for item in MODEL_ITEMS:
            row = ModelRow(box, item, self.token_var)
            self.rows.append(row)

    def _build_actions(self):
        frm = ttk.Frame(self); frm.pack(fill="x", pady=6)
        self.btn_download = ttk.Button(frm, text="Download selected", command=self.start_downloads)
        self.btn_download.pack(side="left")
        ttk.Button(frm, text="Close", command=self.destroy).pack(side="right")

    def _build_footer(self):
        frm = ttk.Frame(self); frm.pack(fill="x", pady=(4, 0))
        ttk.Label(
            frm,
            text="After downloads, start the app from RStudio using: run_transcribe_offline.R",
            foreground="#444"
        ).pack(anchor="w")

    # ---- status helpers
    def _ui_set_status(self, msg):
        self.title(f"Transcribe Offline — Model Downloader   [{msg}]")

    def _handle_retry_event(self, _):
        # Find which row asked for retry by focus traversal
        w = self.focus_get()
        for row in self.rows:
            if row.btn_retry == w or row.frame == w or row.lbl_status == w:
                self._download_one(row)
                break

    # ---- main actions
    def start_downloads(self):
        if self.downloading:
            return
        selected_rows = [r for r in self.rows if r.var_sel.get()]
        if not selected_rows:
            messagebox.showinfo("Nothing selected", "Please select at least one model to download.")
            return
        self.downloading = True
        self.btn_download.state(["disabled"])
        self._ui_set_status("Downloading…")
        threading.Thread(target=self._download_batch, args=(selected_rows,), daemon=True).start()

    def _download_batch(self, rows):
        for r in rows:
            ok = self._download_one(r, in_batch=True)
            # continue through failures; each row shows its own state
        # batch finished
        self.queue.put(("batch_done", None))

    def _download_one(self, row: ModelRow, in_batch: bool = False) -> bool:
        item = row.item
        local_dir = row.local_dir
        local_dir.mkdir(parents=True, exist_ok=True)

        # Token checks for gated models
        token = self.token_var.get().strip()
        if item["gated"] and not token:
            self.queue.put(("popup", ("Token required",
                                      f"'{item['ui']}' is gated (pyannote). Paste your Hugging Face token first.")))
            row.set_status("not_installed")
            return False

        # Build hf download command
        args = ["download", item["repo"], "--revision", "main", "--local-dir", str(local_dir)]
        for inc in item.get("include", []) or []:
            args.extend(["--include", inc])
        for exc in item.get("exclude", []) or []:
            args.extend(["--exclude", exc])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if item["gated"]:
            env["HF_TOKEN"] = token
            env["HUGGINGFACE_HUB_TOKEN"] = token

        # Update UI to downloading with spinner
        self.queue.put(("downloading_start", row))

        # Run process (no log UI; just capture text for error heuristics)
        try:
            proc = subprocess.Popen(
                ["hf"] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
                env=env
            )
        except FileNotFoundError:
            self.queue.put(("popup", ("Hugging Face CLI not found",
                                      "The 'hf' CLI was not found on PATH.\n\n"
                                      "Install/upgrade into this Python:\n  python -m pip install --upgrade huggingface_hub")))
            self.queue.put(("downloading_end", (row, False)))
            return False

        output_buf = []
        # Read output without flooding UI (we're not showing it live)
        for line in iter(proc.stdout.readline, ""):
            if line:
                output_buf.append(line)
        proc.stdout.close()
        ret = proc.wait()

        # Stop spinner
        self.queue.put(("downloading_end", (row, ret == 0)))

        if ret == 0:
            # Cleanup helpers and verify installation
            cleanup_model_dir(local_dir)
            ok = is_installed(local_dir, item.get("markers", []))
            if not ok:
                # Rare: downloaded but markers missing — treat as failure with hint
                self.queue.put(("popup", ("Verification issue",
                                          f"'{item['ui']}' downloaded but expected files were not found.\n"
                                          f"Check folder:\n{local_dir}")))
                self.queue.put(("set_failed", row))
                return False
            self.queue.put(("set_installed", row))
            return True

        # Non-zero return: show reasoned popups for gated models
        out = "\n".join(output_buf[-40:])  # tail for context
        if item["gated"] and any(err in out for err in ["401", "403", "Unauthorized", "permission", "access"]):
            self.queue.put(("popup", ("Access denied",
                                      f"Access/token problem while downloading:\n\n{item['ui']}\n\n"
                                      "• Make sure your token has access to the repo\n"
                                      "• Ensure you've requested access on the model page\n"
                                      "• Paste the token in the field above and try again.\n\n"
                                      f"Repo: {item['repo']}")))
        else:
            self.queue.put(("popup", ("Download failed",
                                      f"'{item['ui']}' failed to download.\n\n"
                                      f"Command:\n{' '.join(shlex.quote(a) for a in ['hf'] + args)}\n\n"
                                      f"Last output:\n{out}")))
        self.queue.put(("set_failed", row))
        return False

    # ---- queue-driven UI updates
    def process_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "downloading_start":
                    row: ModelRow = payload
                    row.set_status("downloading")
                    row.start_spinner(self)
                elif kind == "downloading_end":
                    row, success = payload
                    row.stop_spinner()
                    # status will be finalized by following set_installed/set_failed events
                elif kind == "set_installed":
                    payload.set_status("installed")
                elif kind == "set_failed":
                    payload.set_status("failed")
                elif kind == "popup":
                    title, msg = payload
                    messagebox.showwarning(title, msg)
                elif kind == "batch_done":
                    self.downloading = False
                    self.btn_download.state(["!disabled"])
                    self._ui_set_status("Done.")
                    # Refresh all rows’ installed states in case user downloaded externally too
                    for r in self.rows:
                        r.refresh_status()
                self.queue.task_done()
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

    def mainloop(self, n=0):
        # start queue pump
        self.after(100, self.process_queue)
        super().mainloop()


def main():
    # Optional: warn in console if hf missing
    try:
        subprocess.run(["hf", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except Exception:
        print("Warning: 'hf' CLI not found. Install with:")
        print("  python -m pip install --upgrade huggingface_hub")
    app = ModelDownloaderGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
