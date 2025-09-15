import os
import sys
import threading
import subprocess
import shlex
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser

# ---------------- Paths & caches ----------------
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "content" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)

ICON_PATH = APP_ROOT / "content" / "AppIcon.ico"

# ---------------- Model list ----------------
MODEL_ITEMS = [
    {
        "ui": "Faster-Whisper Large V3 Turbo — Speech-to-Text (fast ASR)",
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "subdir": "mobiuslabsgmbh__faster-whisper-large-v3-turbo",
        "gated": False,
        "include": [],
        "exclude": [],
        "markers": ["model.bin"],
        "blurb": "Fast, accurate Whisper variant for transcription."
    },
    {
        "ui": "Wav2Vec2 Base 960h — word-level alignment (English only) (Meta)",
        "repo": "facebook/wav2vec2-base-960h",
        "subdir": "facebook__wav2vec2-base-960h",
        "gated": False,
        "include": [],  # no include filters
        "exclude": ["*.safetensors", "*.h5"],  # only exclude these
        "markers": ["pytorch_model.bin"],  # simple & reliable
        "blurb": "Model used for word-level alignment (English)."
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
        "markers": ["config.yaml"],
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
def is_installed(local_dir: Path, markers):
    if not local_dir.exists():
        return False
    for m in markers or []:
        if "*" in m or "?" in m:
            if not list(local_dir.glob(m)):
                return False
        else:
            if not (local_dir / m).exists():
                return False
    if not markers:
        for p in local_dir.rglob("*"):
            if p.is_file() and p.name not in {".gitignore"}:
                return True
        return False
    return True

def cleanup_model_dir(local_dir: Path):
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

# --------------- GUI components ----------------
class ModelRow:
    def __init__(self, parent, item, token_var, on_retry):
        self.item = item
        self.token_var = token_var
        self.on_retry = on_retry
        self.var_sel = tk.BooleanVar(value=True)
        self.status = "unknown"
        self.spinner_idx = 0
        self.spinner_running = False

        self.frame = ttk.Frame(parent)
        self.frame.pack(fill="x", padx=8, pady=6)

        self.chk = ttk.Checkbutton(self.frame, variable=self.var_sel)
        self.chk.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0,8), pady=(2,0))

        name = item["ui"] + ("  🔒" if item["gated"] else "")
        self.lbl_name = ttk.Label(self.frame, text=name, font=("Segoe UI", 10, "bold"))
        self.lbl_name.grid(row=0, column=1, sticky="w")
        self.lbl_blurb = ttk.Label(self.frame, text=item["blurb"], foreground="#555", wraplength=740, justify="left")
        self.lbl_blurb.grid(row=1, column=1, sticky="w", pady=(2,0))

        self.lbl_status = ttk.Label(self.frame, text="", width=22, anchor="e")
        self.lbl_status.grid(row=0, column=2, sticky="e", padx=(12, 8))
        self.btn_retry = ttk.Button(self.frame, text="Retry", width=10, command=lambda: self.on_retry(self))
        self.btn_retry.grid(row=1, column=2, sticky="e", padx=(12, 8))
        self.btn_retry.state(["disabled"])

        self.frame.columnconfigure(1, weight=1)

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

class ModelDownloaderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcribe Offline — Model Downloader")
        try:
            if ICON_PATH.exists():
                self.iconbitmap(default=str(ICON_PATH))
        except Exception:
            pass

        self.geometry("1060x720")
        self.minsize(1060, 720)
        self.configure(padx=10, pady=8)

        self.downloading = False

        self._build_header()
        self._build_notice_and_links()
        self._build_token()
        self._build_model_list()
        self._build_actions()
        self._build_footer()

        self._ui_set_status("Ready.")

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
            def opener(u=url):
                if os.name == "nt":
                    os.startfile(u)
                else:
                    webbrowser.open(u)
            lbl = ttk.Label(links, text=f"• {text}", foreground="#0a53a3", cursor="hand2")
            lbl.pack(anchor="w", padx=8, pady=2)
            lbl.bind("<Button-1>", lambda e, fn=opener: fn())

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
        self.rows = []
        for item in MODEL_ITEMS:
            row = ModelRow(box, item, self.token_var, on_retry=self._download_one_async)
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

    def _ui_set_status(self, msg):
        self.title(f"Transcribe Offline — Model Downloader   [{msg}]")

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
            self._download_one(r)
        self.downloading = False
        self.btn_download.state(["!disabled"])
        self._ui_set_status("Done.")
        for r in self.rows:
            r.refresh_status()

    def _download_one_async(self, row):
        if self.downloading:
            return
        self.downloading = True
        self.btn_download.state(["disabled"])
        threading.Thread(target=self._download_one_then_unlock, args=(row,), daemon=True).start()

    def _download_one_then_unlock(self, row):
        self._download_one(row)
        self.downloading = False
        self.btn_download.state(["!disabled"])
        self._ui_set_status("Ready")
        row.refresh_status()

    def _download_one(self, row):
        item = row.item
        local_dir = row.local_dir
        local_dir.mkdir(parents=True, exist_ok=True)

        token = self.token_var.get().strip()
        if item["gated"] and not token:
            messagebox.showwarning("Token required",
                                   f"'{item['ui']}' is gated (pyannote). Paste your Hugging Face token first.")
            row.set_status("not_installed")
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if item["gated"]:
            env["HF_TOKEN"] = token
            env["HUGGINGFACE_HUB_TOKEN"] = token

        # Build a single 'hf download' call using the item's include/exclude settings
        args = ["download", item["repo"], "--revision", "main", "--local-dir", str(local_dir)]
        for inc in item.get("include", []) or []:
            args.extend(["--include", inc])
        for exc in item.get("exclude", []) or []:
            args.extend(["--exclude", exc])

        row.set_status("downloading")
        row.start_spinner(self)

        def run_hf(cmd_args):
            try:
                proc = subprocess.Popen(
                    ["hf"] + cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env
                )
            except FileNotFoundError:
                messagebox.showerror(
                    "Hugging Face CLI not found",
                    "The 'hf' CLI was not found on PATH.\n\n"
                    "Install/upgrade into this Python:\n  python -m pip install --upgrade huggingface_hub"
                )
                return 127, ""
            out = []
            for line in iter(proc.stdout.readline, ""):
                if line:
                    out.append(line)
            proc.stdout.close()
            ret = proc.wait()
            return ret, "".join(out[-80:])

        ret, tail = run_hf(args)

        row.stop_spinner()
        if ret == 0:
            cleanup_model_dir(local_dir)
            if is_installed(local_dir, item.get("markers", [])):
                row.set_status("installed")
            else:
                messagebox.showwarning(
                    "Verification issue",
                    f"'{item['ui']}' downloaded but expected files were not found.\n\n"
                    f"Check folder:\n{local_dir}"
                )
                row.set_status("failed")
        else:
            if item["gated"] and any(k in tail for k in ["401", "403", "Unauthorized", "permission", "access"]):
                messagebox.showwarning(
                    "Access denied",
                    f"Access/token problem while downloading:\n\n{item['ui']}\n\n"
                    "• Make sure your token has access to the repo\n"
                    "• Ensure you've requested access on the model page\n"
                    "• Paste the token in the field above and try again.\n\n"
                    f"Repo: {item['repo']}"
                )
            else:
                messagebox.showwarning(
                    "Download failed",
                    f"'{item['ui']}' failed to download.\n\n"
                    f"Command:\n{' '.join(shlex.quote(a) for a in ['hf'] + args)}\n\n"
                    f"Last output:\n{tail}"
                )
            row.set_status("failed")


def main():
    try:
        subprocess.run(["hf", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except Exception:
        print("Warning: 'hf' CLI not found. Install with:")
        print("  python -m pip install --upgrade huggingface_hub")
    app = ModelDownloaderGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
