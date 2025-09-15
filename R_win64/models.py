import os
import sys
import threading
import subprocess
import shlex
import queue
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------- Paths: put models under content/models (relative to this file) ----------------
APP_ROOT = Path(__file__).resolve().parent
MODELS_DIR = APP_ROOT / "content" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Keep caches inside the project so everything is self-contained
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)

# Try to use your app icon if present
ICON_PATH = APP_ROOT / "content" / "AppIcon.ico"

# ---------------- Model list (checkbox items) ----------------
# Each entry: ui_name, repo, local_subdir, gated(bool), include(list), exclude(list), blurb
MODEL_ITEMS = [
    {
        "ui": "Faster-Whisper Large V3 Turbo — Speech-to-Text (fast ASR)",
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "subdir": "mobiuslabsgmbh__faster-whisper-large-v3-turbo",
        "gated": False,
        "include": [],
        "exclude": [],
        "blurb": "Fast, accurate Whisper variant for transcription."
    },
    {
        "ui": "wav2vec2 Base 960h — ASR backbone/features (Meta)",
        "repo": "facebook/wav2vec2-base-960h",
        "subdir": "facebook__wav2vec2-base-960h",
        "gated": False,
        # Grab classic PyTorch .bin + tokenizer/config files (skip other weight formats)
        "include": [
            "pytorch_model.bin", "*.bin",
            "vocab.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json", "feature_extractor_config.json",
            "*.json", "*.txt"
        ],
        "exclude": ["*.safetensors", "*.h5", "*.ckpt"],
        "blurb": "LibriSpeech 960h model; downloads .bin weights + tokenizer/config files."
    },
    {
        "ui": "WeSpeaker VoxCeleb ResNet34 + LM — Speaker Embeddings",
        "repo": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "subdir": "pyannote__wespeaker-voxceleb-resnet34-LM",
        "gated": False,
        "include": [],
        "exclude": [],
        "blurb": "Speaker representation model (helps tell who is speaking)."
    },
    {
        "ui": "Segmentation 3.0 (pyannote) — GATED",
        "repo": "pyannote/segmentation-3.0",
        "subdir": "pyannote__segmentation-3.0",
        "gated": True,
        "include": [],
        "exclude": [],
        "blurb": "Detects speech segments. Requires accepting terms and using a token."
    },
    {
        "ui": "Speaker Diarization 3.1 (pyannote) — GATED",
        "repo": "pyannote/speaker-diarization-3.1",
        "subdir": "pyannote__speaker-diarization-3.1",
        "gated": True,
        "include": [],
        "exclude": [],
        "blurb": "Who-spoke-when pipeline. Requires access approval and token."
    },
    {
        "ui": "Qwen3-4B Instruct (GGUF Q4_K_M) — Local LLM for notes/QA",
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "subdir": "unsloth__Qwen3-4B-Instruct-2507-GGUF",
        "gated": False,
        "include": ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"],
        "exclude": [],
        "blurb": "Compact assistant model for llama.cpp in GGUF format."
    }
]

# ---------------- Help text & links ----------------
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

# ---------------- Use the modern Hugging Face CLI ----------------
HF_CMD = ["hf"]  # assumes 'hf' is available on PATH (huggingface_hub installed)

# ---------------- GUI ----------------
class ModelDownloaderGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Transcribe Offline — Model Downloader")
        try:
            if ICON_PATH.exists():
                self.iconbitmap(default=str(ICON_PATH))
        except Exception:
            pass

        # Fixed window that shows footer without maximizing
        self.geometry("1024x760")
        self.minsize(1024, 760)  # stable layout regardless of screen DPI
        self.configure(padx=10, pady=8)

        self.queue = queue.Queue()
        self.downloading = False
        self._blank_count = 0
        self._heartbeat_enabled = False

        self._build_header()
        self._build_notice_and_links()
        self._build_token()
        self._build_model_list()
        self._build_actions()
        self._build_log()
        self._build_footer()

        self._ui_set_status("Ready.")
        self.after(80, self._poll_queue)

    # --- UI builders ---
    def _open_url(self, url):
        webbrowser.open(url, new=2)

    def _build_header(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", pady=(4, 8))
        left = ttk.Frame(frm); left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="Download Required Models", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(left, text=f"Destination: {MODELS_DIR}", foreground="#444").pack(anchor="w", pady=(3,0))

    def _build_notice_and_links(self):
        box = ttk.LabelFrame(self, text="Before you start")
        box.pack(fill="x", pady=6)
        ttk.Label(box, text=TOP_NOTE, wraplength=980, justify="left").pack(anchor="w", padx=10, pady=(8,6))

        links = ttk.Frame(box); links.pack(fill="x", padx=6, pady=(0,8))
        for text, url in HELP_STEPS:
            lbl = ttk.Label(links, text=f"• {text}", foreground="#0a53a3", cursor="hand2")
            lbl.pack(anchor="w", padx=8, pady=2)
            lbl.bind("<Button-1>", lambda e, u=url: self._open_url(u))

    def _build_token(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", pady=(2, 8))
        ttk.Label(frm, text="Hugging Face Token (only for 🔒 gated models):").grid(row=0, column=0, sticky="w")
        self.token_var = tk.StringVar()
        self._token_entry = ttk.Entry(frm, textvariable=self.token_var, width=58)
        self._token_entry.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self._show_token = tk.BooleanVar(value=False)

        def toggle_show():
            self._token_entry.config(show="" if self._show_token.get() else "•")
        ttk.Checkbutton(frm, text="Show", variable=self._show_token, command=toggle_show)\
            .grid(row=0, column=2, sticky="w", padx=8)
        self._token_entry.config(show="•")

    def _build_model_list(self):
        box = ttk.LabelFrame(self, text="Models to install")
        box.pack(fill="x", pady=6)

        # Scrollable area for model items — fixed height so footer is always visible
        canvas = tk.Canvas(box, height=200, borderwidth=0, highlightthickness=0)
        vbar = ttk.Scrollbar(box, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.pack(side="left", fill="x", expand=True)
        vbar.pack(side="right", fill="y")

        self.vars = []
        for item in MODEL_ITEMS:
            row = ttk.Frame(inner)
            row.pack(fill="x", padx=8, pady=6)

            var = tk.BooleanVar(value=True)
            self.vars.append(var)
            ttk.Checkbutton(row, variable=var).grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0,8), pady=(2,0))

            name = item["ui"] + ("  🔒" if item["gated"] else "")
            ttk.Label(row, text=name, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w")
            ttk.Label(row, text=item["blurb"], foreground="#555", wraplength=880, justify="left")\
                .grid(row=1, column=1, sticky="w", pady=(2,0))

    def _build_actions(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", pady=6)
        self.btn_download = ttk.Button(frm, text="Start Download", command=self.start_downloads)
        self.btn_download.pack(side="left")
        ttk.Button(frm, text="Close", command=self.destroy).pack(side="right")

    def _build_log(self):
        box = ttk.LabelFrame(self, text="Log (live output from hf / CLI)")
        box.pack(fill="both", expand=True, pady=(0, 6))

        # Fixed-height log so footer stays visible
        self.txt = tk.Text(box, wrap="none", height=18)
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)
        self._log_banner()

    def _build_footer(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", pady=(0, 6))
        ttk.Label(frm, text="After downloads, start the app from RStudio by sourcing:", foreground="#444")\
            .pack(anchor="w")
        launch_cmd = "source(file.path('~','Downloads','Transcribe_Offline','run_transcribe_offline.R'))"
        self.launch_cmd = launch_cmd
        ttk.Label(frm, text=launch_cmd, font=("Consolas", 9)).pack(anchor="w", pady=(2, 6))

        btns = ttk.Frame(frm); btns.pack(fill="x")
        ttk.Button(btns, text="Copy launch command", command=self._copy_launch_command).pack(side="left")
        ttk.Button(btns, text="Open project folder", command=lambda: os.startfile(str(APP_ROOT))).pack(side="left", padx=8)
        ttk.Button(btns, text="Launch app now (Windows)", command=self._launch_runner_now).pack(side="left", padx=8)

    # --- helpers ---
    def _copy_launch_command(self):
        self.clipboard_clear()
        self.clipboard_append(self.launch_cmd)
        self.update()
        messagebox.showinfo("Copied", "Launch command copied to clipboard.")

    def _launch_runner_now(self):
        try:
            runner = APP_ROOT / "run_transcribe_offline.R"
            if not runner.exists():
                messagebox.showerror("Not found", f"Could not find:\n{runner}\nRun setup to generate it.")
                return
            os.startfile(str(runner))  # Windows: open with default .R handler (usually RStudio)
        except Exception as e:
            messagebox.showerror("Error launching", f"Could not launch the runner:\n{e}")

    def _log_banner(self):
        self.txt.insert("end",
            "This window streams the raw Hugging Face CLI output (plus a heartbeat during large files).\n"
            f"Destination folder: {MODELS_DIR}\n\n")
        self.txt.see("end")

    def _ui_set_status(self, msg):
        self.title(f"Transcribe Offline — Model Downloader   [{msg}]")

    def _log_replace_last_line(self, new_text):
        last_index = self.txt.index("end-1c linestart")
        self.txt.delete(last_index, "end-1c")
        self.txt.insert("end", new_text)

    def _append_line(self, text: str):
        # Collapse excessive blank lines to shorten gaps
        if text.strip() == "":
            if self._blank_count >= 1:
                return
            self._blank_count += 1
        else:
            self._blank_count = 0
        self.txt.insert("end", text + "\n")
        self.txt.see("end")

    def log(self, text):
        # Handle \r-updating lines from tqdm/CLI: keep last line updating in place
        if "\r" in text:
            seg = text.split("\r")[-1]
            if "\n" in seg:
                parts = seg.split("\n")
                for p in parts[:-1]:
                    self._append_line(p)
                self._log_replace_last_line(parts[-1])
                self.txt.insert("end", "\n")
            else:
                self._log_replace_last_line(seg)
        else:
            self._append_line(text)

    # --- actions ---
    def start_downloads(self):
        if self.downloading:
            return
        selected = [(item, var.get()) for item, var in zip(MODEL_ITEMS, self.vars) if var.get()]
        if not selected:
            messagebox.showinfo("Nothing selected", "Please select at least one model to download.")
            return

        need_token = any(item["gated"] for item, _ in selected)
        token = self.token_var.get().strip()
        if need_token and not token:
            messagebox.showwarning("Token required",
                                   "You selected gated models (pyannote). Paste your Hugging Face token first.")
            return

        # quick sanity check for 'hf'
        try:
            subprocess.run(HF_CMD + ["--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except Exception:
            messagebox.showerror(
                "Hugging Face CLI not found",
                "The 'hf' CLI was not found on PATH.\n\n"
                "Install/upgrade into this Python:\n  python -m pip install --upgrade huggingface_hub"
            )
            return

        self.downloading = True
        self.btn_download.config(state="disabled")
        self._ui_set_status("Downloading…")
        self.log("Starting downloads…")
        threading.Thread(target=self._worker_download, args=(selected, token), daemon=True).start()

    def _worker_download(self, selected, token):
        for item, _ in selected:
            try:
                self.queue.put(("log", f"\n=== {item['ui']} ==="))
                local_dir = MODELS_DIR / (item["subdir"] if item["subdir"] != "." else "")
                local_dir.mkdir(parents=True, exist_ok=True)

                # Build 'hf download' args (no --local-dir-use-symlinks in new CLI)
                args = [
                    "download",
                    item["repo"],
                    "--revision", "main",
                    "--local-dir", str(local_dir)
                ]
                for inc in item.get("include", []) or []:
                    args.extend(["--include", inc])
                for exc in item.get("exclude", []) or []:
                    args.extend(["--exclude", exc])

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"  # help streaming
                if item["gated"]:
                    env["HF_TOKEN"] = token
                    env["HUGGINGFACE_HUB_TOKEN"] = token

                self._run_cli_stream(args, env)

                # Clean up extra control folders produced by 'hf download' inside this model dir
                self._cleanup_local_dir(local_dir)

                self.queue.put(("log", f"✓ Done: {item['ui']}"))

            except Exception as e:
                self.queue.put(("log", f"✗ Failed: {item['ui']} — {e}"))

        self.queue.put(("done", None))

    def _cleanup_local_dir(self, local_dir: Path):
        # Remove hf helper dirs if present; keep model files
        for p in [local_dir / ".cache", local_dir / "refs"]:
            try:
                if p.exists():
                    # recursive remove
                    for root, dirs, files in os.walk(p, topdown=False):
                        for f in files:
                            try: os.remove(Path(root) / f)
                            except Exception: pass
                        for d in dirs:
                            try: os.rmdir(Path(root) / d)
                            except Exception: pass
                    try: os.rmdir(p)
                    except Exception: pass
            except Exception:
                pass

    def _run_cli_stream(self, args, env):
        # Stream lines as they arrive and add a heartbeat during long silences.
        cmd = HF_CMD + args  # uses 'hf download'
        self.queue.put(("log", "Running: " + " ".join(shlex.quote(a) for a in cmd)))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,                   # line-buffer if possible
            universal_newlines=True,     # text mode
            encoding="utf-8",
            errors="replace",
            env=env
        )

        last_ping = time.time()
        self._heartbeat_enabled = True

        def reader():
            nonlocal last_ping
            for line in iter(proc.stdout.readline, ''):
                last_ping = time.time()
                self.queue.put(("raw", line.rstrip("\n")))
            proc.stdout.close()

        def heartbeat():
            # If no output for 0.7s, print a dot on the same line to show liveness
            spinner_line = ""
            while self._heartbeat_enabled and proc.poll() is None:
                if time.time() - last_ping > 0.7:
                    spinner_line += "."
                    if len(spinner_line) > 40:
                        spinner_line = "."
                    self.queue.put(("heartbeat", spinner_line))
                    last_ping = time.time()
                time.sleep(0.25)

        t_reader = threading.Thread(target=reader, daemon=True)
        t_beat = threading.Thread(target=heartbeat, daemon=True)
        t_reader.start(); t_beat.start()
        ret = proc.wait()
        self._heartbeat_enabled = False
        t_reader.join(timeout=1.0)
        t_beat.join(timeout=1.0)
        if ret != 0:
            raise RuntimeError(f"hf exited with code {ret}")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "raw":
                    self.log(payload)
                elif kind == "heartbeat":
                    # overwrite last line with spinner/dots (no extra blank lines)
                    self._log_replace_last_line(payload)
                elif kind == "done":
                    self.downloading = False
                    self.btn_download.config(state="normal")
                    self._ui_set_status("Done.")
                    self.log("\nAll selected downloads finished.")
                self.queue.task_done()
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)


def main():
    # Optional: warn in console if hf missing
    try:
        subprocess.run(HF_CMD + ["--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except Exception:
        print("Warning: 'hf' CLI not found. Install with:")
        print("  python -m pip install --upgrade huggingface_hub")
    app = ModelDownloaderGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
