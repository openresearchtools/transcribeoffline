import os
import sys
import threading
import subprocess
import shlex
import queue
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

# ---------------- Model list (checkbox items) ----------------
# Each entry: ui_name, repo, local_subdir, gated(bool), include(list), exclude(list), blurb
MODEL_ITEMS = [
    {
        "ui": "Faster-Whisper Large V3 Turbo (speech-to-text, fast ASR)",
        "repo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "subdir": "mobiuslabsgmbh__faster-whisper-large-v3-turbo",
        "gated": False,
        "include": [],
        "exclude": [],
        "blurb": "High-speed Whisper variant for accurate, fast transcription."
    },
    {
        "ui": "wav2vec2 Base 960h (ASR backbone/features)",
        "repo": "facebook/wav2vec2-base-960h",
        "subdir": "facebook__wav2vec2-base-960h",
        "gated": False,
        # Grab classic PyTorch bin + tokenizer/config files
        "include": [
            "pytorch_model.bin", "*.bin",
            "vocab.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json", "feature_extractor_config.json",
            "*.json", "*.txt"
        ],
        # Explicitly avoid alternative weight formats present in the repo
        "exclude": ["*.safetensors", "*.h5", "*.ckpt"],
        "blurb": "LibriSpeech 960h ASR model using .bin weights and tokenizer/config files."
    },
    {
        "ui": "WeSpeaker VoxCeleb ResNet34 + LM (speaker embeddings)",
        "repo": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "subdir": "pyannote__wespeaker-voxceleb-resnet34-LM",
        "gated": False,
        "include": [],
        "exclude": [],
        "blurb": "Speaker representation model; helps recognize who is speaking."
    },
    {
        "ui": "Segmentation 3.0 (pyannote) — GATED",
        "repo": "pyannote/segmentation-3.0",
        "subdir": "pyannote__segmentation-3.0",
        "gated": True,
        "include": [],
        "exclude": [],
        "blurb": "Detects speech segments; requires accepting terms (token)."
    },
    {
        "ui": "Speaker Diarization 3.1 (pyannote) — GATED",
        "repo": "pyannote/speaker-diarization-3.1",
        "subdir": "pyannote__speaker-diarization-3.1",
        "gated": True,
        "include": [],
        "exclude": [],
        "blurb": "Who-spoke-when pipeline; requires access approval (token)."
    },
    {
        "ui": "Qwen3-4B Instruct (GGUF Q4_K_M) (LLM for notes/QA)",
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "subdir": "unsloth__Qwen3-4B-Instruct-2507-GGUF",
        "gated": False,
        "include": ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"],
        "exclude": [],
        "blurb": "Compact local assistant model for llama.cpp (GGUF)."
    },
    {
        "ui": "Silero VAD (voice activity detector)",
        "repo": "snakers4/silero-vad",
        "subdir": ".",  # will place silero_vad.jit directly in content/models
        "gated": False,
        "include": ["silero_vad.jit"],
        "exclude": [],
        "blurb": "Lightweight VAD to detect speech vs. silence."
    },
]

# ---------------- Help links & explanations ----------------
HELP_LINKS = [
    ("Why some models are gated (access tokens)",
     "https://huggingface.co/docs/hub/models-gated"),
    ("Create a free Hugging Face account",
     "https://huggingface.co/join"),
    ("Create / view your access tokens",
     "https://huggingface.co/settings/tokens"),
    ("Request access: pyannote/segmentation-3.0",
     "https://huggingface.co/pyannote/segmentation-3.0"),
    ("Request access: pyannote/speaker-diarization-3.1",
     "https://huggingface.co/pyannote/speaker-diarization-3.1"),
]

# ---------------- CLI detection ----------------
def find_hf_cli():
    """
    Returns (cmd_list, is_python_module) for invoking the Hugging Face CLI.
    Tries huggingface-cli executable, else python -m huggingface_hub.cli .
    """
    candidates = []

    # direct executable on PATH
    candidates.append(("huggingface-cli", False))

    # same dir as current python (Windows Scripts)
    py = Path(sys.executable)
    scripts_dir = py.parent
    cand1 = scripts_dir / "huggingface-cli.exe"
    cand2 = scripts_dir / "huggingface-cli"
    if cand1.exists():
        candidates.insert(0, (str(cand1), False))
    elif cand2.exists():
        candidates.insert(0, (str(cand2), False))

    # python -m fallback
    candidates.append(([sys.executable, "-m", "huggingface_hub.cli"], True))

    for cand, is_mod in candidates:
        try:
            test_cmd = (cand + ["--help"]) if isinstance(cand, list) else [cand, "--help"]
            subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return (cand if isinstance(cand, list) else [cand], is_mod)
        except Exception:
            continue
    return (None, False)

HF_CMD, HF_CMD_IS_MOD = find_hf_cli()

# ---------------- GUI ----------------
class ModelDownloaderGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcribe Offline — Model Downloader")
        self.geometry("910x640")
        self.minsize(880, 580)

        self.queue = queue.Queue()
        self.downloading = False

        self._build_header()
        self._build_links()
        self._build_token_and_dest()
        self._build_model_list()
        self._build_actions()
        self._build_log()

        self._ui_set_status("Ready.")
        self.after(100, self._poll_queue)

    # --- UI builders ---
    def _open_url(self, url):
        webbrowser.open(url, new=2)

    def _build_header(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", padx=12, pady=(10, 4))
        title = ttk.Label(frm, text="Download Required Models", font=("Segoe UI", 14, "bold"))
        title.pack(side="left")
        ttk.Label(frm, text=f"Destination: {MODELS_DIR}", foreground="#444").pack(side="right")

    def _build_links(self):
        box = ttk.LabelFrame(self, text="Why some models are gated & where to get a token")
        box.pack(fill="x", padx=12, pady=6)
        for text, url in HELP_LINKS:
            l = ttk.Label(box, text=f"• {text}", foreground="#0a53a3", cursor="hand2")
            l.pack(anchor="w", padx=8, pady=2)
            l.bind("<Button-1>", lambda e, u=url: self._open_url(u))
        desc = (
            "Some models (the pyannote ones below) are free and open for research/community use, "
            "but are gated so the authors can understand usage and report impact in future grants. "
            "You only need a token for those gated models. Everything else downloads anonymously."
        )
        ttk.Label(box, text=desc, wraplength=860, justify="left").pack(anchor="w", padx=8, pady=(6, 4))

    def _build_token_and_dest(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", padx=12, pady=(2, 6))
        ttk.Label(frm, text="Hugging Face Token (used only for gated models):").grid(row=0, column=0, sticky="w")
        self.token_var = tk.StringVar()
        token_entry = ttk.Entry(frm, textvariable=self.token_var, width=60)
        token_entry.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.show_token = tk.BooleanVar(value=False)
        def toggle_show():
            token_entry.config(show="" if self.show_token.get() else "•")
        ttk.Checkbutton(frm, text="Show", variable=self.show_token, command=toggle_show)\
            .grid(row=0, column=2, sticky="w", padx=8)
        token_entry.config(show="•")

    def _build_model_list(self):
        box = ttk.LabelFrame(self, text="Models to install (choose)")
        box.pack(fill="both", expand=False, padx=12, pady=6)

        self.vars = []
        header = ttk.Frame(box); header.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(header, text="Select", width=8).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Model",  width=42).grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Notes",  width=60).grid(row=0, column=2, sticky="w")

        for item in MODEL_ITEMS:
            row = ttk.Frame(box); row.pack(fill="x", padx=6, pady=2)
            var = tk.BooleanVar(value=True)
            self.vars.append(var)
            ttk.Checkbutton(row, variable=var, width=8).grid(row=0, column=0, sticky="w")
            name = item["ui"] + ("  🔒" if item["gated"] else "")
            ttk.Label(row, text=name, width=42, anchor="w").grid(row=0, column=1, sticky="w")
            ttk.Label(row, text=item["blurb"], width=60, anchor="w").grid(row=0, column=2, sticky="w")

        ttk.Label(box,
                  text="🔒 requires you to paste a Hugging Face token (access is free after accepting terms).",
                  foreground="#555").pack(anchor="w", padx=6, pady=(4, 4))

    def _build_actions(self):
        frm = ttk.Frame(self)
        frm.pack(fill="x", padx=12, pady=6)
        self.btn_download = ttk.Button(frm, text="Start Download", command=self.start_downloads)
        self.btn_download.pack(side="left")
        ttk.Button(frm, text="Close", command=self.destroy).pack(side="right")

    def _build_log(self):
        box = ttk.LabelFrame(self, text="Log")
        box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt = tk.Text(box, wrap="word", height=18)
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)
        self._log_banner()

    # --- helpers ---
    def _log_banner(self):
        self.txt.insert("end",
            "This window streams the Hugging Face CLI output, including file sizes and speeds when available.\n"
            f"Destination folder: {MODELS_DIR}\n\n")
        self.txt.see("end")

    def _ui_set_status(self, msg):
        self.title(f"Transcribe Offline — Model Downloader   [{msg}]")

    def log(self, *parts):
        self.txt.insert("end", " ".join(str(p) for p in parts) + "\n")
        self.txt.see("end")

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

        if HF_CMD is None:
            messagebox.showerror("huggingface-cli not found",
                                 "The Hugging Face CLI could not be located.\n"
                                 "Install it into the same Python used by the app:\n\n"
                                 "python -m pip install huggingface_hub\n\n"
                                 "Then run this downloader again.")
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

                args = ["download", item["repo"],
                        "--revision", "main",
                        "--local-dir", str(local_dir),
                        "--local-dir-use-symlinks", "False"]

                for inc in item.get("include", []) or []:
                    args.extend(["--include", inc])
                for exc in item.get("exclude", []) or []:
                    args.extend(["--exclude", exc])

                env = os.environ.copy()
                if item["gated"]:
                    env["HF_TOKEN"] = token
                    env["HUGGINGFACE_HUB_TOKEN"] = token

                self._run_cli_stream(args, env)

                # Special case: put silero_vad.jit directly under content/models
                if item["repo"] == "snakers4/silero-vad":
                    src = next(local_dir.glob("silero_vad.jit"), None)
                    if src:
                        dst = MODELS_DIR / "silero_vad.jit"
                        try:
                            if dst.exists():
                                dst.unlink()
                            src.replace(dst)
                            try:
                                local_dir.rmdir()
                            except OSError:
                                pass
                            self.queue.put(("log", f"Placed: {dst}"))
                        except Exception as e:
                            self.queue.put(("log", f"Note: could not move silero_vad.jit: {e}"))

                self.queue.put(("log", f"✓ Done: {item['ui']}"))

            except Exception as e:
                self.queue.put(("log", f"✗ Failed: {item['ui']} — {e}"))

        self.queue.put(("done", None))

    def _run_cli_stream(self, args, env):
        cmd = HF_CMD + args
        self.queue.put(("log", "Running:", " ".join(shlex.quote(a) for a in cmd)))
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            env=env
        ) as proc:
            for line in proc.stdout:
                self.queue.put(("log", line.rstrip()))
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"huggingface-cli exited with code {ret}")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self.downloading = False
                    self.btn_download.config(state="normal")
                    self._ui_set_status("Done.")
                    self.log("\nAll selected downloads finished.")
                self.queue.task_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


def main():
    if HF_CMD is None:
        print("Warning: huggingface-cli not found. Install with:")
        print("  python -m pip install huggingface_hub")
    app = ModelDownloaderGUI()
    app.mainloop()

if __name__ == "__main__":
    main()

