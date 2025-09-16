# setup_transcribe_offline.R — Windows x64, RStudio
# One-shot setup: real Downloads, no model downloads here. Creates Transcribe_Offline, installs pinned wheels,
# downloads app + vendors, fetches models.py, writes runner, then LAUNCHES models.py via the reticulate Python.

wlog <- function(...) cat(sprintf("[ %s ] %s\n", format(Sys.time(), "%H:%M:%S"), paste0(...)))
halt <- function(msg) stop(paste("ERROR:", msg), call. = FALSE)
safe_unlink <- function(path) { if (length(path) && any(dir.exists(path) | file.exists(path))) unlink(path, recursive = TRUE, force = TRUE) }
`%||%` <- function(x, y) if (is.null(x) || length(x)==0 || !nzchar(x)) y else x

if (tolower(Sys.info()[["sysname"]]) != "windows") halt("This setup is for Windows x64 only.")

# Resolve real Windows Downloads (handles OneDrive)
get_downloads <- function() {
  ps <- c("-NoProfile","-Command",
          "(New-Object -ComObject Shell.Application).NameSpace('shell:Downloads').Self.Path")
  out <- tryCatch(suppressWarnings(system2("powershell", ps, stdout = TRUE, stderr = TRUE)), error = function(e) "")
  out <- if (length(out)) paste(out, collapse = "\n") else ""
  out <- gsub("\\s+$","", out)
  if (nzchar(out) && dir.exists(out)) return(normalizePath(out, winslash="/"))
  up <- Sys.getenv("USERPROFILE", unset = "")
  od <- Sys.getenv("OneDrive", unset = "")
  cands <- c(if (nzchar(up)) file.path(up,"Downloads"),
             if (nzchar(od)) file.path(od,"Downloads"))
  cands <- cands[dir.exists(cands)]
  if (length(cands)) return(normalizePath(cands[[1]], winslash="/"))
  target <- normalizePath(file.path(up %||% path.expand("~"), "Downloads"), winslash="/", mustWork = FALSE)
  dir.create(target, recursive = TRUE, showWarnings = FALSE)
  target
}

downloads_dir <- get_downloads()
wlog("Downloads:", downloads_dir)

# Project paths
proj_dir <- normalizePath(file.path(downloads_dir, "Transcribe_Offline"), winslash = "/", mustWork = FALSE)
dirs <- list(
  proj_dir = proj_dir,
  content  = file.path(proj_dir, "content"),
  licenses = file.path(proj_dir, "content", "licenses"),
  vendor   = file.path(proj_dir, "content", "vendor"),
  ffmpeg   = file.path(proj_dir, "content", "vendor", "ffmpeg"),
  llama    = file.path(proj_dir, "content", "vendor", "llama.cpp")
)
for (d in dirs) if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
setwd(proj_dir); wlog("Working in:", proj_dir)

# Write runner immediately
write_runner <- function() {
  runner <- '
suppressWarnings(suppressMessages({ library(reticulate) }))
this_file <- normalizePath(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))], mustWork = FALSE)
if (length(this_file)) setwd(dirname(sub("^--file=", "", this_file)))
proj_dir <- getwd()
py_hint <- file.path(proj_dir, "python_path.txt")
if (!file.exists(py_hint)) stop("python_path.txt missing. Run setup_transcribe_offline.R first.")
py <- readLines(py_hint, warn = FALSE); py <- py[nzchar(py)][1]
if (!nzchar(py) || !file.exists(py)) stop("Saved Python not found. Re-run setup_transcribe_offline.R.")
Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE",
           TOKENIZERS_PARALLELISM = "false",
           PATH = paste(file.path(proj_dir, "content", "vendor", "ffmpeg"),
                        file.path(proj_dir, "content", "vendor", "llama.cpp"),
                        Sys.getenv("PATH"), sep = .Platform$path.sep))
app <- file.path(proj_dir, "main.py")
if (!file.exists(app)) stop("main.py missing.")
invisible(system2(py, args = shQuote(app)))
'
  writeLines(runner, file.path(proj_dir, "run_transcribe_offline.R"))
}
write_runner()
wlog("Runner written:", file.path(proj_dir, "run_transcribe_offline.R"))

# reticulate-managed Python 3.11
if (!requireNamespace("reticulate", quietly = TRUE)) { wlog("Installing 'reticulate'…"); install.packages("reticulate") }
library(reticulate)
if (packageVersion("reticulate") < "1.41.0") halt("reticulate >= 1.41.0 is required.")
py_require(python_version = "3.11")

py <- tryCatch({
  cfg <- reticulate::py_config()
  if (!is.null(cfg$python) && nzchar(cfg$python) && file.exists(cfg$python)) cfg$python else ""
}, error = function(e) "")
if (!nzchar(py) || !file.exists(py)) {
  py <- tryCatch({
    reticulate::py_run_string("import sys, pathlib; __executable__ = str(pathlib.Path(sys.executable))")
    p <- reticulate::py$`__executable__`
    if (!is.null(p) && file.exists(p)) p else ""
  }, error = function(e) "")
}
if (!nzchar(py) || !file.exists(py)) halt("Couldn't determine the Python executable after py_require().")
wlog("reticulate Python:", py)

# Verify Tkinter
wlog("Checking Tkinter…")
tk_ok <- tryCatch({
  out <- suppressWarnings(system2(py, c("-c", shQuote("import tkinter as tk; print('OK')")), stdout = TRUE, stderr = TRUE))
  any(grepl("^OK$", out))
}, error = function(e) FALSE)
if (!tk_ok) halt("Tkinter not available in the reticulate-managed Python 3.11.")
writeLines(py, file.path(proj_dir, "python_path.txt")); wlog("Tkinter OK; saved python_path.txt")

# Pinned requirements (pip --no-deps)
req_path <- file.path(proj_dir, "requirements-pinned.txt")
req_txt <- "
asteroid-filterbanks==0.4.0
av==15.1.0
certifi==2025.8.3
cffi==2.0.0
charset-normalizer==2.1.0
colorama==0.4.6
colorlog==6.9.0
ctranslate2==4.6.0
cycler==0.12.1
einops==0.8.1
faster-whisper==1.2.0
filelock==3.19.1
fsspec==2025.9.0
huggingface-hub==0.34.4
HyperPyYAML==1.2.2
idna==3.10
joblib==1.5.2
julius==0.2.7
kiwisolver==1.4.9
lightning==2.5.5
lightning-utilities==0.15.2
matplotlib==3.10.6
mpmath==1.3.0
networkx==3.5
nltk==3.9.1
numpy==2.3.3
omegaconf==2.3.0
onnxruntime==1.22.1
optuna==4.5.0
packaging==25.0
pandas==2.3.2
pillow==11.3.0
primePy==1.3
pyannote-core==6.0.0
pyannote-database==6.0.0
pyannote-metrics==4.0.0
pyannote-pipeline==4.0.0
pyannote.audio==3.4.0
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytorch-lightning==2.5.5
pytorch-metric-learning==2.9.0
pytz==2025.2
PyYAML==6.0.2
regex==2025.9.1
requests==2.32.5
rich==14.1.0
safetensors==0.6.2
scikit-learn==1.7.2
scipy==1.16.2
semver==3.0.4
six==1.17.0
sortedcontainers==2.4.0
sounddevice==0.5.2
soundfile==0.13.1
speechbrain==1.0.3
sympy==1.14.0
tabulate==0.9.0
tensorboardX==2.6.4
threadpoolctl==3.6.0
tokenizers==0.22.0
torch==2.8.0
torch-audiomentations==0.12.0
torch-pitch-shift==1.2.5
torchaudio==2.8.0
torchmetrics==1.8.2
tqdm==4.67.1
transformers==4.56.1
typing-extensions==4.15.0
urllib3==2.5.0
whisperx==3.4.2
"
writeLines(trimws(req_txt), req_path)
system2(py, c("-m","pip","install","--upgrade","pip","setuptools","wheel"))
wlog("Installing pinned wheels (pip --no-deps)…")
st <- system2(py, c("-m","pip","install","--no-deps","-r", shQuote(req_path)))
if (!identical(st, 0L)) halt("pip install --no-deps failed.")

# Download app sources
RAW <- "https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/Windows"
dl <- function(url, dest, mode = "wb") {
  wlog("Downloading:", url)
  utils::download.file(url, destfile = dest, mode = mode, quiet = TRUE)
  if (!file.exists(dest) || file.info(dest)$size == 0) halt(paste("Failed to download:", url))
}
dl(file.path(RAW, "main.py"),                               file.path(proj_dir, "main.py"), mode = "wb")
dl(file.path(RAW, "content/AppIcon.ico"),                   file.path(dirs$content, "AppIcon.ico"), mode = "wb")
dl(file.path(RAW, "content/licenses/LICENSE.txt"),          file.path(dirs$licenses, "LICENSE.txt"), mode = "wb")
dl(file.path(RAW, "content/licenses/README.md"),            file.path(dirs$licenses, "README.md"), mode = "wb")
dl(file.path(RAW, "content/licenses/Third-Party-Licenses.txt"),
   file.path(dirs$licenses, "Third-Party-Licenses.txt"), mode = "wb")
if (!file.exists(file.path(proj_dir, "main.py"))) halt("main.py missing after download.")

# Vendors: FFmpeg (filtered & flattened)
ffmpeg_zip_url <- "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2025-07-31-14-15/ffmpeg-N-120424-g03b9180fe3-win64-lgpl-shared.zip"
ffmpeg_zip <- file.path(proj_dir, "ffmpeg.zip")
dl(ffmpeg_zip_url, ffmpeg_zip)
wlog("Extracting FFmpeg (filtered)…")
tmp_ff <- file.path(proj_dir, "__tmp_ffmpeg")
dir.create(tmp_ff, showWarnings = FALSE, recursive = TRUE)
utils::unzip(ffmpeg_zip, exdir = tmp_ff)
all_ff <- list.files(tmp_ff, recursive = TRUE, full.names = TRUE)
keep_pat <- c("ffmpeg\\.exe$", "avcodec-.*\\.dll$", "avdevice-.*\\.dll$", "avfilter-.*\\.dll$",
              "avformat-.*\\.dll$", "avutil-.*\\.dll$", "swresample-.*\\.dll$", "swscale-.*\\.dll$", "^LICENSE(\\.|$)")
keep_idx <- Reduce("|", lapply(keep_pat, function(p) grepl(p, basename(all_ff), ignore.case = TRUE)))
sel_ff <- all_ff[keep_idx]
if (!length(sel_ff)) halt("Could not locate FFmpeg artifacts in the ZIP.")
dir.create(dirs$ffmpeg, showWarnings = FALSE, recursive = TRUE)
ok <- file.copy(sel_ff, file.path(dirs$ffmpeg, basename(sel_ff)), overwrite = TRUE)
if (!all(ok)) halt("Failed to copy some FFmpeg files.")
safe_unlink(tmp_ff); safe_unlink(ffmpeg_zip)
if (!file.exists(file.path(dirs$ffmpeg, "ffmpeg.exe"))) halt("ffmpeg.exe not found after extraction.")
wlog("FFmpeg OK.")

# Vendors: llama.cpp (required files only)
llama_zip_url <- "https://github.com/ggml-org/llama.cpp/releases/download/b6435/llama-b6435-bin-win-cpu-x64.zip"
llama_zip <- file.path(proj_dir, "llama.zip")
dl(llama_zip_url, llama_zip)
wlog("Extracting llama.cpp (filtered)…")
tmp_ll <- file.path(proj_dir, "__tmp_llama")
dir.create(tmp_ll, showWarnings = FALSE, recursive = TRUE)
utils::unzip(llama_zip, exdir = tmp_ll)
all_ll <- list.files(tmp_ll, recursive = TRUE, full.names = TRUE)
keep_ll_pat <- c("llama-cli\\.exe$", "llama\\.dll$", "mtmd\\.dll$", "^ggml.*\\.dll$", "^libcurl-.*\\.dll$", "^libomp.*\\.dll$", "^LICENSE.*")
keep_ll_idx <- Reduce("|", lapply(keep_ll_pat, function(p) grepl(p, basename(all_ll), ignore.case = TRUE)))
sel_ll <- all_ll[keep_ll_idx]
if (!length(sel_ll)) halt("No required llama.cpp artifacts found in the ZIP.")
dir.create(dirs$llama, showWarnings = FALSE, recursive = TRUE)
ok <- file.copy(sel_ll, file.path(dirs$llama, basename(sel_ll)), overwrite = TRUE)
if (!all(ok)) halt("Failed to copy some llama.cpp files.")
safe_unlink(tmp_ll); safe_unlink(llama_zip)
if (!file.exists(file.path(dirs$llama, "llama-cli.exe"))) halt("llama-cli.exe not found after extraction.")
wlog("llama.cpp OK.")

# Download models.py (GUI) into the project root
models_url <- "https://raw.githubusercontent.com/openresearchtools/transcribeoffline/main/R_win64/models.py"
models_path <- file.path(proj_dir, "models.py")
dl(models_url, models_path, mode = "wb")
wlog("Downloaded models.py:", models_path)

# Re-write runner at end (idempotent)
write_runner()

suppressWarnings(suppressMessages({ library(reticulate) }))
this_file <- normalizePath(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))], mustWork = FALSE)
if (length(this_file)) setwd(dirname(sub("^--file=", "", this_file)))
proj_dir <- getwd()

py_hint <- file.path(proj_dir, "python_path.txt")
if (!file.exists(py_hint)) stop("python_path.txt missing. Run setup_transcribe_offline.R first.")
py <- readLines(py_hint, warn = FALSE); py <- py[nzchar(py)][1]
if (!nzchar(py) || !file.exists(py)) stop("Saved Python not found. Re-run setup_transcribe_offline.R.")

Sys.setenv(
  KMP_DUPLICATE_LIB_OK = "TRUE",
  TOKENIZERS_PARALLELISM = "false",
  PATH = paste(
    file.path(proj_dir, "content", "vendor", "ffmpeg"),
    file.path(proj_dir, "content", "vendor", "llama.cpp"),
    Sys.getenv("PATH"),
    sep = .Platform$path.sep
  ),
  HF_HOME = file.path(proj_dir, "content", "models"),
  HUGGINGFACE_HUB_CACHE = file.path(proj_dir, "content", "models"),
  TRANSFORMERS_CACHE = file.path(proj_dir, "content", "models")
)

app <- file.path(proj_dir, "models.py")
if (!file.exists(app)) stop("models.py missing.")

# Use -u for unbuffered stdout so CLI progress streams nicely into the GUI log.
# Set wait = FALSE so R returns immediately while the Tkinter window opens.
invisible(system2(py, args = c("-u", shQuote(app)), wait = FALSE))


wlog("")
wlog("DONE.")
wlog(sprintf("Project: %s", proj_dir))
wlog("Runner:  %s", file.path(proj_dir, "run_transcribe_offline.R"))
wlog("You can re-open the app later with: source(file.path('~','Downloads','Transcribe_Offline','run_transcribe_offline.R'))")
