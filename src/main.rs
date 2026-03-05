#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod bridge;
mod runtime_installer;
mod settings;

use anyhow::{anyhow, bail, Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream, StreamConfig,
};
use eframe::{egui, App, Frame, NativeOptions};
use regex::Regex;
use rfd::FileDialog;
use serde_json::{json, Value};
use settings::{
    app_paths, default_diarization_models_dir, default_whisper_model_path, ensure_dirs,
    load_settings, save_model_links, save_settings, AppPaths, AppSettings,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bridge::{AudioRunParams, BridgeApi, ChatRunParams, SharedBridgeParams};

const AUDIO_DEVICE_CPU_LABEL: &str = "CPU (no GPU)";
const UI_FONT_SIZE_DEFAULT_PX: f32 = 12.0;
const UI_FONT_SIZE_MIN_PX: f32 = 10.0;
const UI_FONT_SIZE_MAX_PX: f32 = 22.0;
const UI_FONT_SIZE_STEP_PX: f32 = 1.0;
const DIARIZATION_REPO: &str =
    "https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF";
const ANONYMISE_TOOLTIP: &str =
    "Anonymise (beta) requires a chat model configured in Settings -> Chat settings.";
const PLAYBACK_SCROLL_FOLLOW_PAUSE_SEC: f64 = 5.0;
const PLAYBACK_CLICK_SYNC_HOLD_SEC: f64 = 2.5;
const APP_ICON_PNG_BYTES: &[u8] = include_bytes!("../assets/icons/AppIcon.png");
const BUNDLED_TRANSCRIBE_OFFLINE_LICENSE_TXT: &str = include_str!("../LICENSE");
const BUNDLED_THIRD_PARTY_NOTICES_ALL_MD: &str =
    include_str!("../licenses/THIRD_PARTY_NOTICES_ALL.md");
const BUNDLED_THIRD_PARTY_LICENSES_ALL_MD: &str =
    include_str!("../licenses/THIRD_PARTY_LICENSES_ALL.md");
const BUNDLED_ENGINE_THIRD_PARTY_LICENSES_FULL_MD: &str =
    include_str!("../licenses/ENGINE_THIRD_PARTY_LICENSES_FULL.md");
#[cfg(target_os = "windows")]
const APP_ICON_ICO_BYTES: &[u8] = include_bytes!("../assets/icons/AppIcon.ico");

fn decode_app_icon_data() -> Option<egui::IconData> {
    #[cfg(target_os = "windows")]
    {
        let mut cursor = std::io::Cursor::new(APP_ICON_ICO_BYTES);
        if let Ok(icon_dir) = ico::IconDir::read(&mut cursor) {
            let best_entry = icon_dir
                .entries()
                .iter()
                .max_by_key(|entry| (entry.width() as u32) * (entry.height() as u32));
            if let Some(entry) = best_entry {
                if let Ok(icon_image) = entry.decode() {
                    return Some(egui::IconData {
                        rgba: icon_image.rgba_data().to_vec(),
                        width: icon_image.width(),
                        height: icon_image.height(),
                    });
                }
            }
        }
    }

    eframe::icon_data::from_png_bytes(APP_ICON_PNG_BYTES).ok()
}

#[derive(Clone, Copy)]
struct WhisperModelSpec {
    label: &'static str,
    file_name: &'static str,
    url: &'static str,
}

fn clamp_font_px(size_px: f32) -> f32 {
    size_px.clamp(UI_FONT_SIZE_MIN_PX, UI_FONT_SIZE_MAX_PX)
}

fn parse_size_token_bytes(token: &str) -> Option<f64> {
    let cleaned = token.trim().replace(',', "");
    let mut parts = cleaned.split_whitespace();
    let value = parts.next()?.parse::<f64>().ok()?;
    let unit = parts.next().unwrap_or("B").to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "b" => 1.0,
        "kb" | "kib" => 1024.0,
        "mb" | "mib" => 1024.0 * 1024.0,
        "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some(value * multiplier)
}

fn parse_download_fraction(status: &str) -> Option<f32> {
    if !status.to_ascii_lowercase().contains("downloading") {
        return None;
    }
    let before_speed = status.split(" at ").next().unwrap_or(status);
    let slash = before_speed.rfind('/')?;
    let done_part = before_speed[..slash].split(':').next_back()?.trim();
    let total_part = before_speed[slash + 1..].trim();
    let done = parse_size_token_bytes(done_part)?;
    let total = parse_size_token_bytes(total_part)?;
    if total <= 0.0 {
        return None;
    }
    Some((done / total).clamp(0.0, 1.0) as f32)
}

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    "unknown panic payload".to_string()
}

fn append_panic_log(paths: &AppPaths, phase: &str, details: &str) {
    let log_path = paths.data_dir.join("panic.log");
    let _ = fs::create_dir_all(&paths.data_dir);
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        let _ = writeln!(f, "[{now}] {phase}");
        let _ = writeln!(f, "{details}");
        let _ = writeln!(f, "----------------------------------------");
    }
}

fn install_panic_hook(paths: AppPaths) {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = panic_payload_to_string(info.payload());
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let backtrace = std::backtrace::Backtrace::force_capture();
        let details = format!("panic at {location}: {payload}\n{backtrace}");
        append_panic_log(&paths, "panic-hook", &details);
        prev(info);
    }));
}

fn engine_panel_frame() -> egui::Frame {
    egui::Frame::default()
        .fill(egui::Color32::from_rgb(255, 255, 255))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgb(208, 212, 218),
        ))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::same(5))
}

fn accent_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(label).strong())
            .corner_radius(egui::CornerRadius::same(6))
            .min_size(egui::vec2(0.0, 22.0)),
    )
}

fn secondary_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(label)
            .corner_radius(egui::CornerRadius::same(6))
            .min_size(egui::vec2(0.0, 22.0)),
    )
}

fn warning_button(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(label).strong())
            .fill(egui::Color32::from_rgb(186, 58, 58))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(130, 25, 25)))
            .corner_radius(egui::CornerRadius::same(6))
            .min_size(egui::vec2(0.0, 22.0)),
    )
}

fn tab_button(ui: &mut egui::Ui, label: &str, selected: bool) -> egui::Response {
    ui.add(
        egui::Button::new(label)
            .selected(selected)
            .corner_radius(egui::CornerRadius::same(6))
            .min_size(egui::vec2(0.0, 22.0)),
    )
}

fn native_menu_item(ui: &mut egui::Ui, label: &str) -> egui::Response {
    ui.add(egui::Button::new(label).wrap_mode(egui::TextWrapMode::Extend))
}

fn configure_preferred_fonts(ctx: &egui::Context) {
    #[cfg(target_os = "windows")]
    {
        let mut fonts = egui::FontDefinitions::default();
        let mut changed = false;

        let segoe_path = Path::new(r"C:\Windows\Fonts\segoeui.ttf");
        if let Ok(data) = fs::read(segoe_path) {
            fonts.font_data.insert(
                "segoe_ui".to_string(),
                std::sync::Arc::new(egui::FontData::from_owned(data)),
            );
            if let Some(fam) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                fam.insert(0, "segoe_ui".to_string());
            }
            changed = true;
        }

        let consolas_path = Path::new(r"C:\Windows\Fonts\consola.ttf");
        if let Ok(data) = fs::read(consolas_path) {
            fonts.font_data.insert(
                "consolas".to_string(),
                std::sync::Arc::new(egui::FontData::from_owned(data)),
            );
            if let Some(fam) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                fam.insert(0, "consolas".to_string());
            }
            changed = true;
        }

        if changed {
            ctx.set_fonts(fonts);
        }
    }
}

fn has_prefixed_file(dir: &Path, prefix: &str, suffix: &str) -> bool {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return false;
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(prefix) && name.ends_with(suffix) {
            return true;
        }
    }
    false
}

#[cfg(target_os = "windows")]
fn has_prefixed_file_case_insensitive(dir: &Path, prefix: &str, suffix: &str) -> bool {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return false;
    };
    let prefix_lc = prefix.to_ascii_lowercase();
    let suffix_lc = suffix.to_ascii_lowercase();
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.starts_with(&prefix_lc) && name.ends_with(&suffix_lc) {
            return true;
        }
    }
    false
}

#[cfg(target_os = "windows")]
fn detect_installed_windows_runtime_backend(runtime_dir: &Path) -> Option<String> {
    if !runtime_dir.exists() {
        return None;
    }

    let cuda_dirs = [
        runtime_dir.join("vendor").join("cuda"),
        runtime_dir.join("vendor"),
        runtime_dir.join("bin"),
        runtime_dir.to_path_buf(),
    ];
    let has_cuda_marker = cuda_dirs.iter().any(|dir| {
        has_prefixed_file_case_insensitive(dir, "cublas", ".dll")
            || has_prefixed_file_case_insensitive(dir, "cudart", ".dll")
    });
    if has_cuda_marker {
        return Some("cuda".to_string());
    }

    if runtime_dir.join(bridge_library_file_name()).exists() {
        return Some("vulkan".to_string());
    }

    None
}

fn ffmpeg_runtime_dir(runtime_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        runtime_dir.join("vendor").join("ffmpeg").join("bin")
    } else {
        runtime_dir.join("vendor").join("ffmpeg").join("lib")
    }
}

fn runtime_missing_messages(runtime_dir: &Path) -> Vec<String> {
    let mut missing = Vec::new();

    if !runtime_dir.exists() {
        missing.push(format!(
            "Runtime directory does not exist: {}",
            runtime_dir.display()
        ));
        return missing;
    }

    let bridge = runtime_dir.join(bridge_library_file_name());
    if !bridge.exists() {
        missing.push(format!(
            "Missing bridge library: {}",
            bridge_library_file_name()
        ));
    }

    let ffmpeg_dir = ffmpeg_runtime_dir(runtime_dir);
    if !ffmpeg_dir.exists() {
        missing.push(format!(
            "Missing FFmpeg runtime directory: {}",
            ffmpeg_dir.display()
        ));
    } else if cfg!(target_os = "windows") {
        for prefix in ["avcodec", "avformat", "avutil", "swresample", "swscale"] {
            if !has_prefixed_file(&ffmpeg_dir, prefix, ".dll") {
                missing.push(format!("Missing FFmpeg DLL {}*.dll", prefix));
            }
        }
    } else if cfg!(target_os = "macos") {
        for prefix in [
            "libavcodec",
            "libavformat",
            "libavutil",
            "libswresample",
            "libswscale",
        ] {
            if !has_prefixed_file(&ffmpeg_dir, prefix, ".dylib") {
                missing.push(format!("Missing FFmpeg dylib {}*.dylib", prefix));
            }
        }
    } else {
        for prefix in [
            "libavcodec",
            "libavformat",
            "libavutil",
            "libswresample",
            "libswscale",
        ] {
            let has_prefixed = has_prefixed_file(&ffmpeg_dir, prefix, ".so")
                || has_prefixed_file(&ffmpeg_dir, prefix, ".so.0")
                || has_prefixed_file(&ffmpeg_dir, prefix, ".so.1")
                || has_prefixed_file(&ffmpeg_dir, prefix, ".so.2");
            if !has_prefixed {
                let exists = fs::read_dir(&ffmpeg_dir)
                    .ok()
                    .into_iter()
                    .flat_map(|x| x.flatten())
                    .any(|e| e.file_name().to_string_lossy().starts_with(prefix));
                if !exists {
                    missing.push(format!("Missing FFmpeg shared object {}*.so*", prefix));
                }
            }
        }
    }

    missing
}

#[cfg(target_os = "windows")]
const EMBEDDED_UNSIGNED_RUNTIME_UNBLOCK_SCRIPT: &str =
    include_str!("../scripts/unblock-unsigned-runtime.ps1");
#[cfg(not(target_os = "windows"))]
const EMBEDDED_UNSIGNED_RUNTIME_UNBLOCK_SCRIPT: &str =
    include_str!("../scripts/unblock-unsigned-runtime.sh");

fn embedded_unsigned_runtime_script_file_name() -> &'static str {
    if cfg!(windows) {
        "unblock-unsigned-runtime.ps1"
    } else {
        "unblock-unsigned-runtime.sh"
    }
}

fn write_embedded_unsigned_runtime_script() -> Result<(PathBuf, PathBuf)> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_root = env::temp_dir().join(format!("transcribe-offline-unblock-{stamp}"));
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("failed creating '{}'", temp_root.display()))?;

    let script_path = temp_root.join(embedded_unsigned_runtime_script_file_name());
    fs::write(&script_path, EMBEDDED_UNSIGNED_RUNTIME_UNBLOCK_SCRIPT)
        .with_context(|| format!("failed writing '{}'", script_path.display()))?;

    #[cfg(unix)]
    {
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755)).with_context(
            || {
                format!(
                    "failed setting executable mode on '{}'",
                    script_path.display()
                )
            },
        )?;
    }

    Ok((temp_root, script_path))
}

fn run_unsigned_runtime_unblock_script(_paths: &AppPaths, runtime_dir: &Path) -> Result<String> {
    if !runtime_dir.exists() {
        bail!(
            "Runtime directory does not exist: '{}'",
            runtime_dir.display()
        );
    }

    let (temp_root, script_path) = write_embedded_unsigned_runtime_script()?;
    let run_result = (|| -> Result<String> {
        let mut command = if cfg!(windows) {
            let mut cmd = std::process::Command::new("powershell");
            cmd.arg("-NoProfile")
                .arg("-ExecutionPolicy")
                .arg("Bypass")
                .arg("-File")
                .arg(&script_path)
                .arg("-RuntimeDir")
                .arg(runtime_dir);
            cmd
        } else {
            let mut cmd = std::process::Command::new("sh");
            cmd.arg(&script_path).arg(runtime_dir);
            cmd
        };
        command.stdin(std::process::Stdio::null());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());

        let output = command.output().with_context(|| {
            format!(
                "failed to execute embedded unsigned-runtime script '{}'",
                script_path.display()
            )
        })?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !output.status.success() {
            let details = if stderr.is_empty() {
                stdout.clone()
            } else {
                stderr
            };
            bail!(
                "embedded unsigned-runtime script failed ({}): {}",
                script_path.display(),
                details
            );
        }

        let mut message = "Unsigned runtime unblock complete.".to_string();
        if !stdout.is_empty() {
            message.push_str(&format!(" | {stdout}"));
        }
        Ok(message)
    })();

    let _ = fs::remove_dir_all(&temp_root);
    run_result
}

const WHISPER_MODELS: &[WhisperModelSpec] = &[
    WhisperModelSpec {
        label: "large-v3-turbo (recommended)",
        file_name: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true",
    },
    WhisperModelSpec {
        label: "large-v3",
        file_name: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin?download=true",
    },
    WhisperModelSpec {
        label: "medium",
        file_name: "ggml-medium.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin?download=true",
    },
    WhisperModelSpec {
        label: "small",
        file_name: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true",
    },
];

#[allow(dead_code)]
const DIARIZATION_FILES: &[(&str, &str)] = &[
    (
        "segmentation.gguf",
        "https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF/resolve/main/segmentation.gguf?download=true",
    ),
    (
        "embedding.gguf",
        "https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF/resolve/main/embedding.gguf?download=true",
    ),
    (
        "plda.gguf",
        "https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF/resolve/main/plda.gguf?download=true",
    ),
    (
        "xvec_transform.gguf",
        "https://huggingface.co/openresearchtools/speaker-diarization-community-1-GGUF/resolve/main/xvec_transform.gguf?download=true",
    ),
];

#[derive(Debug, Clone)]
struct TranscriptionResult {
    _response_json: Value,
    output_path: PathBuf,
    output_text: String,
    preprocess_note: String,
    mode: String,
    custom_value: String,
    whisper_model: String,
    diarization_enabled: bool,
    timing_prepare_ms: u128,
    timing_read_audio_ms: u128,
    timing_bridge_ms: u128,
    timing_read_output_ms: u128,
    timing_total_ms: u128,
    audio_bytes_len: usize,
}

#[derive(Debug, Clone)]
struct MediaEntry {
    path: PathBuf,
    selected: bool,
}

#[derive(Debug, Clone)]
struct OutputEntry {
    path: PathBuf,
    selected: bool,
}

#[derive(Debug, Clone)]
struct QueuedJob {
    id: u64,
    settings: AppSettings,
    files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct ActiveJob {
    id: u64,
    total_files: usize,
    done_files: usize,
}

#[derive(Debug, Clone)]
struct TranscriptCue {
    line_index: usize,
    start_sec: f64,
    end_sec: f64,
}

#[derive(Debug, Clone, Default)]
struct TranscriptState {
    cues: Vec<TranscriptCue>,
}

#[derive(Debug, Clone, Default)]
struct RuntimeState {
    original_output_path: Option<PathBuf>,
    edited_output_path: Option<PathBuf>,
    active_audio_path: Option<PathBuf>,
    media_entries: Vec<MediaEntry>,
    output_entries: Vec<OutputEntry>,
    running_jobs: usize,
    completed_jobs: usize,
    job_queue: VecDeque<QueuedJob>,
    active_job: Option<ActiveJob>,
    active_stage: String,
    queue_worker_running: bool,
    next_job_id: u64,
    download_status: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AnonymiseEntityKind {
    Name,
    Location,
    Address,
    Phone,
    Email,
    Number,
    Organisation,
    Identifier,
}

const ANONYMISE_ENTITY_ORDER: [AnonymiseEntityKind; 8] = [
    AnonymiseEntityKind::Name,
    AnonymiseEntityKind::Location,
    AnonymiseEntityKind::Address,
    AnonymiseEntityKind::Phone,
    AnonymiseEntityKind::Email,
    AnonymiseEntityKind::Number,
    AnonymiseEntityKind::Organisation,
    AnonymiseEntityKind::Identifier,
];

#[derive(Debug, Clone)]
struct AnonymiseMatch {
    kind: AnonymiseEntityKind,
    value: String,
}

#[derive(Debug, Clone)]
struct AnonymiseRunResult {
    total_matches: usize,
    usable_matches: usize,
    exact_replacements: usize,
    fuzzy_replacements: usize,
    output_path: PathBuf,
}

#[derive(Debug, Clone)]
struct SpeakerRenameEntry {
    speaker_tag: String,
    replacement: String,
}

impl AnonymiseEntityKind {
    fn key(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Location => "location",
            Self::Address => "address",
            Self::Phone => "phone",
            Self::Email => "email",
            Self::Number => "number",
            Self::Organisation => "organisation",
            Self::Identifier => "identifier",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Name => "Names",
            Self::Location => "Locations",
            Self::Address => "Addresses",
            Self::Phone => "Phone numbers",
            Self::Email => "Email addresses",
            Self::Number => "Other numbers",
            Self::Organisation => "Organisations",
            Self::Identifier => "Identifiers",
        }
    }

    fn prompt_guidance(self) -> &'static str {
        match self {
            Self::Name => {
                "Person names, initials, nicknames, and speaker names that identify a person. Ignore structural transcript labels/tokens (for example, SPEAKER_<number> tags and timing headers)."
            }
            Self::Location => {
                "Any geographic/location reference: countries, regions, provinces/states, cities, villages, districts, streets, landmarks, venues, sites, departments, buildings."
            }
            Self::Address => {
                "Street or postal addresses including postcode/zip when part of the address."
            }
            Self::Phone => {
                "Phone, mobile, fax, or extension numbers written with any separators."
            }
            Self::Email => "Email addresses or address-like handles containing domains.",
            Self::Number => {
                "Sensitive numbers not already covered above (account/case/reference numbers)."
            }
            Self::Organisation => {
                "Company, employer, school, clinic, hospital, or institution names."
            }
            Self::Identifier => {
                "Explicit IDs/codes such as MRN, SSN, policy ID, student ID, license plate."
            }
        }
    }

    fn prompt_kv_example(self) -> &'static str {
        match self {
            Self::Name => "name: <person_name_from_transcript>;",
            Self::Location => "location: <location_from_transcript>;",
            Self::Address => "address: <address_from_transcript>;",
            Self::Phone => "phone: <phone_number_from_transcript>;",
            Self::Email => "email: <email_address_from_transcript>;",
            Self::Number => "number: <sensitive_number_from_transcript>;",
            Self::Organisation => "organisation: <organisation_from_transcript>;",
            Self::Identifier => "identifier: <identifier_from_transcript>;",
        }
    }

    fn placeholder(self) -> &'static str {
        match self {
            Self::Name => "[name omitted]",
            Self::Location => "[location omitted]",
            Self::Address => "[address omitted]",
            Self::Phone => "[number omitted]",
            Self::Email => "[email omitted]",
            Self::Number => "[number omitted]",
            Self::Organisation => "[organisation omitted]",
            Self::Identifier => "[identifier omitted]",
        }
    }

    fn supports_fuzzy(self) -> bool {
        matches!(
            self,
            Self::Name | Self::Location | Self::Address | Self::Organisation
        )
    }
}

#[derive(Debug, Clone)]
struct AudioDeviceOption {
    label: String,
    devices_value: String,
    main_gpu: i32,
    is_gpu: bool,
    detail_line: String,
}

#[derive(Debug, Clone)]
enum UiMessage {
    Log(String),
    Status(String),
    Refresh,
    RuntimeInstallDone(Result<PathBuf, String>),
    RuntimeUnblockDone(Result<String, String>),
    JobFileOk {
        job_id: u64,
        index: usize,
        total: usize,
        audio_path: PathBuf,
        result: TranscriptionResult,
    },
    JobFileErr {
        job_id: u64,
        index: usize,
        total: usize,
        audio_path: PathBuf,
        error: String,
    },
    JobDone(u64),
    ChatDone {
        source_path: PathBuf,
        user_prompt: String,
        answer: String,
    },
    ChatFailed(String),
    AnonymiseDone {
        answer: String,
        source_text: String,
    },
    AnonymiseFailed(String),
    PlaybackDecoded {
        audio_path: PathBuf,
        start_sec: f64,
        data: Vec<f32>,
    },
    PlaybackDecodeFailed {
        audio_path: PathBuf,
        error: String,
    },
    DownloadStatus(Option<String>),
    WhisperInstalled(PathBuf),
    DiarizationInstalled(PathBuf),
}

#[derive(Clone)]
struct UiModal {
    title: String,
    message: String,
    is_error: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LegalDocKind {
    ThirdPartyNotices,
    ThirdPartyLicenses,
    EngineThirdPartyLicenses,
}

impl LegalDocKind {
    const ALL: [Self; 3] = [
        Self::ThirdPartyNotices,
        Self::ThirdPartyLicenses,
        Self::EngineThirdPartyLicenses,
    ];

    fn nav_label(self) -> &'static str {
        match self {
            Self::ThirdPartyNotices => "Notices",
            Self::ThirdPartyLicenses => "Third-party licenses",
            Self::EngineThirdPartyLicenses => "Engine licenses",
        }
    }

    fn window_title(self) -> &'static str {
        match self {
            Self::ThirdPartyNotices => "Notices",
            Self::ThirdPartyLicenses => "Third-Party Licenses",
            Self::EngineThirdPartyLicenses => "Engine Licenses",
        }
    }

    fn bundled_markdown(self) -> &'static str {
        match self {
            Self::ThirdPartyNotices => BUNDLED_THIRD_PARTY_NOTICES_ALL_MD,
            Self::ThirdPartyLicenses => BUNDLED_THIRD_PARTY_LICENSES_ALL_MD,
            Self::EngineThirdPartyLicenses => BUNDLED_ENGINE_THIRD_PARTY_LICENSES_FULL_MD,
        }
    }
}

#[derive(Default)]
struct PlaybackBuffer {
    samples_stereo_f32: Vec<f32>,
    total_frames: usize,
    position_frames: f64,
    playing: bool,
    ended: bool,
    speed: f64,
    source_path: Option<PathBuf>,
}

struct PlaybackState {
    stream: Option<Stream>,
    stream_config: Option<StreamConfig>,
    sample_format: Option<SampleFormat>,
    shared: Option<Arc<Mutex<PlaybackBuffer>>>,
    output_sample_rate: u32,
    output_channels: u16,
    audio_path: Option<PathBuf>,
    speed: f64,
    last_seek_hotkey_at: Option<Instant>,
    seek_repeat_count: u32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            stream: None,
            stream_config: None,
            sample_format: None,
            shared: None,
            output_sample_rate: 0,
            output_channels: 0,
            audio_path: None,
            speed: 1.0,
            last_seek_hotkey_at: None,
            seek_repeat_count: 0,
        }
    }
}

struct UiApp {
    paths: AppPaths,
    settings: AppSettings,
    tx: mpsc::Sender<UiMessage>,
    rx: mpsc::Receiver<UiMessage>,
    runtime_state: Arc<Mutex<RuntimeState>>,

    tab: usize,
    status: String,
    status_log: Vec<String>,
    runtime_missing: Vec<String>,
    show_runtime_missing_window: bool,
    runtime_install_in_progress: bool,
    runtime_unblock_in_progress: bool,
    runtime_post_install_prompt: bool,
    runtime_missing_popup_active: bool,
    runtime_missing_on_popup_open: bool,
    show_about: bool,
    show_legal_docs_window: bool,
    legal_doc_kind: LegalDocKind,
    legal_doc_lines: Vec<String>,
    show_logs_window: bool,
    show_runtime_settings: bool,
    show_transcription_settings: bool,
    show_chat_settings: bool,
    show_editing_settings: bool,
    show_anonymise_window: bool,
    show_speaker_rename_window: bool,
    active_modal: Option<UiModal>,
    should_exit: bool,

    whisper_models: usize,
    chat_source_path: Option<PathBuf>,

    audio_devices: Vec<AudioDeviceOption>,
    selected_audio_device: usize,
    runtime_install_backends: Vec<String>,
    selected_runtime_install_backend: usize,

    transcript_state: TranscriptState,
    transcription_text: String,
    edited_text: String,
    editing_enabled: bool,
    edit_pause_until: Option<Instant>,
    edited_cues: Vec<TranscriptCue>,
    transcript_output_path: Option<PathBuf>,
    active_audio_path: Option<PathBuf>,
    transcription_preprocess_note: String,
    chat_history_text: String,
    chat_context_text: String,
    chat_input_text: String,
    is_transcribing: bool,
    is_chatting: bool,
    anonymise_running: bool,
    anonymise_selected_kinds: HashSet<AnonymiseEntityKind>,
    anonymise_last_response: String,
    selected_media_row: Option<usize>,
    selected_output_row: Option<usize>,
    playback_speed_text: String,
    playback_decode_in_progress: bool,
    playback_decode_autoplay: bool,
    playback_pending_start: Option<(PathBuf, f64)>,
    playback_follow_pause_until: Option<Instant>,
    playback_click_sync_target: Option<(f64, Instant)>,
    manual_edit_scroll_sync_y: Option<f32>,
    speaker_rename_entries: Vec<SpeakerRenameEntry>,

    playback: PlaybackState,
    fonts_configured: bool,
    settings_dirty: bool,
    next_save: Instant,
}

impl UiApp {
    fn push_log_only(&mut self, text: &str) {
        self.status_log
            .push(format!("[{}] {}", log_timestamp_hms(), text));
        if self.status_log.len() > 500 {
            let extra = self.status_log.len() - 500;
            self.status_log.drain(0..extra);
        }
    }

    fn new(paths: AppPaths) -> Self {
        let mut settings = match load_settings(&paths) {
            Ok(s) => s,
            Err(_) => AppSettings::default(),
        };
        let loaded_runtime_dir = settings.runtime_dir.clone();
        if settings.whisper_model.trim().is_empty() {
            settings.whisper_model = default_whisper_model_path(&paths).display().to_string();
        }
        if settings.diarization_models_dir.trim().is_empty() {
            settings.diarization_models_dir =
                default_diarization_models_dir(&paths).display().to_string();
        }
        let mode_norm = settings.mode.trim().to_ascii_lowercase();
        if !matches!(mode_norm.as_str(), "transcript" | "speech" | "subtitle") {
            settings.mode = "transcript".to_string();
        }
        let runtime_dir = resolve_runtime_dir(Path::new(settings.runtime_dir.trim()));
        settings.runtime_dir =
            settings::normalize_runtime_dir_alias(&runtime_dir.display().to_string());
        if !loaded_runtime_dir
            .trim()
            .eq_ignore_ascii_case(settings.runtime_dir.trim())
        {
            let _ = save_settings(&paths, &settings);
            let _ = save_model_links(&paths, &settings.to_model_links());
        }
        configure_runtime_dll_search(&runtime_dir);
        let runtime_missing = runtime_missing_messages(&runtime_dir);
        let setup_issues_present = !has_any_installed_whisper_model(&paths)
            || !missing_diarization_files(Path::new(settings.diarization_models_dir.trim()))
                .is_empty();
        let runtime_missing_window = !runtime_missing.is_empty() || setup_issues_present;
        let initial_status = if !runtime_missing.is_empty() {
            "Runtime missing or incomplete. Install/Repair runtime.".to_string()
        } else if setup_issues_present {
            "Setup incomplete. Download required Whisper/Diarization models.".to_string()
        } else {
            "Ready.".to_string()
        };

        let (tx, rx) = mpsc::channel();
        let runtime_state = Arc::new(Mutex::new(RuntimeState::default()));
        let audio_devices = enumerate_audio_device_options(&runtime_dir);
        let selected_audio_device = resolve_audio_device_index(&audio_devices, &settings);
        let mut runtime_install_backends = runtime_backend_options(&paths);
        let mut selected_runtime_install_backend = resolve_runtime_backend_index(
            &runtime_install_backends,
            &settings.runtime_download_backend,
        );
        let runtime_backend_detected = sync_runtime_backend_from_installed_runtime(
            &runtime_dir,
            &mut settings,
            &mut runtime_install_backends,
            &mut selected_runtime_install_backend,
        );
        let runtime_backend_synced = runtime_install_backends
            .get(selected_runtime_install_backend)
            .map(|backend| {
                if settings
                    .runtime_download_backend
                    .trim()
                    .eq_ignore_ascii_case(backend)
                {
                    false
                } else {
                    settings.runtime_download_backend = backend.clone();
                    true
                }
            })
            .unwrap_or(false)
            || runtime_backend_detected;
        let auto_device_applied = should_auto_select_gpu_from_settings(&settings)
            && apply_audio_device_to_settings_from_index(
                &mut settings,
                &audio_devices,
                selected_audio_device,
            );

        let whisper_models = select_whisper_model_index(&settings.whisper_model);

        let mut app = Self {
            paths,
            settings,
            tx,
            rx,
            runtime_state,
            tab: 0,
            status: initial_status.clone(),
            status_log: vec![initial_status],
            runtime_missing,
            show_runtime_missing_window: runtime_missing_window,
            runtime_install_in_progress: false,
            runtime_unblock_in_progress: false,
            runtime_post_install_prompt: false,
            runtime_missing_popup_active: false,
            runtime_missing_on_popup_open: false,
            show_about: false,
            show_legal_docs_window: false,
            legal_doc_kind: LegalDocKind::ThirdPartyNotices,
            legal_doc_lines: LegalDocKind::ThirdPartyNotices
                .bundled_markdown()
                .lines()
                .map(|line| line.to_string())
                .collect(),
            show_logs_window: false,
            show_runtime_settings: false,
            show_transcription_settings: false,
            show_chat_settings: false,
            show_editing_settings: false,
            show_anonymise_window: false,
            show_speaker_rename_window: false,
            active_modal: None,
            should_exit: false,
            whisper_models,
            chat_source_path: None,
            audio_devices,
            selected_audio_device,
            runtime_install_backends,
            selected_runtime_install_backend,
            transcript_state: TranscriptState::default(),
            transcription_text: String::new(),
            edited_text: String::new(),
            editing_enabled: false,
            edit_pause_until: None,
            edited_cues: Vec::new(),
            transcript_output_path: None,
            active_audio_path: None,
            transcription_preprocess_note: String::new(),
            chat_history_text: String::new(),
            chat_context_text: "Load or generate transcript output to use as chat context."
                .to_string(),
            chat_input_text: String::new(),
            is_transcribing: false,
            is_chatting: false,
            anonymise_running: false,
            anonymise_selected_kinds: ANONYMISE_ENTITY_ORDER.iter().copied().collect(),
            anonymise_last_response: String::new(),
            selected_media_row: None,
            selected_output_row: None,
            playback_speed_text: "1.0".to_string(),
            playback_decode_in_progress: false,
            playback_decode_autoplay: false,
            playback_pending_start: None,
            playback_follow_pause_until: None,
            playback_click_sync_target: None,
            manual_edit_scroll_sync_y: None,
            speaker_rename_entries: Vec::new(),
            playback: PlaybackState::default(),
            fonts_configured: false,
            settings_dirty: auto_device_applied || runtime_backend_synced,
            next_save: Instant::now(),
        };

        if auto_device_applied {
            app.push_log_only("Auto-selected macOS Metal GPU for runtime.");
        }

        app
    }

    fn push_status(&mut self, text: &str) {
        self.status = text.to_string();
        self.push_log_only(text);
    }

    fn queue_save(&mut self) {
        self.settings_dirty = true;
        self.next_save = Instant::now() + Duration::from_millis(300);
    }

    fn open_modal(&mut self, title: &str, message: impl Into<String>, is_error: bool) {
        self.active_modal = Some(UiModal {
            title: title.to_string(),
            message: message.into(),
            is_error,
        });
    }

    fn start_runtime_install(&mut self) {
        if self.runtime_install_in_progress || self.runtime_unblock_in_progress {
            return;
        }

        let runtime_dir = resolve_runtime_dir(Path::new(self.settings.runtime_dir.trim()));
        self.settings.runtime_dir =
            settings::normalize_runtime_dir_alias(&runtime_dir.display().to_string());
        self.runtime_post_install_prompt = false;
        self.runtime_install_in_progress = true;
        self.show_runtime_missing_window = true;
        let preferred_runtime_backend = self
            .runtime_install_backends
            .get(self.selected_runtime_install_backend)
            .cloned()
            .filter(|v| !v.trim().is_empty());
        if let Some(backend) = preferred_runtime_backend.as_deref() {
            self.push_status(&format!(
                "Starting runtime install/repair (backend: {backend})..."
            ));
        } else {
            self.push_status("Starting runtime install/repair...");
        }
        if let Ok(mut state) = self.runtime_state.lock() {
            state.download_status = Some("Downloading runtime: preparing...".to_string());
        }
        self.queue_save();

        let tx = self.tx.clone();
        let paths = self.paths.clone();
        let selected_backend = preferred_runtime_backend.clone();
        std::thread::spawn(move || {
            let tx_status = tx.clone();
            let mut last_status = String::new();
            let result = runtime_installer::install_or_repair_runtime_with_backend(
                &runtime_dir,
                &paths,
                selected_backend.as_deref(),
                |status| {
                    if status == last_status {
                        return;
                    }
                    last_status = status.clone();
                    let _ = tx_status.send(UiMessage::Status(status));
                },
            )
            .and_then(|installed_dir| {
                let missing = runtime_missing_messages(&installed_dir);
                if missing.is_empty() {
                    Ok(installed_dir)
                } else {
                    Err(anyhow!(
                        "Installed runtime is incomplete:\n{}",
                        missing.join("\n")
                    ))
                }
            })
            .map_err(|err| err.to_string());
            let _ = tx.send(UiMessage::RuntimeInstallDone(result));
            let _ = tx.send(UiMessage::Refresh);
        });
    }

    fn start_runtime_unblock(&mut self) {
        if self.runtime_unblock_in_progress || self.runtime_install_in_progress {
            return;
        }

        let runtime_dir = resolve_runtime_dir(Path::new(self.settings.runtime_dir.trim()));
        self.settings.runtime_dir =
            settings::normalize_runtime_dir_alias(&runtime_dir.display().to_string());
        self.runtime_unblock_in_progress = true;
        self.show_runtime_missing_window = true;
        self.push_status("Running unsigned runtime unblock script...");
        self.queue_save();

        let tx = self.tx.clone();
        let paths = self.paths.clone();
        std::thread::spawn(move || {
            let result = run_unsigned_runtime_unblock_script(&paths, &runtime_dir)
                .map_err(|err| err.to_string());
            let _ = tx.send(UiMessage::RuntimeUnblockDone(result));
            let _ = tx.send(UiMessage::Refresh);
        });
    }

    fn refresh_runtime_state(&mut self) {
        let runtime_dir = resolve_runtime_dir(Path::new(self.settings.runtime_dir.trim()));
        self.settings.runtime_dir =
            settings::normalize_runtime_dir_alias(&runtime_dir.display().to_string());
        configure_runtime_dll_search(&runtime_dir);
        self.runtime_missing = runtime_missing_messages(&runtime_dir);
        self.runtime_install_backends = runtime_backend_options(&self.paths);
        self.selected_runtime_install_backend = resolve_runtime_backend_index(
            &self.runtime_install_backends,
            &self.settings.runtime_download_backend,
        );
        if sync_runtime_backend_from_installed_runtime(
            &runtime_dir,
            &mut self.settings,
            &mut self.runtime_install_backends,
            &mut self.selected_runtime_install_backend,
        ) {
            self.queue_save();
        }
        self.show_runtime_missing_window =
            !self.runtime_missing.is_empty() || !self.runtime_setup_issues().is_empty();
    }

    fn sync_runtime_missing_popup_session(&mut self) {
        if self.show_runtime_missing_window {
            if !self.runtime_missing_popup_active {
                self.runtime_missing_popup_active = true;
                self.runtime_missing_on_popup_open = !self.runtime_missing.is_empty();
            }
        } else {
            self.runtime_missing_popup_active = false;
            self.runtime_missing_on_popup_open = false;
        }
    }

    fn runtime_busy(&self) -> bool {
        self.runtime_install_in_progress || self.runtime_unblock_in_progress
    }

    fn ui_runtime_management_controls(
        &mut self,
        ui: &mut egui::Ui,
        runtime_dir: &Path,
        auto_close_on_runtime_ready: bool,
    ) -> bool {
        let runtime_is_missing = !self.runtime_missing.is_empty();
        if !runtime_is_missing {
            ui.colored_label(
                egui::Color32::from_rgb(0x2E, 0x7D, 0x32),
                "Runtime appears complete.",
            );
        } else {
            ui.colored_label(
                egui::Color32::RED,
                "Engine runtime is missing or incomplete. Transcription/chat/playback require it.",
            );
            ui.colored_label(
                egui::Color32::from_rgb(160, 25, 25),
                "Required action: click 'Install/Repair runtime (required)'.",
            );
        }
        ui.label(format!("Runtime directory: {}", runtime_dir.display()));
        #[cfg(target_os = "windows")]
        ui.label(
            "Note: To switch runtime variant, close all apps using this runtime, delete the 'engine' runtime folder, reopen this app, then install the desired runtime variant.",
        );
        ui.separator();
        if !runtime_is_missing {
            ui.label("No missing runtime components detected.");
        } else {
            for item in &self.runtime_missing {
                ui.label(format!("- {item}"));
            }
        }
        if self.runtime_post_install_prompt {
            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(0x9A, 0x67, 0x00),
                "Runtime was just installed. Click 'Unblock unsigned runtime', then Recheck.",
            );
        }
        ui.separator();
        if self.runtime_busy() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Runtime maintenance in progress...");
            });
            ui.separator();
        }

        #[cfg(target_os = "windows")]
        {
            let runtime_busy = self.runtime_busy();
            ui.horizontal(|ui| {
                ui.label("Windows runtime variant:");
                if self.runtime_install_backends.is_empty() {
                    ui.label("default");
                    return;
                }

                let selected_text = self
                    .runtime_install_backends
                    .get(self.selected_runtime_install_backend)
                    .cloned()
                    .unwrap_or_else(|| self.runtime_install_backends[0].clone());
                let mut selected_index = self
                    .selected_runtime_install_backend
                    .min(self.runtime_install_backends.len().saturating_sub(1));

                if runtime_busy {
                    ui.add_enabled_ui(false, |ui| {
                        egui::ComboBox::from_id_salt("runtime_backend_windows_combo")
                            .selected_text(selected_text.to_ascii_uppercase())
                            .show_ui(ui, |_| {});
                    });
                } else {
                    egui::ComboBox::from_id_salt("runtime_backend_windows_combo")
                        .selected_text(selected_text.to_ascii_uppercase())
                        .show_ui(ui, |ui| {
                            for (idx, backend) in self.runtime_install_backends.iter().enumerate() {
                                let label = backend.to_ascii_uppercase();
                                ui.selectable_value(&mut selected_index, idx, label);
                            }
                        });
                }

                if selected_index != self.selected_runtime_install_backend {
                    self.selected_runtime_install_backend = selected_index;
                    if let Some(backend) = self
                        .runtime_install_backends
                        .get(self.selected_runtime_install_backend)
                    {
                        self.settings.runtime_download_backend = backend.clone();
                        self.queue_save();
                    }
                }
            });
            if let Some(detected) = detect_installed_windows_runtime_backend(runtime_dir) {
                ui.label(format!(
                    "Detected installed runtime backend: {}",
                    detected.to_ascii_uppercase()
                ));
            }
            ui.separator();
        }

        let mut runtime_ready_now = false;
        ui.horizontal(|ui| {
            let runtime_busy = self.runtime_busy();
            if runtime_busy {
                ui.add_enabled(false, egui::Button::new("Unblock unsigned runtime"));
            } else if secondary_button(ui, "Unblock unsigned runtime").clicked() {
                self.start_runtime_unblock();
            }
            if runtime_busy {
                let label = if runtime_is_missing {
                    "Install/Repair runtime (required)"
                } else {
                    "Install/Repair runtime"
                };
                ui.add_enabled(false, egui::Button::new(label));
            } else {
                let clicked = if runtime_is_missing {
                    warning_button(ui, "Install/Repair runtime (required)").clicked()
                } else {
                    secondary_button(ui, "Install/Repair runtime").clicked()
                };
                if clicked {
                    self.start_runtime_install();
                }
            }
            if ui.button("Open runtime folder").clicked() {
                let _ = open::that(runtime_dir);
            }
            if ui.button("Recheck").clicked() {
                self.refresh_runtime_state();
                if self.runtime_missing.is_empty() {
                    self.push_status("Runtime detected.");
                    runtime_ready_now = true;
                }
            }
        });

        runtime_ready_now && auto_close_on_runtime_ready && !self.runtime_post_install_prompt
    }

    fn ensure_runtime_ready(&mut self, action: &str) -> bool {
        if self.runtime_install_in_progress || self.runtime_unblock_in_progress {
            self.push_status(&format!(
                "{action} blocked: runtime maintenance is in progress."
            ));
            self.show_runtime_missing_window = true;
            return false;
        }
        self.refresh_runtime_state();
        if self.runtime_missing.is_empty() {
            return true;
        }
        self.push_status(&format!("{action} blocked: runtime missing or incomplete."));
        false
    }

    fn ui_modal(&mut self, ctx: &egui::Context) {
        if self.active_modal.is_none() {
            return;
        }

        let Some(modal) = self.active_modal.take() else {
            return;
        };
        let mut is_open = true;
        let mut close_clicked = false;
        let title = modal.title.clone();
        egui::Window::new(title)
            .collapsible(false)
            .resizable(false)
            .open(&mut is_open)
            .show(ctx, |ui| {
                if modal.is_error {
                    ui.colored_label(egui::Color32::RED, &modal.message);
                } else {
                    ui.label(&modal.message);
                }
                ui.separator();
                if ui.button("Close").clicked() {
                    close_clicked = true;
                }
            });

        if close_clicked {
            is_open = false;
        }

        if is_open {
            self.active_modal = Some(modal);
        }
    }

    fn set_legal_doc_kind(&mut self, kind: LegalDocKind) {
        self.legal_doc_kind = kind;
        self.legal_doc_lines = self
            .legal_doc_kind
            .bundled_markdown()
            .lines()
            .map(|line| line.to_string())
            .collect();
    }

    fn open_legal_doc(&mut self, kind: LegalDocKind) {
        self.set_legal_doc_kind(kind);
        self.show_legal_docs_window = true;
    }

    fn ui_about(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        let mut open = true;
        egui::Window::new("About Transcribe Offline")
            .collapsible(false)
            .resizable(true)
            .default_size([920.0, 700.0])
            .min_size([780.0, 560.0])
            .open(&mut open)
            .show(ctx, |ui| {
                ui.heading("Transcribe Offline");
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.label("Native desktop GUI around Openresearchtools-Engine (llama-server-bridge runtime).");
                ui.label("Copyright (c) 2026 L. Rutkauskas");
                ui.separator();
                ui.label("What this app does:");
                ui.label("- Offline transcription of media files into markdown transcripts.");
                ui.label("- Speaker diarization with timestamped speaker turns.");
                ui.label("- Side-by-side transcript editing with audio-linked navigation.");
                ui.label("- Optional anonymisation workflow using a local chat model.");
                ui.label("The engine runtime performs inference/media processing; this desktop app provides setup, queueing, controls, review, and export UI.");
                ui.separator();
                ui.label("Bundled legal documents (embedded in this executable):");
                ui.horizontal_wrapped(|ui| {
                    if secondary_button(ui, "Notices").clicked() {
                        self.open_legal_doc(LegalDocKind::ThirdPartyNotices);
                    }
                    if secondary_button(ui, "Third-party licenses").clicked() {
                        self.open_legal_doc(LegalDocKind::ThirdPartyLicenses);
                    }
                    if secondary_button(ui, "Engine licenses").clicked() {
                        self.open_legal_doc(LegalDocKind::EngineThirdPartyLicenses);
                    }
                });
                ui.separator();
                ui.label("Transcribe Offline license:");
                let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
                let license_lines: Vec<&str> = BUNDLED_TRANSCRIBE_OFFLINE_LICENSE_TXT.lines().collect();
                egui::ScrollArea::vertical()
                    .id_salt("about_app_license_scroll")
                    .max_height(320.0)
                    .auto_shrink([false, false])
                    .show_rows(ui, row_height, license_lines.len(), |ui, row_range| {
                        for row in row_range {
                            if let Some(line) = license_lines.get(row) {
                                ui.add(
                                    egui::Label::new(egui::RichText::new(*line).monospace())
                                        .wrap_mode(egui::TextWrapMode::Extend),
                                );
                            }
                        }
                    });
            });
        if !open {
            self.show_about = false;
        }
    }

    fn ui_legal_docs_window(&mut self, ctx: &egui::Context) {
        if !self.show_legal_docs_window {
            return;
        }

        let mut still_open = true;
        let viewport_id = egui::ViewportId::from_hash_of("legal-docs-window");
        let window_title = format!(
            "Transcribe Offline - {}",
            self.legal_doc_kind.window_title()
        );
        let builder = egui::ViewportBuilder::default()
            .with_title(window_title)
            .with_inner_size([1120.0, 760.0])
            .with_resizable(true);

        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            if ctx.input(|i| i.viewport().close_requested()) {
                still_open = false;
            }

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if accent_button(ui, "Close").clicked() {
                            still_open = false;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    for kind in LegalDocKind::ALL {
                        if ui
                            .selectable_label(self.legal_doc_kind == kind, kind.nav_label())
                            .clicked()
                        {
                            self.set_legal_doc_kind(kind);
                        }
                    }
                });
                ui.separator();
                ui.label(format!("Document: {}", self.legal_doc_kind.window_title()));
                ui.separator();
                let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
                egui::ScrollArea::both()
                    .id_salt("legal_docs_scroll")
                    .auto_shrink([false, false])
                    .show_rows(
                        ui,
                        row_height,
                        self.legal_doc_lines.len(),
                        |ui, row_range| {
                            for row in row_range {
                                if let Some(line) = self.legal_doc_lines.get(row) {
                                    ui.add(
                                        egui::Label::new(egui::RichText::new(line).monospace())
                                            .wrap_mode(egui::TextWrapMode::Extend),
                                    );
                                }
                            }
                        },
                    );
            });
        });

        self.show_legal_docs_window = still_open;
    }

    fn ui_runtime_missing_window(&mut self, ctx: &egui::Context) {
        if !self.show_runtime_missing_window {
            return;
        }
        let mut open = true;
        let mut close_clicked = false;
        let runtime_dir = resolve_runtime_dir(Path::new(self.settings.runtime_dir.trim()));
        let setup_issues = self.runtime_setup_issues();
        let unblock_required =
            self.runtime_missing_on_popup_open && self.runtime_post_install_prompt;
        let blocking_missing = !self.runtime_missing.is_empty() || !setup_issues.is_empty();
        let must_keep_open = self.runtime_install_in_progress
            || self.runtime_unblock_in_progress
            || blocking_missing
            || unblock_required;
        egui::Window::new("Runtime Setup Required")
            .collapsible(false)
            .resizable(true)
            .default_size([980.0, 720.0])
            .open(&mut open)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("runtime_setup_required_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.label("Complete required setup once and the app is ready.");
                        ui.label("* Required: Runtime, Whisper model, Diarization models.");
                        let _ = self.ui_runtime_management_controls(ui, &runtime_dir, false);
                        ui.separator();
                        self.ui_runtime_device_panel(ui);
                        self.ui_runtime_models_panel(ui, true);
                        self.ui_runtime_parameters_panel(ui);
                        ui.separator();
                        if unblock_required {
                            ui.colored_label(
                                egui::Color32::from_rgb(160, 25, 25),
                                "Runtime was missing on first open and just installed. Run 'Unblock unsigned runtime' before closing.",
                            );
                        }
                        if !self.runtime_missing.is_empty() {
                            ui.colored_label(
                                egui::Color32::from_rgb(160, 25, 25),
                                "Required runtime files are still missing.",
                            );
                        }
                        if !setup_issues.is_empty() {
                            for issue in &setup_issues {
                                ui.colored_label(egui::Color32::from_rgb(160, 25, 25), issue);
                            }
                        }
                        ui.horizontal(|ui| {
                            if accent_button(ui, "Save settings").clicked() {
                                self.queue_save();
                                self.persist_if_needed();
                            }
                            if must_keep_open {
                                ui.add_enabled(false, egui::Button::new("Close"));
                            } else if ui.button("Close").clicked() {
                                close_clicked = true;
                            }
                        });
                    });
            });
        if must_keep_open {
            open = true;
        } else if close_clicked {
            open = false;
        }
        self.show_runtime_missing_window = open
            && (!self.runtime_missing.is_empty()
                || self.runtime_install_in_progress
                || self.runtime_unblock_in_progress
                || self.runtime_post_install_prompt
                || !setup_issues.is_empty()
                || unblock_required);
    }

    fn ui_runtime_settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_runtime_settings {
            return;
        }
        let mut open = self.show_runtime_settings;
        let mut close_clicked = false;
        let runtime_dir = resolve_runtime_dir(Path::new(self.settings.runtime_dir.trim()));
        egui::Window::new("Runtime Setup")
            .collapsible(false)
            .resizable(true)
            .default_size([980.0, 720.0])
            .open(&mut open)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("runtime_setup_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.label("Runtime setup: install/unblock runtime, set device, required models, and optional chat model.");
                        ui.label("* Required: Runtime, Whisper model, Diarization models.");
                        let _ = self.ui_runtime_management_controls(ui, &runtime_dir, false);
                        ui.separator();
                        self.ui_runtime_device_panel(ui);
                        self.ui_runtime_models_panel(ui, true);
                        self.ui_runtime_parameters_panel(ui);
                        ui.separator();
                        ui.horizontal(|ui| {
                            if accent_button(ui, "Save settings").clicked() {
                                self.queue_save();
                                self.persist_if_needed();
                            }
                            if ui.button("Close").clicked() {
                                close_clicked = true;
                            }
                        });
                    });
            });
        if close_clicked {
            open = false;
        }
        self.show_runtime_settings = open;
    }

    fn ui_menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Mode", |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                if native_menu_item(ui, "Transcription").clicked() {
                    self.tab = 0;
                    ui.close();
                }
                if native_menu_item(ui, "Chat").clicked() {
                    self.tab = 1;
                    ui.close();
                }
            });

            ui.menu_button("Settings", |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                if native_menu_item(ui, "Runtime").clicked() {
                    self.show_runtime_settings = true;
                    ui.close();
                }
                if native_menu_item(ui, "Transcription settings").clicked() {
                    self.show_transcription_settings = true;
                    ui.close();
                }
                if native_menu_item(ui, "Chat settings").clicked() {
                    self.show_chat_settings = true;
                    ui.close();
                }
                if native_menu_item(ui, "Editing settings").clicked() {
                    self.show_editing_settings = true;
                    ui.close();
                }
            });

            ui.menu_button("View", |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                if native_menu_item(ui, "Increase text size").clicked() {
                    self.settings.ui_font_size_px =
                        clamp_font_px(self.settings.ui_font_size_px + UI_FONT_SIZE_STEP_PX);
                    self.queue_save();
                    ui.close();
                }
                if native_menu_item(ui, "Decrease text size").clicked() {
                    self.settings.ui_font_size_px =
                        clamp_font_px(self.settings.ui_font_size_px - UI_FONT_SIZE_STEP_PX);
                    self.queue_save();
                    ui.close();
                }
                if native_menu_item(ui, "Reset text size").clicked() {
                    self.settings.ui_font_size_px = UI_FONT_SIZE_DEFAULT_PX;
                    self.queue_save();
                    ui.close();
                }
            });

            ui.menu_button("Help", |ui| {
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                if native_menu_item(ui, "About").clicked() {
                    self.show_about = true;
                    ui.close();
                }
                if native_menu_item(ui, "Notices").clicked() {
                    self.open_legal_doc(LegalDocKind::ThirdPartyNotices);
                    ui.close();
                }
                if native_menu_item(ui, "Third-party licenses").clicked() {
                    self.open_legal_doc(LegalDocKind::ThirdPartyLicenses);
                    ui.close();
                }
                if native_menu_item(ui, "Engine licenses").clicked() {
                    self.open_legal_doc(LegalDocKind::EngineThirdPartyLicenses);
                    ui.close();
                }
            });
        });
    }

    fn persist_if_needed(&mut self) {
        if !self.settings_dirty || self.next_save > Instant::now() {
            return;
        }

        if let Err(err) = save_settings(&self.paths, &self.settings) {
            self.push_status(&format!("Failed to save settings: {err}"));
        } else if let Err(err) = save_model_links(&self.paths, &self.settings.to_model_links()) {
            self.push_status(&format!("Failed to save model links: {err}"));
        } else {
            self.settings_dirty = false;
            self.next_save = Instant::now() + Duration::from_secs(1);
        }
    }

    fn process_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                UiMessage::Log(line) => self.push_status(&line),
                UiMessage::Status(text) => {
                    let text_lc = text.to_ascii_lowercase();
                    if let Ok(mut state) = self.runtime_state.lock() {
                        if text_lc.contains("downloading ")
                            || text_lc.starts_with("extracting runtime")
                        {
                            state.download_status = Some(text.clone());
                        }
                    }
                    self.push_status(&text);
                }
                UiMessage::Refresh => {}
                UiMessage::RuntimeInstallDone(result) => {
                    self.runtime_install_in_progress = false;
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.download_status = None;
                    }
                    match result {
                        Ok(runtime_dir) => {
                            self.settings.runtime_dir = settings::normalize_runtime_dir_alias(
                                &runtime_dir.display().to_string(),
                            );
                            configure_runtime_dll_search(&runtime_dir);
                            self.runtime_missing = runtime_missing_messages(&runtime_dir);
                            if self.runtime_missing.is_empty() {
                                self.runtime_post_install_prompt = true;
                                self.show_runtime_missing_window = true;
                                self.push_status(
                                    "Runtime install complete. Run 'Unblock unsigned runtime', then Recheck.",
                                );
                            } else {
                                self.runtime_post_install_prompt = false;
                                self.show_runtime_missing_window = true;
                                self.push_status(
                                    "Runtime install finished but validation still reports missing components.",
                                );
                            }
                            self.audio_devices = enumerate_audio_device_options(&runtime_dir);
                            self.selected_audio_device =
                                resolve_audio_device_index(&self.audio_devices, &self.settings);
                            self.runtime_install_backends = runtime_backend_options(&self.paths);
                            self.selected_runtime_install_backend = resolve_runtime_backend_index(
                                &self.runtime_install_backends,
                                &self.settings.runtime_download_backend,
                            );
                            if let Some(backend) = self
                                .runtime_install_backends
                                .get(self.selected_runtime_install_backend)
                            {
                                self.settings.runtime_download_backend = backend.clone();
                            }
                            sync_runtime_backend_from_installed_runtime(
                                &runtime_dir,
                                &mut self.settings,
                                &mut self.runtime_install_backends,
                                &mut self.selected_runtime_install_backend,
                            );
                            if should_auto_select_gpu_from_settings(&self.settings)
                                && apply_audio_device_to_settings_from_index(
                                    &mut self.settings,
                                    &self.audio_devices,
                                    self.selected_audio_device,
                                )
                            {
                                self.push_log_only("Auto-selected macOS Metal GPU for runtime.");
                            }
                            self.queue_save();
                        }
                        Err(error) => {
                            self.runtime_post_install_prompt = false;
                            self.show_runtime_missing_window = true;
                            self.push_status(&format!("Runtime install failed: {error}"));
                            self.open_modal("Runtime install failed", error, true);
                        }
                    }
                }
                UiMessage::RuntimeUnblockDone(result) => {
                    self.runtime_unblock_in_progress = false;
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.download_status = None;
                    }
                    match result {
                        Ok(message) => {
                            self.runtime_post_install_prompt = false;
                            self.push_status(&message);
                            self.refresh_runtime_state();
                            if self.runtime_missing.is_empty()
                                && self.runtime_setup_issues().is_empty()
                            {
                                self.show_runtime_missing_window = false;
                                self.push_status("Runtime/setup detected.");
                            }
                        }
                        Err(error) => {
                            self.show_runtime_missing_window = true;
                            self.push_status(&format!("Unsigned runtime unblock failed: {error}"));
                            self.open_modal("Unsigned runtime unblock failed", error, true);
                        }
                    }
                }
                UiMessage::JobFileOk {
                    job_id,
                    index,
                    total,
                    audio_path,
                    result,
                } => {
                    let edited_path = edited_file_path(&result.output_path);
                    // Always reset edited mirror from fresh transcript on new run.
                    let edited_text = result.output_text.clone();
                    if let Some(parent) = edited_path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    if let Err(err) = fs::write(&edited_path, &edited_text) {
                        self.push_log_only(&format!(
                            "Failed to reset edited output '{}': {err}",
                            edited_path.display()
                        ));
                    }

                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.original_output_path = Some(result.output_path.clone());
                        state.edited_output_path = Some(edited_path.clone());
                        state.active_audio_path = Some(audio_path.clone());
                        ensure_output_entry(
                            &mut state.output_entries,
                            result.output_path.clone(),
                            false,
                        );
                        ensure_output_entry(&mut state.output_entries, edited_path, false);
                        if let Some(active) = &mut state.active_job {
                            active.done_files = index;
                            active.total_files = total;
                            active.id = job_id;
                        }
                    }

                    self.active_audio_path = Some(audio_path.clone());
                    self.settings.audio_file = audio_path.display().to_string();
                    self.transcription_preprocess_note = result.preprocess_note;
                    self.transcription_text = result.output_text.clone();
                    self.transcript_output_path = Some(result.output_path.clone());
                    self.edited_text = edited_text.clone();
                    self.edited_cues = parse_transcript(&edited_text).cues;
                    self.transcript_state = parse_transcript(&self.transcription_text);
                    self.push_status(&format!(
                        "Job #{job_id} file {index}/{total} complete in {:.2}s",
                        result.timing_total_ms as f64 / 1000.0
                    ));
                    self.push_log_only(&format!(
                        "Job #{job_id} file {index}/{total} details: audio='{}' mode='{}' custom='{}' whisper_model='{}' diarization={} audio_bytes={} output='{}' timings_ms[prepare={}, read_audio={}, bridge={}, read_output={}, total={}]",
                        audio_path.display(),
                        result.mode,
                        result.custom_value,
                        result.whisper_model,
                        result.diarization_enabled,
                        result.audio_bytes_len,
                        result.output_path.display(),
                        result.timing_prepare_ms,
                        result.timing_read_audio_ms,
                        result.timing_bridge_ms,
                        result.timing_read_output_ms,
                        result.timing_total_ms
                    ));
                    if let Ok(mut bridge_json) =
                        serde_json::to_string_pretty(&result._response_json)
                    {
                        if bridge_json.len() > 16_000 {
                            bridge_json.truncate(16_000);
                            bridge_json.push_str("\n... [truncated]");
                        }
                        self.push_log_only(&format!(
                            "Job #{job_id} file {index}/{total} bridge response json:\n{bridge_json}"
                        ));
                    }
                    self.ensure_chat_source_selected();
                    self.queue_save();
                }
                UiMessage::JobFileErr {
                    job_id,
                    index,
                    total,
                    audio_path,
                    error,
                } => {
                    self.push_status(&format!(
                        "Job #{job_id} file {index}/{total} failed ({}): {error}",
                        audio_path.display()
                    ));
                }
                UiMessage::JobDone(job_id) => {
                    self.push_status(&format!("Job #{job_id} complete."));
                    let is_idle = self
                        .runtime_state
                        .lock()
                        .ok()
                        .map(|s| s.running_jobs == 0 && s.job_queue.is_empty())
                        .unwrap_or(true);
                    if is_idle {
                        self.is_transcribing = false;
                    }
                }
                UiMessage::ChatDone {
                    source_path,
                    user_prompt,
                    answer,
                } => {
                    self.is_chatting = false;
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.running_jobs = state.running_jobs.saturating_sub(1);
                        state.completed_jobs += 1;
                    }
                    match self.append_chat_log_entry(&source_path, &user_prompt, &answer) {
                        Ok(log_path) => {
                            if self
                                .chat_source_path
                                .as_ref()
                                .map(|p| p == &source_path)
                                .unwrap_or(false)
                            {
                                let _ = self.load_chat_source_file(source_path.clone());
                            }
                            self.push_status(&format!(
                                "Chat complete. Log saved: {}",
                                log_path.display()
                            ));
                        }
                        Err(err) => {
                            self.push_status(&format!(
                                "Chat complete, but failed to save log: {err}"
                            ));
                        }
                    }
                }
                UiMessage::ChatFailed(error) => {
                    self.is_chatting = false;
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.running_jobs = state.running_jobs.saturating_sub(1);
                        state.completed_jobs += 1;
                    }
                    self.push_status(&format!("Chat failed: {error}"));
                }
                UiMessage::AnonymiseDone {
                    answer,
                    source_text,
                } => {
                    self.anonymise_running = false;
                    self.anonymise_last_response = answer.clone();
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.running_jobs = state.running_jobs.saturating_sub(1);
                        state.completed_jobs += 1;
                    }
                    match self.apply_anonymise_response(&answer, &source_text) {
                        Ok(result) => {
                            self.push_status(&format!(
                                "Anonymise complete: {} extracted ({} usable), {} exact + {} fuzzy replacements. Saved: {}",
                                result.total_matches,
                                result.usable_matches,
                                result.exact_replacements,
                                result.fuzzy_replacements,
                                result.output_path.display()
                            ));
                        }
                        Err(err) => {
                            self.push_status(&format!("Anonymise failed: {err}"));
                            self.open_modal("Anonymise failed", err.to_string(), true);
                        }
                    }
                }
                UiMessage::AnonymiseFailed(error) => {
                    self.anonymise_running = false;
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.running_jobs = state.running_jobs.saturating_sub(1);
                        state.completed_jobs += 1;
                    }
                    self.push_status(&format!("Anonymise failed: {error}"));
                    self.open_modal("Anonymise failed", error, true);
                }
                UiMessage::PlaybackDecoded {
                    audio_path,
                    start_sec,
                    data,
                } => {
                    self.playback_decode_in_progress = false;
                    let autoplay = self.playback_decode_autoplay;
                    self.playback_decode_autoplay = false;
                    if autoplay {
                        match playback_set_decoded_buffer(
                            &mut self.playback,
                            &audio_path,
                            data,
                            start_sec,
                        ) {
                            Ok(now) => self.push_status(&format!(
                                "Playback started {:.2}s ({})",
                                now,
                                audio_path.display()
                            )),
                            Err(err) => self.push_status(&format!("Playback start failed: {err}")),
                        }
                    } else {
                        match playback_set_decoded_buffer_paused(
                            &mut self.playback,
                            &audio_path,
                            data,
                        ) {
                            Ok(_) => self.push_log_only(&format!(
                                "Playback preloaded to RAM: {}",
                                audio_path.display()
                            )),
                            Err(err) => {
                                self.push_status(&format!("Playback preload failed: {err}"))
                            }
                        }
                    }
                    if let Some((pending_path, pending_start)) = self.playback_pending_start.take()
                    {
                        if pending_path == audio_path {
                            match playback_start_from(
                                &mut self.playback,
                                &pending_path,
                                pending_start,
                            ) {
                                Ok(_) => self.push_status(&format!(
                                    "Playback queued seek {:.2}s",
                                    pending_start.max(0.0)
                                )),
                                Err(err) => {
                                    self.push_status(&format!("Playback queued seek failed: {err}"))
                                }
                            }
                        } else {
                            self.request_playback_start(
                                pending_path,
                                pending_start,
                                "queued playback request",
                            );
                        }
                    }
                }
                UiMessage::PlaybackDecodeFailed { audio_path, error } => {
                    self.playback_decode_in_progress = false;
                    self.playback_decode_autoplay = false;
                    self.push_status(&format!(
                        "Playback decode failed ({}): {error}",
                        audio_path.display()
                    ));
                    if let Some((pending_path, pending_start)) = self.playback_pending_start.take()
                    {
                        self.request_playback_start(
                            pending_path,
                            pending_start,
                            "queued playback request",
                        );
                    }
                }
                UiMessage::DownloadStatus(status) => {
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.download_status = status;
                    }
                }
                UiMessage::WhisperInstalled(path) => {
                    self.settings.whisper_model = path.display().to_string();
                    self.whisper_models = select_whisper_model_index(&self.settings.whisper_model);
                    self.push_status(&format!("Whisper model installed: {}", path.display()));
                    self.queue_save();
                }
                UiMessage::DiarizationInstalled(path) => {
                    self.settings.diarization_models_dir = path.display().to_string();
                    self.push_status(&format!("Diarization models installed: {}", path.display()));
                    self.queue_save();
                }
            }
        }
    }

    fn apply_audio_device(&mut self) {
        if let Some(option) = self.audio_devices.get(self.selected_audio_device) {
            if option.is_gpu {
                self.settings.whisper_no_gpu = false;
                self.settings.main_gpu = option.main_gpu;
                self.settings.devices = option.devices_value.clone();
            } else {
                self.settings.whisper_no_gpu = true;
                self.settings.devices.clear();
                self.settings.main_gpu = 0;
            }
            self.queue_save();
        }
    }

    fn ensure_whisper_model_path_for_run(&mut self) -> bool {
        let configured = self.settings.whisper_model.trim();
        if !configured.is_empty() && Path::new(configured).exists() {
            return true;
        }
        if let Some(spec) = first_installed_whisper_model(&self.paths) {
            let path = whisper_model_dest_path(&self.paths, spec.file_name);
            self.settings.whisper_model = path.display().to_string();
            self.whisper_models = select_whisper_model_index(&self.settings.whisper_model);
            self.queue_save();
            return true;
        }
        false
    }

    fn runtime_setup_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if !has_any_installed_whisper_model(&self.paths) {
            issues.push(
                "No Whisper model installed. Download at least one Whisper model.".to_string(),
            );
        }
        let diar_dir = Path::new(self.settings.diarization_models_dir.trim());
        let missing = missing_diarization_files(diar_dir);
        if !missing.is_empty() {
            issues.push(format!(
                "Diarization model pack is incomplete. Missing: {}.",
                missing.join(", ")
            ));
        }
        issues
    }

    fn ensure_transcription_setup_ready(&mut self) -> bool {
        let _ = self.ensure_whisper_model_path_for_run();
        let issues = self.runtime_setup_issues();
        if issues.is_empty() {
            return true;
        }
        self.show_runtime_missing_window = true;
        self.push_status(&format!("Transcription blocked: {}", issues.join(" | ")));
        false
    }

    fn ui_runtime_device_panel(&mut self, ui: &mut egui::Ui) {
        let is_gpu = self
            .audio_devices
            .get(self.selected_audio_device)
            .map(|o| o.is_gpu)
            .unwrap_or(false);
        engine_panel_frame().show(ui, |ui| {
            ui.heading("Execution Device");
            ui.horizontal(|ui| {
                ui.label("Execution device:");
                let before = self.selected_audio_device;
                egui::ComboBox::from_id_salt("device_combo_runtime_settings")
                    .selected_text(
                        self.audio_devices
                            .get(self.selected_audio_device)
                            .map(|o| o.label.clone())
                            .unwrap_or_else(|| AUDIO_DEVICE_CPU_LABEL.to_string()),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, opt) in self.audio_devices.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_audio_device, idx, &opt.label);
                        }
                    });
                if before != self.selected_audio_device {
                    self.apply_audio_device();
                }
            });
            ui.label(resolve_audio_device_label(
                &self.audio_devices,
                selected_gpu_index_from_settings(&self.settings),
            ));
            ui.label(if is_gpu {
                "GPU selected: app sends only the selected GPU index to runtime."
            } else {
                "CPU selected: app sends no GPU flag, runtime runs CPU mode."
            });
            ui.label("Detected devices");
            let mut detected = self
                .audio_devices
                .iter()
                .map(|o| o.detail_line.clone())
                .collect::<Vec<_>>()
                .join("\n");
            ui.add(
                egui::TextEdit::multiline(&mut detected)
                    .desired_rows(4)
                    .desired_width(f32::INFINITY)
                    .interactive(false),
            );
            if !is_gpu {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("CPU threads:");
                    ui.add(egui::DragValue::new(&mut self.settings.n_threads).range(0..=256));
                    ui.label("CPU batch threads:");
                    ui.add(egui::DragValue::new(&mut self.settings.n_threads_batch).range(0..=256));
                });
            }
        });
    }

    fn ui_runtime_models_panel(&mut self, ui: &mut egui::Ui, include_chat_model: bool) {
        let whisper_any_installed = has_any_installed_whisper_model(&self.paths);
        engine_panel_frame().show(ui, |ui| {
            ui.heading("Whisper Model *");
            ui.horizontal(|ui| {
                ui.label("Whisper model:");
                egui::ComboBox::from_id_salt("whisper_model_runtime_settings")
                    .selected_text(
                        WHISPER_MODELS
                            .get(self.whisper_models)
                            .map(|m| whisper_combo_label(&self.paths, m))
                            .unwrap_or_else(|| "Select model".to_string()),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, spec) in WHISPER_MODELS.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.whisper_models,
                                idx,
                                whisper_combo_label(&self.paths, spec),
                            );
                        }
                    });
                let download_clicked = if !whisper_any_installed {
                    warning_button(ui, "Download (required)").clicked()
                } else {
                    accent_button(ui, "Download").clicked()
                };
                if download_clicked {
                    self.do_download_whisper();
                }
                if ui.button("Open models folder").clicked() {
                    let _ = open::that(self.paths.models_dir.clone());
                }
            });
            if let Some(spec) = WHISPER_MODELS.get(self.whisper_models) {
                let selected_path = whisper_model_dest_path(&self.paths, spec.file_name)
                    .display()
                    .to_string();
                if self.settings.whisper_model != selected_path {
                    self.settings.whisper_model = selected_path;
                    self.queue_save();
                }
            }
            let configured_exists = Path::new(self.settings.whisper_model.trim()).exists();
            if whisper_any_installed {
                if configured_exists {
                    ui.label("Installed in your app data models folder.");
                } else {
                    ui.label("Installed (at least one model). Choose an installed model or download selected.");
                }
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(160, 25, 25),
                    "Not installed yet. Click Download to install this Whisper model.",
                );
            }
        });

        engine_panel_frame().show(ui, |ui| {
            ui.heading("Diarization Models (Pyannote GGUF) *");
            let missing =
                missing_diarization_files(Path::new(self.settings.diarization_models_dir.trim()));
            ui.horizontal(|ui| {
                let download_clicked = if missing.is_empty() {
                    accent_button(ui, "Download diarization pack").clicked()
                } else {
                    warning_button(ui, "Download diarization pack (required)").clicked()
                };
                if download_clicked {
                    self.do_download_diarization();
                }
                if ui.button("Open diarization folder").clicked() {
                    let _ = open::that(self.paths.diarization_models_dir.clone());
                }
            });
            if missing.is_empty() {
                ui.label("Installed in your app data models folder.");
            } else {
                ui.colored_label(
                    egui::Color32::from_rgb(160, 25, 25),
                    format!("Missing {} file(s): {}", missing.len(), missing.join(", ")),
                );
            }
        });

        if include_chat_model {
            engine_panel_frame().show(ui, |ui| {
                ui.heading("Chat Model (Optional)");
                ui.horizontal(|ui| {
                    ui.label("Chat model file:");
                    ui.text_edit_singleline(&mut self.settings.chat_model);
                    if ui.button("Browse").clicked() {
                        self.do_pick_chat_model();
                    }
                });
                ui.label("Optional for chat. Also required for Anonymise (beta).");
            });
        }
    }

    fn ui_runtime_parameters_panel(&mut self, ui: &mut egui::Ui) {
        let is_gpu = self
            .audio_devices
            .get(self.selected_audio_device)
            .map(|o| o.is_gpu)
            .unwrap_or(false);
        let selected_device_label = self
            .audio_devices
            .get(self.selected_audio_device)
            .map(|o| o.label.clone())
            .unwrap_or_else(|| AUDIO_DEVICE_CPU_LABEL.to_string());
        engine_panel_frame().show(ui, |ui| {
            ui.heading("Engine Runtime Parameters *");
            ui.label("Applies to transcription and optional chat.");
            ui.label(format!(
                "Device: {} (selected in the Runtime device dropdown above)",
                selected_device_label
            ));
            ui.horizontal(|ui| {
                ui.label("n_ctx:");
                ui.add(egui::DragValue::new(&mut self.settings.n_ctx).range(-1..=131072));
                ui.label("n_batch:");
                ui.add(egui::DragValue::new(&mut self.settings.n_batch).range(1..=8192));
                ui.label("n_ubatch:");
                ui.add(egui::DragValue::new(&mut self.settings.n_ubatch).range(1..=8192));
                ui.label("n_parallel:");
                ui.add(egui::DragValue::new(&mut self.settings.n_parallel).range(1..=16));
            });
            if is_gpu {
                ui.label(
                    "GPU mode: runtime receives only selected GPU index and applies its own full-offload defaults.",
                );
            } else {
                ui.horizontal(|ui| {
                    ui.label("n_threads:");
                    ui.add(egui::DragValue::new(&mut self.settings.n_threads).range(0..=256));
                    ui.label("n_threads_batch:");
                    ui.add(
                        egui::DragValue::new(&mut self.settings.n_threads_batch).range(0..=256),
                    );
                });
                ui.label("CPU mode: keep thread values at 0 for automatic defaults.");
            }
            if secondary_button(ui, "Reset runtime defaults").clicked() {
                let defaults = AppSettings::default();
                self.settings.n_ctx = defaults.n_ctx;
                self.settings.n_batch = defaults.n_batch;
                self.settings.n_ubatch = defaults.n_ubatch;
                self.settings.n_parallel = defaults.n_parallel;
                self.settings.n_threads = defaults.n_threads;
                self.settings.n_threads_batch = defaults.n_threads_batch;
                self.queue_save();
            }
        });
    }

    fn safe_dialog_call<T, F>(&mut self, action: &str, call: F) -> Option<T>
    where
        F: FnOnce() -> Option<T>,
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)) {
            Ok(result) => result,
            Err(_) => {
                let mut details =
                    format!("{action} failed because the native file dialog backend crashed.");
                if cfg!(target_os = "linux") {
                    details.push_str(
                        "\n\nLinux tip: ensure a desktop session is running and install xdg-desktop-portal plus zenity.",
                    );
                }
                self.open_modal("File dialog failed", details, true);
                self.push_status("File dialog backend crashed; app stayed open.");
                None
            }
        }
    }

    fn do_pick_chat_model(&mut self) {
        if let Some(path) = self.safe_dialog_call("Select chat model file", || {
            FileDialog::new()
                .set_title("Select chat model file")
                .add_filter("Model", &["gguf"])
                .pick_file()
        }) {
            self.settings.chat_model = path.display().to_string();
            self.queue_save();
        }
    }

    fn do_start_chat(&mut self) {
        if self.is_chatting {
            return;
        }
        if self.anonymise_running {
            self.push_status("Chat blocked: anonymisation is currently running.");
            return;
        }
        if !self.ensure_runtime_ready("Chat") {
            return;
        }
        if self.settings.chat_model.trim().is_empty() {
            self.open_modal("Missing model", "Select chat model first.", true);
            self.push_status("Select chat model first.");
            return;
        }
        let user_prompt = self.chat_input_text.trim().to_string();
        if user_prompt.is_empty() {
            self.open_modal("Missing prompt", "Chat prompt is empty.", true);
            self.push_status("Chat prompt is empty.");
            return;
        }
        self.ensure_chat_source_selected();
        let Some(source_path) = self.chat_source_path.clone() else {
            self.open_modal(
                "Missing context file",
                "Select a transcript file in Chat source dropdown first.",
                true,
            );
            self.push_status("Chat blocked: no chat source selected.");
            return;
        };
        let context = match fs::read_to_string(&source_path) {
            Ok(text) => text,
            Err(err) => {
                self.open_modal(
                    "Failed to read context file",
                    format!("{}: {err}", source_path.display()),
                    true,
                );
                self.push_status(&format!(
                    "Chat blocked: failed to read context file '{}'",
                    source_path.display()
                ));
                return;
            }
        };

        let mut settings = self.settings.clone();
        settings.chat_prompt = format!(
            "You are a transcript Q&A assistant.\n\
             Use only the provided transcript markdown context.\n\
             If the answer is not present in context, say so clearly.\n\n\
             Transcript markdown:\n\
             {context}\n\n\
             User request:\n\
             {user_prompt}"
        );
        settings.chat_context_file.clear();

        let tx = self.tx.clone();
        self.is_chatting = true;
        if let Ok(mut state) = self.runtime_state.lock() {
            state.running_jobs += 1;
        }
        self.push_status("Running chat...");
        std::thread::spawn(move || match run_chat(settings) {
            Ok(answer) => {
                let _ = tx.send(UiMessage::ChatDone {
                    source_path,
                    user_prompt,
                    answer,
                });
            }
            Err(e) => {
                let _ = tx.send(UiMessage::ChatFailed(e.to_string()));
            }
        });
    }

    fn anonymise_selected_in_order(&self) -> Vec<AnonymiseEntityKind> {
        ANONYMISE_ENTITY_ORDER
            .iter()
            .copied()
            .filter(|kind| self.anonymise_selected_kinds.contains(kind))
            .collect()
    }

    fn ensure_editing_target_loaded(&mut self) -> Result<String> {
        let output_path = if let Some(path) = self.transcript_output_path.clone() {
            path
        } else {
            self.selected_output_path().ok_or_else(|| {
                anyhow!("select or load an output transcript before editing operations")
            })?
        };

        let reload_needed = self
            .transcript_output_path
            .as_ref()
            .map(|loaded| loaded != &output_path)
            .unwrap_or(true)
            || self.transcription_text.trim().is_empty();
        if reload_needed {
            self.open_output_transcript(&output_path)?;
        }

        if self.edited_text.trim().is_empty() {
            self.edited_text = self.transcription_text.clone();
        }
        self.edited_cues = parse_transcript(&self.edited_text).cues;

        if !self.editing_enabled {
            self.editing_enabled = true;
            self.settings.split_view = true;
            self.queue_save();
            self.push_status("Editing enabled.");
        }
        self.edit_pause_until = None;

        if self.edited_text.trim().is_empty() {
            bail!("edited transcript is empty");
        }
        Ok(self.edited_text.clone())
    }

    fn do_start_anonymise(&mut self) {
        if self.anonymise_running {
            return;
        }
        if self.is_chatting {
            self.push_status("Anonymise blocked: chat is currently running.");
            return;
        }
        if !self.ensure_runtime_ready("Anonymise") {
            return;
        }
        if self.settings.chat_model.trim().is_empty() {
            self.open_modal(
                "Missing chat model",
                "Anonymise (beta) requires a chat model in Settings -> Chat settings.",
                true,
            );
            self.push_status("Anonymise blocked: set a chat model first.");
            return;
        }
        let selected_kinds = self.anonymise_selected_in_order();
        if selected_kinds.is_empty() {
            self.open_modal(
                "Nothing selected",
                "Select at least one entity type to anonymise.",
                true,
            );
            self.push_status("Anonymise blocked: select at least one entity type.");
            return;
        }
        let source_text = match self.ensure_editing_target_loaded() {
            Ok(text) => text,
            Err(err) => {
                self.open_modal("Anonymise blocked", err.to_string(), true);
                self.push_status(&format!("Anonymise blocked: {err}"));
                return;
            }
        };

        let mut settings = self.settings.clone();

        let tx = self.tx.clone();
        self.anonymise_running = true;
        self.anonymise_last_response.clear();
        if let Ok(mut state) = self.runtime_state.lock() {
            state.running_jobs += 1;
        }
        self.push_status("Running Anonymise (beta)...");
        let prompt = build_anonymise_prompt(&source_text, &selected_kinds);
        settings.chat_prompt = prompt;
        settings.chat_context_file.clear();
        std::thread::spawn(move || match run_chat(settings) {
            Ok(answer) => {
                let _ = tx.send(UiMessage::AnonymiseDone {
                    answer,
                    source_text,
                });
            }
            Err(e) => {
                let _ = tx.send(UiMessage::AnonymiseFailed(e.to_string()));
            }
        });
    }

    fn apply_anonymise_response(
        &mut self,
        raw_response: &str,
        source_text: &str,
    ) -> Result<AnonymiseRunResult> {
        let matches = parse_anonymise_matches(raw_response)?;
        let candidates = build_replacement_candidates(&matches);
        let base_text = if source_text.trim().is_empty() {
            if self.edited_text.trim().is_empty() {
                self.transcription_text.clone()
            } else {
                self.edited_text.clone()
            }
        } else {
            source_text.to_string()
        };
        let (rewritten, exact_replacements, fuzzy_replacements) =
            apply_anonymise_to_text(&base_text, &matches);
        if exact_replacements + fuzzy_replacements == 0 && !candidates.is_empty() {
            let sample = candidates
                .iter()
                .take(3)
                .map(|entry| format!("{}: {}", entry.kind.key(), entry.value))
                .collect::<Vec<_>>()
                .join(" | ");
            self.push_log_only(&format!(
                "Anonymise extracted values but none matched transcript text. Sample: {sample}"
            ));
        }
        self.edited_text = rewritten;
        self.edited_cues = parse_transcript(&self.edited_text).cues;
        self.edit_pause_until = None;
        let output_path = self.save_edited_output()?;
        Ok(AnonymiseRunResult {
            total_matches: matches.len(),
            usable_matches: candidates.len(),
            exact_replacements,
            fuzzy_replacements,
            output_path,
        })
    }

    fn open_speaker_rename_window(&mut self) {
        if let Err(err) = self.ensure_editing_target_loaded() {
            self.open_modal("Rename speakers blocked", err.to_string(), true);
            self.push_status(&format!("Rename speakers blocked: {err}"));
            return;
        }
        self.refresh_speaker_rename_entries();
        self.show_speaker_rename_window = true;
    }

    fn refresh_speaker_rename_entries(&mut self) {
        // Keep slot detection anchored to original transcript tags so rename remains usable
        // after edited text no longer contains SPEAKER_XX tokens.
        let original_text = self
            .runtime_state
            .lock()
            .ok()
            .and_then(|state| state.original_output_path.clone())
            .and_then(|path| fs::read_to_string(path).ok());
        let source = if let Some(text) = original_text.as_deref() {
            text
        } else if !self.transcription_text.trim().is_empty() {
            self.transcription_text.as_str()
        } else {
            self.edited_text.as_str()
        };
        let detected = detect_speaker_slots(source);
        let previous = self
            .speaker_rename_entries
            .iter()
            .map(|entry| (entry.speaker_tag.clone(), entry.replacement.clone()))
            .collect::<HashMap<_, _>>();
        self.speaker_rename_entries = detected
            .into_iter()
            .map(|speaker_tag| SpeakerRenameEntry {
                replacement: previous.get(&speaker_tag).cloned().unwrap_or_default(),
                speaker_tag,
            })
            .collect();
    }

    fn apply_speaker_renames(&mut self) -> Result<usize> {
        let _ = self.ensure_editing_target_loaded()?;
        if self.speaker_rename_entries.is_empty() {
            return Ok(0);
        }
        let mut updated = self.edited_text.clone();
        let mut replaced_total = 0usize;
        for entry in &self.speaker_rename_entries {
            let replacement = entry.replacement.trim();
            if replacement.is_empty() || replacement.eq_ignore_ascii_case(&entry.speaker_tag) {
                continue;
            }
            let pattern = format!(r"(?i)\b{}\b", regex::escape(&entry.speaker_tag));
            let re = Regex::new(&pattern).context("invalid speaker rename pattern")?;
            let count = re.find_iter(&updated).count();
            if count == 0 {
                continue;
            }
            updated = re.replace_all(&updated, replacement).to_string();
            replaced_total += count;
        }
        if replaced_total > 0 {
            self.edited_text = updated;
            self.edited_cues = parse_transcript(&self.edited_text).cues;
            self.edit_pause_until = None;
            let saved = self.save_edited_output()?;
            self.push_status(&format!(
                "Speaker rename applied: {} replacements. Saved: {}",
                replaced_total,
                saved.display()
            ));
        }
        Ok(replaced_total)
    }

    fn do_download_whisper(&mut self) {
        let spec = WHISPER_MODELS
            .get(self.whisper_models)
            .copied()
            .unwrap_or(WHISPER_MODELS[0]);
        let dest = whisper_model_dest_path(&self.paths, spec.file_name);
        start_model_download(
            self.tx.clone(),
            self.runtime_state.clone(),
            "Whisper model".to_string(),
            spec.label.to_string(),
            spec.url.to_string(),
            dest,
        );
        self.push_status(&format!("Downloading {}", spec.label));
    }

    fn do_download_diarization(&mut self) {
        let dir = if self.settings.diarization_models_dir.trim().is_empty() {
            let default = default_diarization_models_dir(&self.paths);
            self.settings.diarization_models_dir = default.display().to_string();
            default
        } else {
            PathBuf::from(self.settings.diarization_models_dir.trim())
        };
        start_diarization_download(self.tx.clone(), self.runtime_state.clone(), dir);
    }

    fn do_add_media_files(&mut self) {
        if let Some(paths) = self.safe_dialog_call("Select media files", || {
            FileDialog::new()
                .set_title("Select media files")
                .add_filter("Supported media", supported_media_extensions())
                .pick_files()
        }) {
            let mut found_existing_outputs = false;
            if let Ok(mut state) = self.runtime_state.lock() {
                for path in paths {
                    if !state.media_entries.iter().any(|e| e.path == path) {
                        state.media_entries.push(MediaEntry {
                            path: path.clone(),
                            selected: true,
                        });
                    }
                    let output_paths = existing_output_paths_for_media(&path);
                    if !output_paths.is_empty() {
                        found_existing_outputs = true;
                    }
                    for output_path in output_paths {
                        ensure_output_entry(&mut state.output_entries, output_path, false);
                    }
                }
            }
            if found_existing_outputs {
                self.push_status(
                    "Added media files. Existing .md/.edited.md outputs were auto-added.",
                );
            } else {
                self.push_status("Added media files.");
            }
        }
    }

    fn do_add_output_files(&mut self) {
        if let Some(paths) = self.safe_dialog_call("Select output/result files", || {
            FileDialog::new()
                .set_title("Select output/result files")
                .pick_files()
        }) {
            if let Ok(mut state) = self.runtime_state.lock() {
                for path in paths {
                    ensure_output_entry(&mut state.output_entries, path.clone(), false);
                    if is_edited_output_path(&path) {
                        let original = original_output_path_for_any(&path);
                        if original.exists() {
                            ensure_output_entry(&mut state.output_entries, original, false);
                        }
                    } else {
                        let edited = edited_file_path(&path);
                        if edited.exists() {
                            ensure_output_entry(&mut state.output_entries, edited, false);
                        }
                    }
                }
            }
            self.push_status("Added output files.");
        }
    }

    fn clear_loaded_transcript(&mut self) {
        self.editing_enabled = false;
        self.edit_pause_until = None;
        self.transcript_output_path = None;
        self.transcription_text.clear();
        self.edited_text.clear();
        self.edited_cues.clear();
        self.speaker_rename_entries.clear();
        self.show_speaker_rename_window = false;
        self.transcript_state = TranscriptState::default();
        if let Ok(mut state) = self.runtime_state.lock() {
            state.original_output_path = None;
            state.edited_output_path = None;
        }
    }

    fn selected_output_path(&self) -> Option<PathBuf> {
        let row = self.selected_output_row?;
        self.runtime_state.lock().ok().and_then(|state| {
            state
                .output_entries
                .get(row)
                .map(|entry| entry.path.clone())
        })
    }

    fn open_output_transcript(&mut self, output_path: &Path) -> Result<()> {
        if !output_path.exists() {
            bail!("output file does not exist: '{}'", output_path.display());
        }
        let selected_path = output_path.to_path_buf();
        let original_path = original_output_path_for_any(&selected_path);
        let selected_is_edited = is_edited_output_path(&selected_path);
        let output_text_source = if selected_is_edited {
            selected_path.clone()
        } else {
            original_path.clone()
        };
        let output_text = fs::read_to_string(&output_text_source).with_context(|| {
            format!(
                "failed to read output file '{}'",
                output_text_source.display()
            )
        })?;
        let edited_path = edited_file_path(&original_path);
        let edited_text = if edited_path.exists() {
            fs::read_to_string(&edited_path).unwrap_or_else(|_| output_text.clone())
        } else {
            output_text.clone()
        };
        let displayed_transcript_text = if selected_is_edited {
            edited_text.clone()
        } else {
            output_text.clone()
        };

        self.transcript_output_path = Some(selected_path.clone());
        self.transcription_text = displayed_transcript_text.clone();
        self.edited_text = edited_text.clone();
        self.transcript_state = parse_transcript(&self.transcription_text);
        self.edited_cues = parse_transcript(&edited_text).cues;
        self.refresh_speaker_rename_entries();
        if let Some(audio_guess) = guess_audio_path_for_output(&original_path) {
            self.active_audio_path = Some(audio_guess.clone());
            self.settings.audio_file = audio_guess.display().to_string();
        }

        if let Ok(mut state) = self.runtime_state.lock() {
            state.original_output_path = Some(original_path.clone());
            state.edited_output_path = Some(edited_path.clone());
            ensure_output_entry(&mut state.output_entries, original_path, false);
            if edited_path.exists() {
                ensure_output_entry(&mut state.output_entries, edited_path, false);
            }
            if let Some(audio_guess) = self.active_audio_path.clone() {
                state.active_audio_path = Some(audio_guess);
            }
        }

        if self.chat_source_path.is_none() {
            let _ = self.load_chat_source_file(selected_path);
        }

        Ok(())
    }

    fn enqueue_job(&mut self, files: Vec<PathBuf>) {
        if !self.ensure_runtime_ready("Transcription") {
            return;
        }
        if !self.ensure_transcription_setup_ready() {
            return;
        }
        if files.is_empty() {
            self.open_modal("No files", "No selected media files.", true);
            self.push_status("No selected media files.");
            return;
        }

        let mut enqueue_id = 0u64;
        if let Ok(mut state) = self.runtime_state.lock() {
            state.next_job_id += 1;
            enqueue_id = state.next_job_id;
            state.job_queue.push_back(QueuedJob {
                id: enqueue_id,
                settings: self.settings.clone(),
                files,
            });
        }
        self.is_transcribing = true;
        self.push_status(&format!("Queued job #{enqueue_id}."));
        start_queue_worker_if_needed(self.runtime_state.clone(), self.tx.clone());
    }

    fn chat_source_candidates(&self) -> Vec<PathBuf> {
        let mut out = Vec::<PathBuf>::new();
        let mut seen = HashSet::<String>::new();
        if let Ok(state) = self.runtime_state.lock() {
            for entry in &state.output_entries {
                if !entry.path.exists() {
                    continue;
                }
                let key = entry.path.to_string_lossy().to_ascii_lowercase();
                if seen.insert(key) {
                    out.push(entry.path.clone());
                }
            }
        }
        if let Some(path) = self.transcript_output_path.clone() {
            let key = path.to_string_lossy().to_ascii_lowercase();
            if path.exists() && seen.insert(key) {
                out.push(path);
            }
        }
        out
    }

    fn load_chat_source_file(&mut self, source_path: PathBuf) -> Result<()> {
        if !source_path.exists() {
            bail!(
                "chat source file does not exist: '{}'",
                source_path.display()
            );
        }
        let context = fs::read_to_string(&source_path)
            .with_context(|| format!("failed to read '{}'", source_path.display()))?;
        let log_path = chat_log_path_for_output(&source_path);
        let history = if log_path.exists() {
            fs::read_to_string(&log_path).unwrap_or_default()
        } else {
            String::new()
        };
        self.chat_source_path = Some(source_path);
        self.chat_context_text = context;
        self.chat_history_text = history;
        Ok(())
    }

    fn ensure_chat_source_selected(&mut self) {
        if let Some(current) = self.chat_source_path.clone() {
            if current.exists() {
                return;
            }
        }
        let candidates = self.chat_source_candidates();
        let preferred = self
            .transcript_output_path
            .clone()
            .filter(|p| p.exists())
            .or_else(|| candidates.first().cloned());
        if let Some(path) = preferred {
            let _ = self.load_chat_source_file(path);
        }
    }

    fn append_chat_log_entry(
        &mut self,
        source_path: &Path,
        user_prompt: &str,
        answer: &str,
    ) -> Result<PathBuf> {
        let log_path = chat_log_path_for_output(source_path);
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create '{}'", parent.display()))?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("failed to open '{}'", log_path.display()))?;
        let stamp = log_timestamp_hms();
        let block = format!(
            "[{stamp}] You: {user_prompt}\n\n[{stamp}] Assistant: {answer}\n\n----------------\n\n"
        );
        file.write_all(block.as_bytes())
            .with_context(|| format!("failed to append '{}'", log_path.display()))?;
        Ok(log_path)
    }

    fn clear_chat_log_for_source(&mut self, source_path: &Path) -> Result<PathBuf> {
        let log_path = chat_log_path_for_output(source_path);
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create '{}'", parent.display()))?;
        }
        fs::write(&log_path, "")
            .with_context(|| format!("failed to clear '{}'", log_path.display()))?;
        if self
            .chat_source_path
            .as_ref()
            .map(|p| p == source_path)
            .unwrap_or(false)
        {
            self.chat_history_text.clear();
        }
        Ok(log_path)
    }

    fn ui_playback(&mut self, ui: &mut egui::Ui) {
        let current = playback_current_time(&self.playback);
        let total = playback_total_time(&self.playback);
        let audio_path = self.resolve_active_audio_path().filter(|p| p.exists());
        let has_audio = audio_path.is_some();

        ui.separator();
        ui.label("Playback");
        ui.label(format!(
            "{} / {}{}",
            secs_to_hms(current),
            secs_to_hms(total),
            if playback_is_ended(&self.playback) {
                " (ended)"
            } else {
                ""
            }
        ));
        if !has_audio {
            ui.label(
                "No active audio selected. Select media/output from a completed transcript to enable playback.",
            );
        }

        ui.horizontal(|ui| {
            if accent_button(ui, "Play/Pause (Ctrl+Space)").clicked() {
                if !has_audio {
                    self.push_status("No active audio selected.");
                    return;
                }
                if !self.ensure_runtime_ready("Playback") {
                    return;
                }
                let path = audio_path.as_ref().expect("audio path present").clone();
                if playback_has_audio(&self.playback) && !playback_is_ended(&self.playback) {
                    match playback_toggle_pause(&mut self.playback, &path, 0.0) {
                        Ok(now) => self.push_status(&format!("Playback {:.2}s", now.max(0.0))),
                        Err(err) => self.push_status(&format!("Playback failed: {err}")),
                    };
                } else {
                    let offset = self.settings.playback_toggle_offset_sec as f64;
                    self.request_playback_start(path, offset, "play");
                }
            }
            if accent_button(ui, "Restart").clicked() {
                if !has_audio {
                    self.push_status("No active audio selected.");
                    return;
                }
                if !self.ensure_runtime_ready("Playback") {
                    return;
                }
                self.request_playback_start(
                    audio_path.as_ref().expect("audio path present").clone(),
                    0.0,
                    "restart",
                );
            }
            if accent_button(ui, "<").clicked() {
                if !has_audio {
                    self.push_status("No active audio selected.");
                    return;
                }
                if !self.ensure_runtime_ready("Playback") {
                    return;
                }
                let step = self.settings.seek_step_sec as f64;
                let growth = self.settings.seek_step_growth_sec as f64;
                if playback_has_audio(&self.playback) {
                    match playback_seek_relative(&mut self.playback, -step, growth) {
                        Ok(now) => self.push_status(&format!("Playback {:.2}s", now.max(0.0))),
                        Err(err) => self.push_status(&format!("Seek failed: {err}")),
                    };
                } else {
                    let now = playback_current_time(&self.playback);
                    self.request_playback_start(
                        audio_path.as_ref().expect("audio path present").clone(),
                        (now - step).max(0.0),
                        "seek back",
                    );
                }
            }
            if accent_button(ui, ">").clicked() {
                if !has_audio {
                    self.push_status("No active audio selected.");
                    return;
                }
                if !self.ensure_runtime_ready("Playback") {
                    return;
                }
                let step = self.settings.seek_step_sec as f64;
                let growth = self.settings.seek_step_growth_sec as f64;
                if playback_has_audio(&self.playback) {
                    match playback_seek_relative(&mut self.playback, step, growth) {
                        Ok(now) => self.push_status(&format!("Playback {:.2}s", now.max(0.0))),
                        Err(err) => self.push_status(&format!("Seek failed: {err}")),
                    };
                } else {
                    let now = playback_current_time(&self.playback);
                    self.request_playback_start(
                        audio_path.as_ref().expect("audio path present").clone(),
                        (now + step).max(0.0),
                        "seek forward",
                    );
                }
            }
            ui.label("Speed x");
            let speed_changed = ui
                .add(egui::TextEdit::singleline(&mut self.playback_speed_text).desired_width(120.0))
                .changed();
            if speed_changed {
                let speed = parse_f32_or(1.0, &self.playback_speed_text).clamp(0.1, 4.0);
                self.playback.speed = speed as f64;
                playback_apply_speed(&mut self.playback);
                self.push_status(&format!("Speed {:.2}x", speed));
            }
            if self.playback_decode_in_progress {
                ui.spinner();
                ui.label("Decoding...");
            }
        });
    }

    fn mode_note_text_parity(&self) -> &'static str {
        if self.settings.mode.eq_ignore_ascii_case("transcript") {
            "Transcript mode includes speaker diarization."
        } else if self.settings.mode.eq_ignore_ascii_case("speech") {
            "Speech mode generates Markdown transcript without diarization."
        } else {
            "Subtitle mode generates SRT output without diarization."
        }
    }

    fn banner_text_parity(&self) -> String {
        if let Ok(state) = self.runtime_state.lock() {
            let mut text = format!(
                "{} running | {} queued | {} complete",
                state.running_jobs,
                state.job_queue.len(),
                state.completed_jobs
            );
            if let Some(active) = &state.active_job {
                text.push_str(&format!(
                    " | job #{} file {}/{}",
                    active.id, active.done_files, active.total_files
                ));
            }
            return text;
        }
        "0 running | 0 queued | 0 complete".to_string()
    }

    fn ui_header_parity(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Mode:");
            let mut mode = self.settings.mode.clone();
            egui::ComboBox::from_id_salt("mode_combo_parity")
                .selected_text(mode.clone())
                .width(116.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut mode, "transcript".to_string(), "transcript");
                    ui.selectable_value(&mut mode, "speech".to_string(), "speech");
                    ui.selectable_value(&mut mode, "subtitle".to_string(), "subtitle");
                });
            if mode != self.settings.mode {
                self.settings.mode = mode;
                self.queue_save();
            }

            ui.label("Speakers:");
            ui.add_enabled_ui(
                self.settings.mode.eq_ignore_ascii_case("transcript"),
                |ui| {
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut self.settings.speaker_count)
                                .desired_width(54.0),
                        )
                        .changed()
                    {
                        self.queue_save();
                    }
                },
            );
            ui.add_space(8.0);
            ui.label(self.mode_note_text_parity());
            ui.add_space(6.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if tab_button(ui, "Chat", self.tab == 1).clicked() {
                    self.tab = 1;
                }
                if tab_button(ui, "Transcription", self.tab == 0).clicked() {
                    self.tab = 0;
                }
            });
        });
        ui.separator();
    }

    fn ui_status_bar_parity(&mut self, ui: &mut egui::Ui) {
        let mut open_logs = false;
        let queue_banner = self.banner_text_parity();
        let (download_status, active_stage) = self
            .runtime_state
            .lock()
            .ok()
            .map(|state| (state.download_status.clone(), state.active_stage.clone()))
            .unwrap_or((None, String::new()));
        let status_text = {
            let current = self.status.clone();
            if let Some(active_download) = download_status.as_ref() {
                if current.trim() == active_download.trim() {
                    String::new()
                } else {
                    current
                }
            } else {
                current
            }
        };
        let row = ui.horizontal_wrapped(|ui| {
            if !status_text.trim().is_empty() {
                if ui
                    .add(egui::Label::new(status_text).sense(egui::Sense::click()))
                    .clicked()
                {
                    open_logs = true;
                }
                ui.separator();
            }
            if !active_stage.trim().is_empty() {
                if ui
                    .add(
                        egui::Label::new(
                            egui::RichText::new(format!("Stage: {active_stage}")).strong(),
                        )
                        .sense(egui::Sense::click()),
                    )
                    .clicked()
                {
                    open_logs = true;
                }
                ui.separator();
            }
            if let Some(active_download) = download_status.as_ref() {
                let progress = parse_download_fraction(active_download);
                let mut banner = active_download.clone();
                if let Some(fraction) = progress {
                    banner.push_str(&format!(" ({:.0}%)", fraction * 100.0));
                }
                if ui
                    .add(egui::Label::new(banner).sense(egui::Sense::click()))
                    .clicked()
                {
                    open_logs = true;
                }
                if let Some(fraction) = progress {
                    ui.add(egui::ProgressBar::new(fraction).desired_width(120.0));
                }
                ui.separator();
            }
            if ui
                .add(egui::Label::new(queue_banner).sense(egui::Sense::click()))
                .clicked()
            {
                open_logs = true;
            }
        });
        if open_logs || row.response.clicked() {
            self.show_logs_window = true;
        }
    }

    fn ui_logs_window_parity(&mut self, ctx: &egui::Context) {
        if !self.show_logs_window {
            return;
        }
        let mut still_open = true;
        let viewport_id = egui::ViewportId::from_hash_of("logs-window");
        let builder = egui::ViewportBuilder::default()
            .with_title("Transcribe Offline - Logs")
            .with_inner_size([980.0, 460.0])
            .with_resizable(true);
        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            if ctx.input(|i| i.viewport().close_requested()) {
                still_open = false;
            }
            egui::CentralPanel::default().show(ctx, |ui| {
                let runtime_snapshot = self.runtime_state.lock().ok().map(|state| {
                    (
                        state.running_jobs,
                        state.job_queue.len(),
                        state.completed_jobs,
                        state.active_stage.clone(),
                    )
                });
                if let Some((running, queued, completed, stage)) = runtime_snapshot {
                    ui.label(format!(
                        "Jobs: running={} queued={} complete={} stage={}",
                        running,
                        queued,
                        completed,
                        if stage.trim().is_empty() {
                            "<idle>"
                        } else {
                            stage.as_str()
                        }
                    ));
                }
                ui.separator();
                let bottom_controls_h = 38.0;
                let log_h = (ui.available_height() - bottom_controls_h).max(120.0);
                egui::ScrollArea::vertical()
                    .id_salt("logs_window_scroll")
                    .max_height(log_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for line in &self.status_log {
                            ui.label(egui::RichText::new(line).monospace());
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if secondary_button(ui, "Clear logs").clicked() {
                        self.status_log.clear();
                        self.push_log_only("Log cleared.");
                    }
                    if accent_button(ui, "Close logs window").clicked() {
                        still_open = false;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });
        self.show_logs_window = still_open;
    }

    fn resolve_active_audio_path(&self) -> Option<PathBuf> {
        if let Some(path) = &self.active_audio_path {
            return Some(path.clone());
        }
        if let Some(path) = self
            .transcript_output_path
            .as_ref()
            .and_then(|out| guess_audio_path_for_output(out))
        {
            return Some(path);
        }
        self.runtime_state
            .lock()
            .ok()
            .and_then(|s| s.active_audio_path.clone())
    }

    fn set_click_sync_target(&mut self, target_sec: f64) {
        self.playback_click_sync_target = Some((
            target_sec.max(0.0),
            Instant::now() + Duration::from_secs_f64(PLAYBACK_CLICK_SYNC_HOLD_SEC),
        ));
    }

    fn playback_time_for_transcript_sync(&mut self) -> f64 {
        let actual = playback_current_time(&self.playback);
        if let Some((target, deadline)) = self.playback_click_sync_target {
            let now = Instant::now();
            if now > deadline || actual + 0.20 >= target {
                self.playback_click_sync_target = None;
                actual
            } else {
                target
            }
        } else {
            actual
        }
    }

    fn follow_paused_for_user_navigation(&self) -> bool {
        let now = Instant::now();
        let edit_pause = self
            .edit_pause_until
            .map(|deadline| now < deadline)
            .unwrap_or(false);
        let scroll_pause = self
            .playback_follow_pause_until
            .map(|deadline| now < deadline)
            .unwrap_or(false);
        edit_pause || scroll_pause
    }

    fn playback_follow_active(&self) -> bool {
        playback_is_playing(&self.playback) || self.playback_click_sync_target.is_some()
    }

    fn pause_follow_after_manual_scroll(&mut self) {
        self.playback_follow_pause_until =
            Some(Instant::now() + Duration::from_secs_f64(PLAYBACK_SCROLL_FOLLOW_PAUSE_SEC));
    }

    fn user_scrolled_in_rect(&self, ctx: &egui::Context, inner_rect: egui::Rect) -> bool {
        ctx.input(|i| {
            let pointer_over = i
                .pointer
                .hover_pos()
                .map(|pos| inner_rect.contains(pos))
                .unwrap_or(false);
            if !pointer_over {
                return false;
            }
            let delta = i.raw_scroll_delta + i.smooth_scroll_delta;
            delta.x.abs() > f32::EPSILON || delta.y.abs() > f32::EPSILON
        })
    }

    fn mark_follow_pause_on_scroll_area(&mut self, ctx: &egui::Context, inner_rect: egui::Rect) {
        let user_scrolled_here = self.user_scrolled_in_rect(ctx, inner_rect);
        if user_scrolled_here {
            self.pause_follow_after_manual_scroll();
        }
    }

    fn maybe_predecode_active_audio(&mut self) {
        if self.playback_decode_in_progress {
            return;
        }
        let Some(path) = self.resolve_active_audio_path() else {
            return;
        };
        if !path.exists() {
            return;
        }
        let path = normalize_playback_path(&path);
        if playback_has_loaded_path(&self.playback, &path) {
            return;
        }
        if playback_ensure_stream(&mut self.playback).is_err() {
            return;
        }
        let sample_rate = self.playback.output_sample_rate.max(16_000);
        self.playback_decode_in_progress = true;
        self.playback_decode_autoplay = false;
        self.playback_pending_start = None;
        self.push_log_only(&format!(
            "Preloading audio to RAM for instant playback: {}",
            path.display()
        ));
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let result = decode_audio_to_stereo_f32(&path, sample_rate);
            match result {
                Ok(data) => {
                    let _ = tx.send(UiMessage::PlaybackDecoded {
                        audio_path: path,
                        start_sec: 0.0,
                        data,
                    });
                }
                Err(err) => {
                    let _ = tx.send(UiMessage::PlaybackDecodeFailed {
                        audio_path: path,
                        error: err.to_string(),
                    });
                }
            }
        });
    }

    fn request_playback_start(&mut self, path: PathBuf, start_sec: f64, reason: &str) {
        let path = normalize_playback_path(&path);
        let start_sec = start_sec.max(0.0);
        if playback_has_loaded_path(&self.playback, &path) {
            match playback_start_from(&mut self.playback, &path, start_sec) {
                Ok(_) => self.push_status(&format!("Playback {:.2}s", start_sec)),
                Err(err) => self.push_status(&format!("Playback start failed: {err}")),
            }
            return;
        }

        if self.playback_decode_in_progress {
            self.playback_pending_start = Some((path, start_sec));
            self.push_status("Playback decode in progress. Queued latest seek/start request.");
            return;
        }

        if let Err(err) = playback_ensure_stream(&mut self.playback) {
            self.push_status(&format!("Playback stream init failed: {err}"));
            return;
        }
        let sample_rate = self.playback.output_sample_rate.max(16_000);
        self.playback_decode_in_progress = true;
        self.playback_decode_autoplay = true;
        self.playback_pending_start = None;
        self.push_status(&format!(
            "Decoding audio for playback ({reason}) at {:.2}s...",
            start_sec
        ));
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let result = decode_audio_to_stereo_f32(&path, sample_rate);
            match result {
                Ok(data) => {
                    let _ = tx.send(UiMessage::PlaybackDecoded {
                        audio_path: path,
                        start_sec,
                        data,
                    });
                }
                Err(err) => {
                    let _ = tx.send(UiMessage::PlaybackDecodeFailed {
                        audio_path: path,
                        error: err.to_string(),
                    });
                }
            }
        });
    }

    fn seek_to_line(&mut self, line_idx: usize) {
        let target = cue_start_for_line(&self.transcript_state.cues, line_idx);
        let Some(target) = target else {
            return;
        };
        self.playback_follow_pause_until = None;
        self.edit_pause_until = None;
        self.set_click_sync_target(target);
        if !self.ensure_runtime_ready("Playback") {
            return;
        }
        let Some(path) = self.resolve_active_audio_path() else {
            self.push_status("Playback seek blocked: no active audio path for this transcript.");
            return;
        };
        let offset = self.settings.playback_toggle_offset_sec as f64;
        let seek_to = (target + offset).max(0.0);
        self.request_playback_start(path, seek_to, "line click");
        self.push_status(&format!(
            "Playback line seek: {} at {:.2}s",
            line_idx + 1,
            seek_to
        ));
    }

    fn seek_to_edited_line(&mut self, line_idx: usize) {
        let target = cue_start_for_line(&self.edited_cues, line_idx)
            .or_else(|| cue_start_for_line(&self.transcript_state.cues, line_idx));
        let Some(target) = target else {
            return;
        };
        self.playback_follow_pause_until = None;
        self.edit_pause_until = None;
        self.set_click_sync_target(target);
        if !self.ensure_runtime_ready("Playback") {
            return;
        }
        let Some(path) = self.resolve_active_audio_path() else {
            self.push_status("Playback seek blocked: no active audio path for this transcript.");
            return;
        };
        let offset = self.settings.playback_toggle_offset_sec as f64;
        let seek_to = (target + offset).max(0.0);
        self.request_playback_start(path, seek_to, "edited line double-click");
        self.push_status(&format!(
            "Playback edited-line seek: {} at {:.2}s",
            line_idx + 1,
            seek_to
        ));
    }

    fn toggle_editing(&mut self) {
        if !self.editing_enabled {
            let Some(path) = self
                .selected_output_path()
                .or_else(|| self.transcript_output_path.clone())
            else {
                self.open_modal(
                    "No transcript selected",
                    "Select or load a transcript in Output files before editing.",
                    true,
                );
                self.push_status("Edit blocked: select an output transcript first.");
                return;
            };
            let needs_reload = self
                .transcript_output_path
                .as_ref()
                .map(|loaded| loaded != &path)
                .unwrap_or(true)
                || self.transcription_text.trim().is_empty();
            if needs_reload {
                if let Err(err) = self.open_output_transcript(&path) {
                    self.open_modal("Failed to open transcript", err.to_string(), true);
                    self.push_status(&format!("Edit blocked: {err}"));
                    return;
                }
            }
            if self.edited_text.trim().is_empty() {
                self.edited_text = self.transcription_text.clone();
            }
            self.edited_cues = parse_transcript(&self.edited_text).cues;
        } else if let Err(err) = self.save_edited_output() {
            self.open_modal("Failed to save edits", err.to_string(), true);
            self.push_status(&format!("Stop editing blocked: {err}"));
            return;
        }

        self.editing_enabled = !self.editing_enabled;
        self.settings.split_view = self.editing_enabled;
        self.edit_pause_until = None;
        self.push_status(if self.editing_enabled {
            "Editing enabled."
        } else {
            "Editing stopped."
        });
        self.queue_save();
    }

    fn save_edited_output(&mut self) -> Result<PathBuf> {
        if self.edited_text.trim().is_empty() {
            bail!("edited text is empty");
        }
        let selected_path = if let Some(path) = self.transcript_output_path.clone() {
            path
        } else {
            self.runtime_state
                .lock()
                .ok()
                .and_then(|state| state.original_output_path.clone())
                .ok_or_else(|| anyhow!("no transcript output selected for editing"))?
        };
        let original_path = original_output_path_for_any(&selected_path);
        if !original_path.exists() {
            if !(is_edited_output_path(&selected_path) && selected_path.exists()) {
                bail!(
                    "selected transcript does not exist: '{}'",
                    original_path.display()
                );
            }
        }

        self.transcript_output_path = Some(selected_path.clone());
        let output_path = edited_file_path(&original_path);
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create '{}'", parent.display()))?;
        }
        fs::write(&output_path, &self.edited_text)
            .with_context(|| format!("failed to write '{}'", output_path.display()))?;
        if let Ok(mut state) = self.runtime_state.lock() {
            state.original_output_path = Some(original_path.clone());
            state.edited_output_path = Some(output_path.clone());
            ensure_output_entry(&mut state.output_entries, original_path, false);
            ensure_output_entry(&mut state.output_entries, output_path.clone(), false);
        }
        Ok(output_path)
    }

    fn apply_font_size(&mut self, ctx: &egui::Context) {
        if !self.fonts_configured {
            configure_preferred_fonts(ctx);
            self.fonts_configured = true;
        }
        ctx.options_mut(|opts| {
            opts.tessellation_options = Default::default();
        });

        let body = clamp_font_px(self.settings.ui_font_size_px);
        if (self.settings.ui_font_size_px - body).abs() > 0.001 {
            self.settings.ui_font_size_px = body;
            self.queue_save();
        }

        let ppp = ctx.pixels_per_point().max(0.01);
        let snap = |size: f32| -> f32 { ((size * ppp).round() / ppp).max(1.0) };

        let body = snap(body);
        let heading = snap(body + 4.0);
        let small = body;
        let monospace = snap((body - 1.0).max(9.0));
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::light();
        style.visuals.override_text_color = None;
        style.visuals.panel_fill = egui::Color32::from_rgb(246, 246, 247);
        style.visuals.window_fill = egui::Color32::from_rgb(255, 255, 255);
        style.visuals.faint_bg_color = egui::Color32::from_rgb(249, 249, 250);
        style.visuals.extreme_bg_color = egui::Color32::from_rgb(236, 236, 238);
        style.visuals.selection.bg_fill = egui::Color32::from_rgb(223, 227, 232);
        style.visuals.selection.stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 56, 64));
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::proportional(heading),
        );
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(body));
        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(body));
        style
            .text_styles
            .insert(egui::TextStyle::Small, egui::FontId::proportional(small));
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            egui::FontId::monospace(monospace),
        );
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(8.0, 2.0);
        style.spacing.interact_size = egui::vec2(34.0, 22.0);
        style.spacing.menu_margin = egui::Margin::same(6);
        style.spacing.combo_width = 148.0;
        style.spacing.menu_width = 260.0;
        style.spacing.text_edit_width = 168.0;

        ctx.set_style(style);
    }

    fn handle_hotkeys(&mut self, ctx: &egui::Context) {
        let Some(path) = self.resolve_active_audio_path() else {
            return;
        };
        let mut trigger_toggle = false;
        let mut seek_delta = 0.0f64;
        ctx.input(|i| {
            let ctrl_or_cmd = i.modifiers.ctrl || i.modifiers.command;
            if !ctrl_or_cmd {
                return;
            }
            if i.key_pressed(egui::Key::Space) {
                trigger_toggle = true;
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::Comma) {
                seek_delta = -(self.settings.seek_step_sec as f64);
            } else if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::Period) {
                seek_delta = self.settings.seek_step_sec as f64;
            }
        });
        if trigger_toggle {
            let offset = self.settings.playback_toggle_offset_sec as f64;
            if playback_has_audio(&self.playback) && !playback_is_ended(&self.playback) {
                let _ = playback_toggle_pause(&mut self.playback, &path, 0.0);
            } else {
                self.request_playback_start(path.clone(), offset, "hotkey play");
            }
        }
        if seek_delta.abs() > 0.0 {
            if playback_has_audio(&self.playback) {
                let _ = playback_seek_relative(
                    &mut self.playback,
                    seek_delta,
                    self.settings.seek_step_growth_sec as f64,
                );
            } else {
                let target = (playback_current_time(&self.playback) + seek_delta).max(0.0);
                self.request_playback_start(path, target, "hotkey seek");
            }
        }
    }

    fn ui_transcription_parity(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            engine_panel_frame().show(&mut cols[0], |ui| {
                let list_height = 128.0;
                ui.heading("Media files (inputs)");
                ui.set_min_height(list_height + 24.0);
                egui::ScrollArea::vertical()
                    .id_salt("media_list_parity")
                    .max_height(list_height)
                    .show(ui, |ui| {
                        let entries = self
                            .runtime_state
                            .lock()
                            .ok()
                            .map(|s| s.media_entries.clone())
                            .unwrap_or_default();
                        let mut updates = Vec::<(usize, bool)>::new();
                        let mut selected_audio: Option<PathBuf> = None;
                        if entries.is_empty() {
                            ui.allocate_space(egui::vec2(ui.available_width(), list_height - 10.0));
                        } else {
                            for (idx, entry) in entries.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    let mut selected = entry.selected;
                                    if ui.checkbox(&mut selected, "").changed() {
                                        updates.push((idx, selected));
                                    }
                                    if ui
                                        .selectable_label(
                                            self.selected_media_row == Some(idx),
                                            entry.path.display().to_string(),
                                        )
                                        .clicked()
                                    {
                                        self.selected_media_row = Some(idx);
                                        selected_audio = Some(entry.path.clone());
                                    }
                                });
                            }
                        }
                        if !updates.is_empty() {
                            if let Ok(mut state) = self.runtime_state.lock() {
                                for (idx, selected) in updates {
                                    if let Some(item) = state.media_entries.get_mut(idx) {
                                        item.selected = selected;
                                    }
                                }
                            }
                        }
                        if let Some(path) = selected_audio {
                            self.settings.audio_file = path.display().to_string();
                            self.active_audio_path = Some(path);
                            self.queue_save();
                        }
                    });
            });

            engine_panel_frame().show(&mut cols[1], |ui| {
                let list_height = 128.0;
                ui.heading("Output files (results)");
                ui.set_min_height(list_height + 24.0);
                egui::ScrollArea::vertical()
                    .id_salt("output_list_parity")
                    .max_height(list_height)
                    .show(ui, |ui| {
                        let entries = self
                            .runtime_state
                            .lock()
                            .ok()
                            .map(|s| s.output_entries.clone())
                            .unwrap_or_default();
                        let mut updates = Vec::<(usize, bool)>::new();
                        let mut output_to_open: Option<PathBuf> = None;
                        if entries.is_empty() {
                            ui.allocate_space(egui::vec2(ui.available_width(), list_height - 10.0));
                        } else {
                            for (idx, entry) in entries.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    let mut selected = entry.selected;
                                    if ui.checkbox(&mut selected, "").changed() {
                                        updates.push((idx, selected));
                                        if selected {
                                            self.selected_output_row = Some(idx);
                                            output_to_open = Some(entry.path.clone());
                                        }
                                    }
                                    if ui
                                        .selectable_label(
                                            self.selected_output_row == Some(idx),
                                            entry.path.display().to_string(),
                                        )
                                        .clicked()
                                    {
                                        self.selected_output_row = Some(idx);
                                        output_to_open = Some(entry.path.clone());
                                    }
                                });
                            }
                        }
                        if !updates.is_empty() {
                            if let Ok(mut state) = self.runtime_state.lock() {
                                for (idx, selected) in updates {
                                    if let Some(item) = state.output_entries.get_mut(idx) {
                                        item.selected = selected;
                                    }
                                }
                            }
                        }
                        if let Some(path) = output_to_open {
                            match self.open_output_transcript(&path) {
                                Ok(_) => {
                                    self.push_status(&format!(
                                        "Loaded output transcript: {}",
                                        path.display()
                                    ));
                                }
                                Err(err) => {
                                    self.open_modal(
                                        "Failed to load output transcript",
                                        err.to_string(),
                                        true,
                                    );
                                    self.push_status(&format!(
                                        "Failed to load output transcript: {err}"
                                    ));
                                }
                            }
                        }
                    });
            });
        });

        ui.columns(2, |cols| {
            cols[0].horizontal_wrapped(|ui| {
                if secondary_button(ui, "Add...").clicked() {
                    self.do_add_media_files();
                }
                if secondary_button(ui, "Remove selected").clicked() {
                    if let Ok(mut state) = self.runtime_state.lock() {
                        state.media_entries.retain(|e| !e.selected);
                    }
                }
                if secondary_button(ui, "Select all").clicked() {
                    if let Ok(mut state) = self.runtime_state.lock() {
                        for e in &mut state.media_entries {
                            e.selected = true;
                        }
                    }
                }
                if secondary_button(ui, "Select none").clicked() {
                    if let Ok(mut state) = self.runtime_state.lock() {
                        for e in &mut state.media_entries {
                            e.selected = false;
                        }
                    }
                }
                if accent_button(ui, "Run selected >").clicked() {
                    let files = self
                        .runtime_state
                        .lock()
                        .ok()
                        .map(|s| {
                            s.media_entries
                                .iter()
                                .filter(|e| e.selected)
                                .map(|e| e.path.clone())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    self.enqueue_job(files);
                }
            });

            cols[1].horizontal_wrapped(|ui| {
                if secondary_button(ui, "Add...").clicked() {
                    self.do_add_output_files();
                }
                if secondary_button(ui, "Remove selected").clicked() {
                    let current_output = self.transcript_output_path.clone();
                    let mut removed_current = false;
                    let mut any_removed = false;
                    if let Ok(mut state) = self.runtime_state.lock() {
                        if let Some(current) = current_output.as_ref() {
                            removed_current = state
                                .output_entries
                                .iter()
                                .any(|entry| entry.selected && entry.path == *current);
                        }
                        any_removed = state.output_entries.iter().any(|entry| entry.selected);
                        state.output_entries.retain(|e| !e.selected);
                        if removed_current {
                            state.original_output_path = None;
                            state.edited_output_path = None;
                        }
                    }
                    if any_removed {
                        self.selected_output_row = None;
                    }
                    if removed_current {
                        self.clear_loaded_transcript();
                        self.push_status(
                            "Removed selected output transcript. Select another output to edit.",
                        );
                    }
                }
                if secondary_button(ui, "Select all").clicked() {
                    if let Ok(mut state) = self.runtime_state.lock() {
                        for e in &mut state.output_entries {
                            e.selected = true;
                        }
                    }
                }
                if secondary_button(ui, "Select none").clicked() {
                    if let Ok(mut state) = self.runtime_state.lock() {
                        for e in &mut state.output_entries {
                            e.selected = false;
                        }
                    }
                }
            });
        });

        ui.separator();
        let transcript_panel_height = ui.available_height().max(320.0);
        engine_panel_frame().show(ui, |ui| {
            ui.set_min_height(transcript_panel_height);
            ui.horizontal(|ui| {
                ui.heading("Transcript");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let info = ui.small_button("i");
                    info.on_hover_text(ANONYMISE_TOOLTIP);
                    if self.anonymise_running {
                        ui.add_enabled(false, egui::Button::new("Anonymise (beta)"));
                    } else if accent_button(ui, "Anonymise (beta)").clicked() {
                        self.show_anonymise_window = true;
                    }
                    if secondary_button(ui, "Rename speakers").clicked() {
                        self.open_speaker_rename_window();
                    }
                    let label = if self.editing_enabled {
                        "Stop editing"
                    } else {
                        "Edit"
                    };
                    let can_start_editing = self
                        .selected_output_path()
                        .map(|path| path.exists())
                        .unwrap_or(false)
                        || self
                            .transcript_output_path
                            .as_ref()
                            .map(|path| path.exists())
                            .unwrap_or(false);
                    let edit_enabled = self.editing_enabled || can_start_editing;
                    if edit_enabled && accent_button(ui, label).clicked() {
                        self.toggle_editing();
                    } else if !edit_enabled {
                        ui.add_enabled(false, egui::Button::new(label));
                    }
                });
            });
            if !self.editing_enabled && self.transcript_output_path.is_none() {
                ui.label("Select an output transcript in the results list to enable editing.");
            }

            let transcript_height = (ui.available_height() - 130.0).max(220.0);
            let sync_time = self.playback_time_for_transcript_sync();
            let current_line = cue_index_at_time(&self.transcript_state.cues, sync_time);
            let active_range = cue_line_range_at_time(
                &self.transcript_state.cues,
                sync_time,
                self.transcription_text.lines().count(),
            );
            let active_anchor = active_range.map(|(start, _)| start);
            let edited_current_line = cue_index_at_time(&self.edited_cues, sync_time);
            let edited_active_range = cue_line_range_at_time(
                &self.edited_cues,
                sync_time,
                self.edited_text.lines().count(),
            );
            let edited_active_anchor = edited_active_range
                .map(|(start, _)| start)
                .or(edited_current_line)
                .or(active_anchor);
            let transcript_lines = self
                .transcription_text
                .lines()
                .map(|line| line.to_string())
                .collect::<Vec<_>>();
            let playback_follow_active = self.playback_follow_active();
            let follow_paused = !playback_follow_active || self.follow_paused_for_user_navigation();
            if self.editing_enabled {
                let mut original_scroll_info = None::<(f32, bool)>;
                let mut edited_scroll_info = None::<(f32, bool)>;
                let manual_edit_sync_enabled = !playback_follow_active;
                let manual_sync_y = self.manual_edit_scroll_sync_y.unwrap_or(0.0).max(0.0);
                ui.columns(2, |cols| {
                    engine_panel_frame().show(&mut cols[0], |ui| {
                        ui.set_min_height(transcript_height);
                        ui.set_max_height(transcript_height);
                        let highlight_range = active_range;
                        let mut scroll_area = egui::ScrollArea::vertical()
                            .id_salt("original_transcript_lines")
                            .auto_shrink([false, false])
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                            )
                            .max_height(transcript_height);
                        if manual_edit_sync_enabled {
                            scroll_area = scroll_area.vertical_scroll_offset(manual_sync_y);
                        }
                        let scroll_output = scroll_area.show(ui, |ui| {
                            let mut layouter =
                                move |ui: &egui::Ui,
                                      text: &dyn egui::TextBuffer,
                                      wrap_width: f32| {
                                    layouter_with_omitted_highlights(
                                        ui,
                                        text.as_str(),
                                        wrap_width,
                                        highlight_range,
                                    )
                                };
                            let output = egui::TextEdit::multiline(&mut self.transcription_text)
                                .id_salt("original_transcript_text")
                                .font(egui::TextStyle::Body)
                                .desired_width(f32::INFINITY)
                                .min_size(egui::vec2(ui.available_width(), transcript_height))
                                .interactive(false)
                                .layouter(&mut layouter)
                                .show(ui);
                            let seek_overlay = ui.interact(
                                output.response.rect,
                                ui.make_persistent_id("original_transcript_seek_overlay"),
                                egui::Sense::click(),
                            );
                            if seek_overlay.clicked() || seek_overlay.double_clicked() {
                                if let Some(pointer_pos) = seek_overlay.interact_pointer_pos() {
                                    let cursor = output
                                        .galley
                                        .cursor_from_pos(pointer_pos - output.galley_pos);
                                    let line_idx = char_index_to_line_index(
                                        &self.transcription_text,
                                        cursor.index,
                                    );
                                    self.seek_to_line(line_idx);
                                }
                            }
                            if !follow_paused {
                                if let Some(line_idx) = active_anchor {
                                    let char_idx =
                                        line_start_char_index(&self.transcription_text, line_idx);
                                    let cursor = egui::text::CCursor::new(char_idx);
                                    let cursor_rect = output
                                        .galley
                                        .pos_from_cursor(cursor)
                                        .translate(output.galley_pos.to_vec2());
                                    ui.scroll_to_rect(cursor_rect, Some(egui::Align::Center));
                                }
                            }
                        });
                        let user_scrolled =
                            self.user_scrolled_in_rect(ui.ctx(), scroll_output.inner_rect);
                        if user_scrolled {
                            self.pause_follow_after_manual_scroll();
                        }
                        original_scroll_info =
                            Some((scroll_output.state.offset.y.max(0.0), user_scrolled));
                    });

                    engine_panel_frame().show(&mut cols[1], |ui| {
                        ui.set_min_height(transcript_height);
                        ui.set_max_height(transcript_height);
                        let highlight_range = edited_active_range;
                        let mut scroll_area = egui::ScrollArea::vertical()
                            .id_salt("edited_transcript_scroll")
                            .auto_shrink([false, false])
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                            )
                            .max_height(transcript_height);
                        if manual_edit_sync_enabled {
                            scroll_area = scroll_area.vertical_scroll_offset(manual_sync_y);
                        }
                        let scroll_output = scroll_area.show(ui, |ui| {
                            let mut layouter =
                                move |ui: &egui::Ui,
                                      text: &dyn egui::TextBuffer,
                                      wrap_width: f32| {
                                    layouter_with_omitted_highlights(
                                        ui,
                                        text.as_str(),
                                        wrap_width,
                                        highlight_range,
                                    )
                                };
                            let output = egui::TextEdit::multiline(&mut self.edited_text)
                                .id_salt("edited_transcript_text")
                                .font(egui::TextStyle::Body)
                                .desired_width(f32::INFINITY)
                                .min_size(egui::vec2(ui.available_width(), transcript_height))
                                .layouter(&mut layouter)
                                .show(ui);

                            if output.response.clicked() {
                                let delay =
                                    self.settings.edit_cursor_resync_delay_sec.max(0.0) as f64;
                                self.edit_pause_until =
                                    Some(Instant::now() + Duration::from_secs_f64(delay));
                            }
                            let changed = output.response.changed();
                            if changed {
                                let delay = self.settings.edit_autosave_delay_sec.max(0.5) as f64;
                                self.edit_pause_until =
                                    Some(Instant::now() + Duration::from_secs_f64(delay));
                                self.edited_cues = parse_transcript(&self.edited_text).cues;
                            }
                            if output.response.double_clicked() {
                                if let Some(pointer_pos) = output.response.interact_pointer_pos() {
                                    let cursor = output
                                        .galley
                                        .cursor_from_pos(pointer_pos - output.galley_pos);
                                    let line_idx =
                                        char_index_to_line_index(&self.edited_text, cursor.index);
                                    self.seek_to_edited_line(line_idx);
                                } else if let Some(cursor_range) = output.cursor_range {
                                    let line_idx = char_index_to_line_index(
                                        &self.edited_text,
                                        cursor_range.primary.index,
                                    );
                                    self.seek_to_edited_line(line_idx);
                                }
                            }
                            if !follow_paused {
                                if let Some(line_idx) = edited_active_anchor {
                                    let char_idx =
                                        line_start_char_index(&self.edited_text, line_idx);
                                    let cursor = egui::text::CCursor::new(char_idx);
                                    let range = egui::text::CCursorRange::one(cursor);
                                    let mut state = output.state.clone();
                                    state.cursor.set_char_range(Some(range));
                                    state.store(ui.ctx(), output.response.id);
                                    let cursor_rect = output
                                        .galley
                                        .pos_from_cursor(cursor)
                                        .translate(output.galley_pos.to_vec2());
                                    ui.scroll_to_rect(cursor_rect, Some(egui::Align::Center));
                                }
                            }
                        });
                        let user_scrolled =
                            self.user_scrolled_in_rect(ui.ctx(), scroll_output.inner_rect);
                        if user_scrolled {
                            self.pause_follow_after_manual_scroll();
                        }
                        edited_scroll_info =
                            Some((scroll_output.state.offset.y.max(0.0), user_scrolled));
                    });
                });
                let selected_sync_y = if manual_edit_sync_enabled {
                    match (original_scroll_info, edited_scroll_info) {
                        (Some((left_y, left_scrolled)), Some((right_y, right_scrolled))) => {
                            let left_delta = (left_y - manual_sync_y).abs();
                            let right_delta = (right_y - manual_sync_y).abs();
                            if left_delta > 0.5 || right_delta > 0.5 {
                                if left_delta >= right_delta {
                                    Some(left_y)
                                } else {
                                    Some(right_y)
                                }
                            } else if left_scrolled && !right_scrolled {
                                Some(left_y)
                            } else if right_scrolled && !left_scrolled {
                                Some(right_y)
                            } else {
                                self.manual_edit_scroll_sync_y.or(Some(left_y))
                            }
                        }
                        (Some((left_y, _)), None) => Some(left_y),
                        (None, Some((right_y, _))) => Some(right_y),
                        (None, None) => self.manual_edit_scroll_sync_y,
                    }
                } else {
                    original_scroll_info
                        .map(|(left_y, _)| left_y)
                        .or_else(|| edited_scroll_info.map(|(right_y, _)| right_y))
                        .or(self.manual_edit_scroll_sync_y)
                };
                self.manual_edit_scroll_sync_y = selected_sync_y.map(|v| v.max(0.0));
            } else {
                engine_panel_frame().show(ui, |ui| {
                    ui.set_min_height(transcript_height);
                    ui.set_max_height(transcript_height);
                    let scroll_output = egui::ScrollArea::vertical()
                        .id_salt("original_transcript_lines")
                        .auto_shrink([false, false])
                        .scroll_bar_visibility(
                            egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                        )
                        .max_height(transcript_height)
                        .show(ui, |ui| {
                            if transcript_lines.is_empty() {
                                ui.allocate_space(egui::vec2(
                                    ui.available_width(),
                                    (transcript_height - 8.0).max(0.0),
                                ));
                            } else {
                                for (line_idx, line) in transcript_lines.iter().enumerate() {
                                    let active = active_range
                                        .map(|(start, end)| line_idx >= start && line_idx < end)
                                        .unwrap_or(current_line == Some(line_idx));
                                    let response = transcript_line_label(ui, line, active);
                                    if response.clicked() {
                                        self.seek_to_line(line_idx);
                                    }
                                    if active && !follow_paused && Some(line_idx) == active_anchor {
                                        response.scroll_to_me(Some(egui::Align::Center));
                                    }
                                }
                            }
                        });
                    self.mark_follow_pause_on_scroll_area(ui.ctx(), scroll_output.inner_rect);
                });
            }

            self.ui_playback(ui);
        });
    }

    fn ui_chat_parity(&mut self, ui: &mut egui::Ui) {
        self.ensure_chat_source_selected();
        let candidates = self.chat_source_candidates();
        let mut selected_source = self.chat_source_path.clone();
        engine_panel_frame().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Chat");
                ui.separator();
                let selected_label = selected_source
                    .as_ref()
                    .and_then(|p| p.file_name().and_then(|n| n.to_str()))
                    .map(|s| s.to_string())
                    .or_else(|| selected_source.as_ref().map(|p| p.display().to_string()))
                    .unwrap_or_else(|| "<output>".to_string());
                let combo_width = (ui.available_width() * 0.55).clamp(220.0, 760.0);
                egui::ComboBox::from_id_salt("chat_source_dropdown")
                    .selected_text(selected_label)
                    .width(combo_width)
                    .show_ui(ui, |ui| {
                        if candidates.is_empty() {
                            ui.label("No output files available.");
                        } else {
                            for path in &candidates {
                                let label = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| path.display().to_string());
                                let selected =
                                    selected_source.as_ref().map(|p| p == path).unwrap_or(false);
                                let response = ui.selectable_label(selected, label);
                                let response = response.on_hover_text(path.display().to_string());
                                if response.clicked() {
                                    selected_source = Some(path.clone());
                                    ui.close();
                                }
                            }
                        }
                    });
                if accent_button(ui, "Clear chat log").clicked() {
                    if let Some(source_path) = selected_source.clone() {
                        match self.clear_chat_log_for_source(&source_path) {
                            Ok(log_path) => self
                                .push_status(&format!("Cleared chat log: {}", log_path.display())),
                            Err(err) => {
                                self.push_status(&format!("Failed to clear chat log: {err}"))
                            }
                        }
                    }
                }
                if let Some(source_path) = selected_source.as_ref() {
                    let log_path = chat_log_path_for_output(source_path);
                    let log_name = log_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("chat.log");
                    let hint = ui.label(egui::RichText::new(format!("Log: {log_name}")).weak());
                    hint.on_hover_text(log_path.display().to_string());
                }
            });
        });

        if selected_source != self.chat_source_path {
            if let Some(path) = selected_source {
                if let Err(err) = self.load_chat_source_file(path.clone()) {
                    self.push_status(&format!(
                        "Failed to load chat source '{}': {err}",
                        path.display()
                    ));
                }
            } else {
                self.chat_source_path = None;
                self.chat_context_text.clear();
                self.chat_history_text.clear();
            }
        }

        let total_height = ui.available_height().max(0.0);
        let spacing = 8.0;
        let mut prompt_height = (total_height * 0.24).clamp(72.0, 170.0);
        let max_prompt_height = (total_height - spacing).max(40.0);
        prompt_height = prompt_height.min(max_prompt_height);
        let panes_outer_height = (total_height - prompt_height - spacing).max(0.0);
        let pane_body_height = (panes_outer_height - 24.0).max(0.0);

        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), panes_outer_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.columns(2, |cols| {
                    engine_panel_frame().show(&mut cols[0], |ui| {
                        ui.label("Context (selected transcript)");
                        ui.allocate_ui_with_layout(
                            egui::vec2(ui.available_width(), pane_body_height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt("chat_context_scroll")
                                    .auto_shrink([false, false])
                                    .scroll_bar_visibility(
                                        egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                                    )
                                    .max_height(pane_body_height)
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(self.chat_context_text.as_str())
                                                .wrap()
                                                .selectable(true),
                                        );
                                    });
                            },
                        );
                    });
                    engine_panel_frame().show(&mut cols[1], |ui| {
                        ui.label("Chat history");
                        ui.allocate_ui_with_layout(
                            egui::vec2(ui.available_width(), pane_body_height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt("chat_history_scroll")
                                    .auto_shrink([false, false])
                                    .scroll_bar_visibility(
                                        egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                                    )
                                    .max_height(pane_body_height)
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::Label::new(self.chat_history_text.as_str())
                                                .wrap()
                                                .selectable(true),
                                        );
                                    });
                            },
                        );
                    });
                });
            },
        );

        ui.add_space(spacing);
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), prompt_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                engine_panel_frame().show(ui, |ui| {
                    ui.label("Prompt");
                    let reserved_for_header_and_actions = 74.0;
                    let input_height = (prompt_height - reserved_for_header_and_actions).max(28.0);
                    ui.add_sized(
                        [ui.available_width(), input_height],
                        egui::TextEdit::multiline(&mut self.chat_input_text)
                            .desired_width(f32::INFINITY),
                    );
                    ui.horizontal(|ui| {
                        let can_send = !self.is_chatting && self.chat_source_path.is_some();
                        let send = if can_send {
                            accent_button(ui, "Send")
                        } else {
                            ui.add_enabled(false, egui::Button::new("Send"))
                        };
                        if send.clicked() {
                            self.do_start_chat();
                        }
                        if self.chat_source_path.is_none() {
                            ui.label("Select an output file above to enable chat.");
                        }
                    });
                });
            },
        );
    }

    fn ui_transcription_settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_transcription_settings {
            return;
        }
        let mut open = self.show_transcription_settings;
        let mut should_close = false;
        egui::Window::new("Transcription Settings")
            .open(&mut open)
            .default_size([760.0, 320.0])
            .show(ctx, |ui| {
                ui.label("Runtime/device/models moved to Settings -> Runtime Setup.");
                engine_panel_frame().show(ui, |ui| {
                    ui.heading("Transcription Tuning");
                    ui.horizontal(|ui| {
                        ui.label("Subtitle custom:");
                        ui.text_edit_singleline(&mut self.settings.subtitle_custom_mode);
                        ui.label("Speech custom:");
                        ui.text_edit_singleline(&mut self.settings.speech_custom_mode);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Word-time offset sec:");
                        ui.text_edit_singleline(&mut self.settings.whisper_word_time_offset_sec);
                    });
                    ui.label("Word timing offset shifts whisper timestamps so speaker turns align better in diarized output. Keep 0.73 unless you need to retune.");
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if secondary_button(ui, "Open Runtime Setup").clicked() {
                        self.show_runtime_settings = true;
                    }
                    if accent_button(ui, "Save").clicked() {
                        self.queue_save();
                        self.persist_if_needed();
                        should_close = true;
                    }
                    if ui.button("Close").clicked() {
                        should_close = true;
                    }
                });
            });
        if should_close {
            open = false;
        }
        self.show_transcription_settings = open;
    }

    fn ui_chat_settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_chat_settings {
            return;
        }
        let mut open = self.show_chat_settings;
        let mut should_close = false;
        egui::Window::new("Chat Settings")
            .open(&mut open)
            .default_size([820.0, 420.0])
            .show(ctx, |ui| {
                ui.label("Choose the chat model and response settings. If you are unsure, keep defaults.");
                engine_panel_frame().show(ui, |ui| {
                    ui.heading("Model");
                    ui.horizontal(|ui| {
                        ui.label("Chat model file:");
                        ui.text_edit_singleline(&mut self.settings.chat_model);
                        if ui.button("Browse").clicked() {
                            self.do_pick_chat_model();
                        }
                    });
                    if ui.button("Open models folder").clicked() {
                        let _ = open::that(self.paths.models_dir.clone());
                    }
                });
                engine_panel_frame().show(ui, |ui| {
                    ui.heading("Generation");
                    ui.horizontal(|ui| {
                        ui.label("n_ctx:");
                        ui.add(egui::DragValue::new(&mut self.settings.n_ctx).range(-1..=131072));
                        ui.label("n_batch:");
                        ui.add(egui::DragValue::new(&mut self.settings.n_batch).range(1..=8192));
                        ui.label("n_ubatch:");
                        ui.add(egui::DragValue::new(&mut self.settings.n_ubatch).range(1..=8192));
                    });
                    ui.horizontal(|ui| {
                        ui.label("n_parallel:");
                        ui.add(egui::DragValue::new(&mut self.settings.n_parallel).range(1..=16));
                        ui.label("n_threads:");
                        ui.add(egui::DragValue::new(&mut self.settings.n_threads).range(0..=256));
                        ui.label("n_threads_batch:");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.n_threads_batch).range(0..=256),
                        );
                    });
                    ui.label("Device/GPU routing is selected in Runtime Setup -> Execution Device.");
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if accent_button(ui, "Save").clicked() {
                        self.queue_save();
                        self.persist_if_needed();
                        should_close = true;
                    }
                    if ui.button("Close").clicked() {
                        should_close = true;
                    }
                });
            });
        if should_close {
            open = false;
        }
        self.show_chat_settings = open;
    }

    fn ui_editing_settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_editing_settings {
            return;
        }
        let mut open = self.show_editing_settings;
        let mut should_close = false;
        egui::Window::new("Editing Settings")
            .open(&mut open)
            .default_size([620.0, 320.0])
            .show(ctx, |ui| {
                ui.label("Editing always uses split view: original text on the left and editable copy on the right.");
                engine_panel_frame().show(ui, |ui| {
                    ui.heading("Timing");
                    ui.horizontal(|ui| {
                        ui.label("Cursor pause before auto-follow (seconds):");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.edit_cursor_resync_delay_sec)
                                .range(0.0..=120.0)
                                .speed(0.1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Autosave delay after typing stops (seconds):");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.edit_autosave_delay_sec)
                                .range(0.1..=120.0)
                                .speed(0.1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Seek step for < / > (seconds):");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.seek_step_sec)
                                .range(0.1..=60.0)
                                .speed(0.1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Extra seek distance when repeating quickly (seconds):");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.seek_step_growth_sec)
                                .range(0.0..=60.0)
                                .speed(0.1),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Start offset for Play/Pause from transcript line (seconds):");
                        ui.add(
                            egui::DragValue::new(&mut self.settings.playback_toggle_offset_sec)
                                .range(-60.0..=60.0)
                                .speed(0.1),
                        );
                    });
                });
                ui.separator();
                ui.horizontal(|ui| {
                    if accent_button(ui, "Save").clicked() {
                        self.settings.split_view = true;
                        self.queue_save();
                        self.persist_if_needed();
                        should_close = true;
                    }
                    if ui.button("Close").clicked() {
                        should_close = true;
                    }
                });
            });
        if should_close {
            open = false;
        }
        self.show_editing_settings = open;
    }

    fn ui_anonymise_window(&mut self, ctx: &egui::Context) {
        if !self.show_anonymise_window {
            return;
        }
        let mut open = self.show_anonymise_window;
        let mut should_close = false;
        egui::Window::new("Anonymise (beta)")
            .open(&mut open)
            .default_size([720.0, 560.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(
                    "Use the chat model to extract identifiers and replace them in the edited transcript.",
                );
                ui.label("If no edited file exists yet, this will create it and enable split editing view.");
                ui.separator();

                let info = ui.small_button("i");
                info.on_hover_text(ANONYMISE_TOOLTIP);

                if self.settings.chat_model.trim().is_empty() {
                    ui.colored_label(
                        egui::Color32::from_rgb(160, 25, 25),
                        "Chat model is not configured. Set it in Settings -> Chat settings.",
                    );
                } else {
                    ui.label(format!(
                        "Using chat model: {}",
                        self.settings.chat_model.trim()
                    ));
                }

                ui.separator();
                ui.heading("Entity types");
                for kind in ANONYMISE_ENTITY_ORDER {
                    let mut selected = self.anonymise_selected_kinds.contains(&kind);
                    if ui.checkbox(&mut selected, kind.label()).changed() {
                        if selected {
                            self.anonymise_selected_kinds.insert(kind);
                        } else {
                            self.anonymise_selected_kinds.remove(&kind);
                        }
                    }
                    ui.label(
                        egui::RichText::new(kind.prompt_guidance())
                            .small()
                            .weak(),
                    );
                    ui.add_space(2.0);
                }

                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    if secondary_button(ui, "Select all").clicked() {
                        self.anonymise_selected_kinds =
                            ANONYMISE_ENTITY_ORDER.iter().copied().collect();
                    }
                    if secondary_button(ui, "Clear selection").clicked() {
                        self.anonymise_selected_kinds.clear();
                    }
                });

                if !self.anonymise_last_response.trim().is_empty() {
                    ui.separator();
                    ui.collapsing("Last raw model response", |ui| {
                        let mut preview = self.anonymise_last_response.clone();
                        ui.add(
                            egui::TextEdit::multiline(&mut preview)
                                .desired_rows(8)
                                .desired_width(f32::INFINITY)
                                .interactive(false),
                        );
                    });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    let run_clicked = if self.anonymise_running {
                        ui.add_enabled(false, egui::Button::new("Run anonymisation"))
                            .clicked()
                    } else {
                        accent_button(ui, "Run anonymisation").clicked()
                    };
                    if run_clicked {
                        self.do_start_anonymise();
                    }
                    if ui.button("Close").clicked() {
                        should_close = true;
                    }
                });
            });
        if should_close {
            open = false;
        }
        self.show_anonymise_window = open;
    }

    fn ui_speaker_rename_window(&mut self, ctx: &egui::Context) {
        if !self.show_speaker_rename_window {
            return;
        }
        let mut open = self.show_speaker_rename_window;
        let mut should_close = false;
        egui::Window::new("Rename Speakers")
            .open(&mut open)
            .default_size([520.0, 440.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Rename detected speaker tags in the edited transcript.");
                ui.label("Changes are written to the edited file and visible immediately.");
                ui.separator();
                ui.horizontal(|ui| {
                    if secondary_button(ui, "Rescan speakers").clicked() {
                        self.refresh_speaker_rename_entries();
                    }
                });
                ui.separator();

                if self.speaker_rename_entries.is_empty() {
                    ui.label("No speaker tags detected (expected format like SPEAKER_00).");
                } else {
                    egui::ScrollArea::vertical()
                        .id_salt("speaker_rename_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for entry in &mut self.speaker_rename_entries {
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new(&entry.speaker_tag).monospace());
                                    ui.label("->");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut entry.replacement)
                                            .desired_width(260.0)
                                            .hint_text("Name or label"),
                                    );
                                });
                            }
                        });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if accent_button(ui, "Apply renames").clicked() {
                        match self.apply_speaker_renames() {
                            Ok(0) => self.push_status(
                                "Speaker rename: no replacements made (empty names or no matches).",
                            ),
                            Ok(_) => {}
                            Err(err) => {
                                self.open_modal("Speaker rename failed", err.to_string(), true);
                                self.push_status(&format!("Speaker rename failed: {err}"));
                            }
                        }
                    }
                    if ui.button("Close").clicked() {
                        should_close = true;
                    }
                });
            });
        if should_close {
            open = false;
        }
        self.show_speaker_rename_window = open;
    }

    fn on_ui_panic(&mut self, phase: &str, payload: &(dyn std::any::Any + Send)) {
        let panic_text = panic_payload_to_string(payload);
        let backtrace = std::backtrace::Backtrace::force_capture();
        let details = format!("{phase}: {panic_text}\n{backtrace}");
        append_panic_log(&self.paths, phase, &details);
        self.open_modal(
            "Unexpected UI error",
            format!(
                "Recovered from a UI error in '{phase}'.\n\n{panic_text}\n\nA panic log was written to:\n{}",
                self.paths.data_dir.join("panic.log").display()
            ),
            true,
        );
        self.push_status("Recovered from a UI panic; details saved to panic.log.");
    }

    fn update_frame(&mut self, ctx: &egui::Context) {
        self.apply_font_size(ctx);
        self.process_messages();
        self.sync_runtime_missing_popup_session();
        self.persist_if_needed();
        self.handle_hotkeys(ctx);
        self.maybe_predecode_active_audio();

        if self.editing_enabled {
            if let Some(deadline) = self.edit_pause_until {
                if Instant::now() >= deadline {
                    if let Err(err) = self.save_edited_output() {
                        self.push_status(&format!("Autosave failed: {err}"));
                    }
                    self.edit_pause_until = None;
                }
            }
        }

        egui::TopBottomPanel::top("top-menu").show(ctx, |ui| {
            self.ui_menu_bar(ui);
        });

        egui::TopBottomPanel::bottom("status-bar").show(ctx, |ui| {
            self.ui_status_bar_parity(ui);
        });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(ctx.style().visuals.panel_fill)
                    .inner_margin(egui::Margin::same(6)),
            )
            .show(ctx, |ui| match self.tab {
                0 => {
                    self.ui_header_parity(ui);
                    self.ui_transcription_parity(ui);
                }
                1 => self.ui_chat_parity(ui),
                _ => {}
            });

        self.ui_logs_window_parity(ctx);
        self.ui_runtime_settings_window(ctx);
        self.ui_transcription_settings_window(ctx);
        self.ui_chat_settings_window(ctx);
        self.ui_editing_settings_window(ctx);
        self.ui_anonymise_window(ctx);
        self.ui_speaker_rename_window(ctx);
        self.ui_runtime_missing_window(ctx);
        self.ui_modal(ctx);
        self.ui_about(ctx);
        self.ui_legal_docs_window(ctx);

        if self.should_exit {
            self.should_exit = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        let download_active = self
            .runtime_state
            .lock()
            .ok()
            .and_then(|state| state.download_status.clone())
            .is_some();
        if self.is_transcribing
            || self.is_chatting
            || self.anonymise_running
            || self.playback_decode_in_progress
            || playback_is_playing(&self.playback)
            || (self.editing_enabled && self.edit_pause_until.is_some())
            || self.runtime_install_in_progress
            || self.runtime_unblock_in_progress
            || download_active
        {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
    }
}

impl App for UiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.update_frame(ctx);
        }));
        if let Err(payload) = result {
            self.on_ui_panic("app-update", &*payload);
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn start_queue_worker_if_needed(
    runtime_state: Arc<Mutex<RuntimeState>>,
    tx: mpsc::Sender<UiMessage>,
) {
    let should_start = {
        if let Ok(mut state) = runtime_state.lock() {
            if state.queue_worker_running || state.job_queue.is_empty() {
                false
            } else {
                state.queue_worker_running = true;
                true
            }
        } else {
            false
        }
    };

    if !should_start {
        return;
    }

    std::thread::spawn(move || loop {
        let job = {
            if let Ok(mut state) = runtime_state.lock() {
                if let Some(job) = state.job_queue.pop_front() {
                    state.running_jobs += 1;
                    state.active_job = Some(ActiveJob {
                        id: job.id,
                        total_files: job.files.len(),
                        done_files: 0,
                    });
                    let _ = tx.send(UiMessage::Refresh);
                    Some(job)
                } else {
                    state.queue_worker_running = false;
                    state.active_job = None;
                    state.active_stage.clear();
                    let _ = tx.send(UiMessage::Refresh);
                    None
                }
            } else {
                None
            }
        };

        let Some(job) = job else {
            break;
        };

        let total = job.files.len();
        let _ = tx.send(UiMessage::Log(format!(
            "Running job #{} with {} file(s)...",
            job.id, total
        )));

        for (i, file) in job.files.iter().enumerate() {
            let idx = i + 1;
            let mut run_settings = job.settings.clone();
            run_settings.audio_file = file.display().to_string();
            let diarization_on = run_settings.mode.trim().eq_ignore_ascii_case("transcript");
            let device_label = selected_gpu_index_from_settings(&run_settings)
                .map(|idx| format!("gpu:{idx}"))
                .unwrap_or_else(|| "cpu".to_string());
            let _ = tx.send(UiMessage::Log(format!(
                "Job #{} file {idx}/{total} config: mode='{}' custom='{}' whisper_model='{}' diarization={} diarization_models_dir='{}' device='{}' n_ctx={} n_batch={} audio='{}'",
                job.id,
                run_settings.mode,
                run_settings.custom_mode,
                run_settings.whisper_model,
                diarization_on,
                run_settings.diarization_models_dir,
                device_label,
                run_settings.n_ctx,
                run_settings.n_batch,
                run_settings.audio_file
            )));
            if let Ok(mut state) = runtime_state.lock() {
                state.active_stage = format!("job #{} file {idx}/{total}: preparing", job.id);
            }
            let _ = tx.send(UiMessage::Status(format!(
                "Job #{} file {idx}/{total}: preparing {}",
                job.id,
                file.display()
            )));
            match run_transcription_with_progress(run_settings, |stage| {
                if let Ok(mut state) = runtime_state.lock() {
                    state.active_stage = format!("job #{} file {idx}/{total}: {stage}", job.id);
                }
                let _ = tx.send(UiMessage::Status(format!(
                    "Job #{} file {idx}/{total}: {stage}",
                    job.id
                )));
                let _ = tx.send(UiMessage::Log(format!(
                    "Job #{} file {idx}/{total} stage detail: {stage}",
                    job.id
                )));
            }) {
                Ok(result) => {
                    if let Ok(mut state) = runtime_state.lock() {
                        if let Some(active) = &mut state.active_job {
                            active.done_files = idx;
                        }
                    }
                    let _ = tx.send(UiMessage::JobFileOk {
                        job_id: job.id,
                        index: idx,
                        total,
                        audio_path: file.clone(),
                        result,
                    });
                }
                Err(e) => {
                    if let Ok(mut state) = runtime_state.lock() {
                        if let Some(active) = &mut state.active_job {
                            active.done_files = idx;
                        }
                    }
                    let _ = tx.send(UiMessage::JobFileErr {
                        job_id: job.id,
                        index: idx,
                        total,
                        audio_path: file.clone(),
                        error: e.to_string(),
                    });
                }
            }
            let _ = tx.send(UiMessage::Refresh);
        }

        if let Ok(mut state) = runtime_state.lock() {
            state.running_jobs = state.running_jobs.saturating_sub(1);
            state.completed_jobs += 1;
            state.active_job = None;
            state.active_stage.clear();
        }
        let _ = tx.send(UiMessage::JobDone(job.id));
        let _ = tx.send(UiMessage::Refresh);
    });
}

fn select_whisper_model_index(whisper_model: &str) -> usize {
    if whisper_model.trim().is_empty() {
        0
    } else {
        whisper_label_from_path(Path::new(whisper_model))
            .and_then(|label| {
                WHISPER_MODELS
                    .iter()
                    .position(|spec| spec.label.eq_ignore_ascii_case(label))
            })
            .or_else(|| {
                WHISPER_MODELS
                    .iter()
                    .position(|spec| spec.file_name.eq_ignore_ascii_case(whisper_model))
            })
            .unwrap_or(0)
    }
}

fn selected_gpu_index_from_settings(settings: &AppSettings) -> Option<i32> {
    let devices_trimmed = settings.devices.trim();
    if devices_trimmed.is_empty() || devices_trimmed.eq_ignore_ascii_case("none") {
        // Legacy fallback: keep explicit non-zero main_gpu selection from older settings.
        if !settings.whisper_no_gpu && settings.main_gpu > 0 {
            return Some(settings.main_gpu);
        }
        return None;
    }

    if let Some(index) = devices_trimmed
        .split(',')
        .next()
        .map(str::trim)
        .and_then(|v| v.parse::<i32>().ok())
    {
        return Some(index.max(0));
    }

    if !settings.whisper_no_gpu && settings.main_gpu > 0 {
        return Some(settings.main_gpu);
    }

    None
}

fn default_runtime_backends_for_platform() -> Vec<String> {
    if cfg!(target_os = "windows") {
        return vec!["vulkan".to_string(), "cuda".to_string()];
    }
    if cfg!(target_os = "macos") {
        return vec!["metal".to_string()];
    }
    vec!["vulkan".to_string()]
}

fn sort_runtime_backends_for_platform(options: &mut [String]) {
    if cfg!(target_os = "windows") {
        options.sort_by_key(|v| match v.to_ascii_lowercase().as_str() {
            "vulkan" => 0,
            "cuda" => 1,
            _ => 9,
        });
    }
}

fn runtime_backend_options(paths: &AppPaths) -> Vec<String> {
    let mut options = runtime_installer::available_runtime_backends(paths)
        .unwrap_or_else(|_| default_runtime_backends_for_platform())
        .into_iter()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>();
    options.sort();
    options.dedup();
    if options.is_empty() {
        options = default_runtime_backends_for_platform();
    }
    sort_runtime_backends_for_platform(&mut options);
    options
}

fn resolve_runtime_backend_index(backends: &[String], configured_backend: &str) -> usize {
    if backends.is_empty() {
        return 0;
    }
    let configured = configured_backend.trim();
    if configured.is_empty() {
        return 0;
    }
    backends
        .iter()
        .position(|v| v.eq_ignore_ascii_case(configured))
        .unwrap_or(0)
}

fn sync_runtime_backend_from_installed_runtime(
    runtime_dir: &Path,
    settings: &mut AppSettings,
    backends: &mut Vec<String>,
    selected_backend_index: &mut usize,
) -> bool {
    #[cfg(target_os = "windows")]
    {
        if backends.is_empty() {
            return false;
        }
        let Some(detected_backend) = detect_installed_windows_runtime_backend(runtime_dir) else {
            return false;
        };
        if !backends
            .iter()
            .any(|v| v.eq_ignore_ascii_case(&detected_backend))
        {
            backends.push(detected_backend.clone());
            backends.sort();
            backends.dedup();
            sort_runtime_backends_for_platform(backends);
        }
        let detected_idx = resolve_runtime_backend_index(backends, &detected_backend);
        let mut changed = false;
        if *selected_backend_index != detected_idx {
            *selected_backend_index = detected_idx;
            changed = true;
        }
        if !settings
            .runtime_download_backend
            .trim()
            .eq_ignore_ascii_case(&detected_backend)
        {
            settings.runtime_download_backend = detected_backend;
            changed = true;
        }
        return changed;
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = (runtime_dir, settings, backends, selected_backend_index);
        false
    }
}

fn should_auto_select_gpu_from_settings(settings: &AppSettings) -> bool {
    !settings.whisper_no_gpu && selected_gpu_index_from_settings(settings).is_none()
}

fn apply_audio_device_to_settings_from_index(
    settings: &mut AppSettings,
    devices: &[AudioDeviceOption],
    index: usize,
) -> bool {
    let Some(option) = devices.get(index) else {
        return false;
    };
    if !option.is_gpu {
        return false;
    }

    let changed = settings.whisper_no_gpu
        || settings.main_gpu != option.main_gpu
        || settings.devices != option.devices_value;
    settings.whisper_no_gpu = false;
    settings.main_gpu = option.main_gpu;
    settings.devices = option.devices_value.clone();
    changed
}

#[cfg(target_os = "macos")]
fn is_metal_gpu_option(device: &AudioDeviceOption) -> bool {
    device.label.to_ascii_lowercase().contains("metal")
        || device.detail_line.to_ascii_lowercase().contains("metal")
}

#[cfg(target_os = "macos")]
fn preferred_platform_gpu_index(devices: &[AudioDeviceOption]) -> Option<usize> {
    devices
        .iter()
        .position(|d| d.is_gpu && is_metal_gpu_option(d))
        .or_else(|| devices.iter().position(|d| d.is_gpu))
}

#[cfg(not(target_os = "macos"))]
fn preferred_platform_gpu_index(_devices: &[AudioDeviceOption]) -> Option<usize> {
    None
}

fn resolve_audio_device_index(devices: &[AudioDeviceOption], settings: &AppSettings) -> usize {
    let selected_gpu = selected_gpu_index_from_settings(settings);
    let Some(selected_gpu) = selected_gpu else {
        if settings.whisper_no_gpu {
            return 0;
        }
        if let Some(found) = preferred_platform_gpu_index(devices) {
            return found;
        }
        return 0;
    };

    if let Some(found) = devices
        .iter()
        .position(|d| d.is_gpu && d.main_gpu == selected_gpu)
    {
        return found;
    }
    if let Some(found) = devices
        .iter()
        .position(|d| d.is_gpu && d.devices_value == selected_gpu.to_string())
    {
        return found;
    }
    if let Some(found) = devices.iter().position(|d| d.is_gpu) {
        return found;
    }
    0
}

fn log_timestamp_hms() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let day = secs % 86_400;
    let h = day / 3600;
    let m = (day % 3600) / 60;
    let s = day % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn transcript_line_label(ui: &mut egui::Ui, line: &str, active: bool) -> egui::Response {
    let text = if active {
        egui::RichText::new(line)
            .background_color(egui::Color32::from_rgb(255, 245, 200))
            .color(egui::Color32::from_rgb(75, 50, 0))
            .strong()
    } else {
        egui::RichText::new(line)
    };
    ui.add(egui::Label::new(text).sense(egui::Sense::click()))
}

fn detect_speaker_slots(text: &str) -> Vec<String> {
    let mut max_idx = None::<u32>;
    let mut width = 2usize;
    for cap in speaker_tag_regex().captures_iter(text) {
        let digits = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        if let Ok(idx) = digits.parse::<u32>() {
            max_idx = Some(max_idx.map(|cur| cur.max(idx)).unwrap_or(idx));
            width = width.max(digits.len());
        }
    }
    let Some(max_idx) = max_idx else {
        return Vec::new();
    };
    (0..=max_idx)
        .map(|idx| format!("SPEAKER_{idx:0width$}", width = width))
        .collect()
}

fn speaker_tag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bSPEAKER[_ ]?(\d{1,3})\b").expect("valid regex"))
}

fn line_start_char_index(text: &str, line_index: usize) -> usize {
    if line_index == 0 {
        return 0;
    }
    let byte_index = text
        .match_indices('\n')
        .nth(line_index.saturating_sub(1))
        .map(|(idx, _)| idx + 1)
        .unwrap_or(text.len());
    text[..byte_index].chars().count()
}

fn char_index_to_line_index(text: &str, char_index: usize) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut line = 0usize;
    let mut chars_seen = 0usize;
    for ch in text.chars() {
        if chars_seen >= char_index {
            break;
        }
        if ch == '\n' {
            line = line.saturating_add(1);
        }
        chars_seen = chars_seen.saturating_add(1);
    }
    line
}

fn is_edited_output_path(path: &Path) -> bool {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|stem| stem.ends_with(".edited"))
        .unwrap_or(false)
}

fn original_output_path_for_any(path: &Path) -> PathBuf {
    if !is_edited_output_path(path) {
        return path.to_path_buf();
    }

    let parent = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let base_stem = stem.strip_suffix(".edited").unwrap_or(stem);
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        parent.join(format!("{base_stem}.{ext}"))
    } else {
        parent.join(base_stem)
    }
}

fn ensure_output_entry(entries: &mut Vec<OutputEntry>, path: PathBuf, selected: bool) {
    if let Some(existing) = entries.iter_mut().find(|entry| entry.path == path) {
        if selected {
            existing.selected = true;
        }
        return;
    }
    entries.push(OutputEntry { path, selected });
}

fn existing_output_paths_for_media(media_path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(parent) = media_path.parent() else {
        return out;
    };
    let Some(stem) = media_path.file_stem().and_then(|s| s.to_str()) else {
        return out;
    };
    if stem.trim().is_empty() {
        return out;
    }

    let original = parent.join(format!("{stem}.md"));
    if original.is_file() {
        out.push(original);
    }

    let edited = parent.join(format!("{stem}.edited.md"));
    if edited.is_file() {
        out.push(edited);
    }

    out
}

fn supported_media_extensions() -> &'static [&'static str] {
    &[
        "wav", "mp3", "m4a", "mp4", "mkv", "flac", "ogg", "opus", "aac", "wma", "webm", "mov",
        "aiff", "aif",
    ]
}

fn looks_like_audio_extension(ext: &str) -> bool {
    supported_media_extensions()
        .iter()
        .any(|allowed| ext.eq_ignore_ascii_case(allowed))
}

fn guess_audio_path_for_output(output_path: &Path) -> Option<PathBuf> {
    let parent = output_path.parent()?;
    let stem = output_path.file_stem()?.to_string_lossy();
    let base = stem.strip_suffix(".edited").unwrap_or(&stem).to_string();

    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let ext_ok = p
                .extension()
                .and_then(|e| e.to_str())
                .map(looks_like_audio_extension)
                .unwrap_or(false);
            if !ext_ok {
                continue;
            }
            let Some(audio_stem) = p.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if audio_stem.eq_ignore_ascii_case(base.as_str()) {
                return Some(p);
            }
        }
    }
    None
}

fn build_anonymise_prompt(source_text: &str, selected_kinds: &[AnonymiseEntityKind]) -> String {
    let selected_keys = selected_kinds
        .iter()
        .map(|kind| kind.key())
        .collect::<Vec<_>>()
        .join(", ");
    let selected_fragment_examples = selected_kinds
        .iter()
        .map(|kind| kind.prompt_kv_example())
        .collect::<Vec<_>>()
        .join(" ");
    let selected_example_response = format!("{{{selected_fragment_examples}}}");
    let selected_empty_response = "{none}".to_string();
    let selected_key_guidance = selected_kinds
        .iter()
        .map(|kind| {
            format!(
                "- key: {}\n  meaning: {}\n  Required fragment example: {}\n  Repeat key for every match.",
                kind.key(),
                kind.prompt_guidance(),
                kind.prompt_kv_example()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let selected_field_specs = selected_kinds
        .iter()
        .map(|kind| {
            format!(
                "  - key: {}\n    type: repeated_fragment_in_string\n    guidance: |\n      {}\n      Required fragment example:\n      {}\n      Emit one `key: value;` per match.",
                kind.key(),
                kind.prompt_guidance(),
                kind.prompt_kv_example()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "name: anonymisation_extraction\n\
         description: >\n\
           Extract selected anonymisation entities from transcript text and return one strict pseudo-JSON string.\n\
         system_prompt: |\n\
            You are a meticulous anonymisation extraction assistant.\n\
            Work strictly from the supplied transcript text.\n\
            Ignore transcript metadata labels and timing markers themselves (for example, SPEAKER_<number>, turn headers, and [hh:mm:ss - hh:mm:ss] ranges).\n\
            Important: ignore only those labels/markers, not the spoken content that follows them.\n\
            Return exactly one pseudo-JSON string only.\n\
            Never include commentary, code fences, or <think> sections.\n\
            Never invent values.\n\
            Copy matched values letter-by-letter from the transcript.\n\
          format_instructions: |\n\
           Output rules (strict):\n\
             - Respond with ONE line in this exact shape: `{{key: value; key: value;}}`\n\
             - Use outer braces exactly once.\n\
             - Allowed keys are only: {selected_keys}\n\
             - Repeat the key for every single match.\n\
             - Copy each value letter-by-letter from transcript.\n\
             - Never extract transcript labels/metadata tokens themselves (SPEAKER_<number>, turn labels, timing brackets).\n\
             - Still extract entities from the actual spoken text within those turns.\n\
             - Never output placeholders/tokens from examples (anything inside `<...>`).\n\
             - If nothing matched, return exactly: {selected_empty_response}\n\
             - Do not output quoted keys, arrays, or wrapper keys.\n\
           Example response:\n\
             {selected_example_response}\n\
           Example response when nothing matched:\n\
             {selected_empty_response}\n\
         fields:\n\
         {selected_field_specs}\n\
         category_guidance: |\n\
         {selected_key_guidance}\n\
         transcript: |\n\
           <<<BEGIN_TRANSCRIPT>>>\n\
           {source_text}\n\
           <<<END_TRANSCRIPT>>>"
    )
}

fn parse_anonymise_matches(raw_response: &str) -> Result<Vec<AnonymiseMatch>> {
    let trimmed = raw_response.trim();
    let normalized_none = trimmed
        .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
        .to_ascii_lowercase();

    if normalized_none == "none"
        || normalized_none == "nomatches"
        || normalized_none == "nomatch"
        || normalized_none == "noentities"
        || normalized_none == "empty"
    {
        return Ok(Vec::new());
    }

    if let Ok(parsed_direct) = serde_json::from_str::<Value>(trimmed) {
        let parsed = parse_anonymise_matches_from_value(&parsed_direct);
        if !parsed.is_empty() || is_valid_empty_anonymise_json(&parsed_direct) {
            return Ok(parsed);
        }
    }

    let mut multi_json_matches = Vec::new();
    let mut saw_valid_empty_json = false;
    for json_object in extract_json_objects(raw_response) {
        if let Ok(parsed) = serde_json::from_str::<Value>(&json_object) {
            let parsed_matches = parse_anonymise_matches_from_value(&parsed);
            if !parsed_matches.is_empty() {
                multi_json_matches.extend(parsed_matches);
            } else if is_valid_empty_anonymise_json(&parsed) {
                saw_valid_empty_json = true;
            }
        }
    }
    if !multi_json_matches.is_empty() {
        return Ok(multi_json_matches);
    }
    if saw_valid_empty_json {
        return Ok(Vec::new());
    }

    let fallback = parse_anonymise_matches_from_key_value_text(raw_response);
    if !fallback.is_empty() {
        return Ok(fallback);
    }

    if normalized_none.contains("none") || normalized_none.contains("nomatch") {
        return Ok(Vec::new());
    }

    let snippet = raw_response
        .lines()
        .take(6)
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .take(320)
        .collect::<String>();
    Err(anyhow!(
        "model response did not contain parseable JSON objects or key:value matches; response starts with: {}",
        snippet
    ))
}

fn parse_anonymise_matches_from_value(parsed: &Value) -> Vec<AnonymiseMatch> {
    let mut out = Vec::new();

    if let Some(obj) = parsed.as_object() {
        if let Some(payload) = obj
            .get("anonymisation_extractions")
            .or_else(|| obj.get("anonymization_extractions"))
            .or_else(|| obj.get("extractions"))
            .or_else(|| obj.get("matches_text"))
            .or_else(|| obj.get("matches_string"))
            .and_then(Value::as_str)
        {
            let direct = parse_anonymise_matches_from_key_value_text(payload);
            if !direct.is_empty() {
                return direct;
            }
            let normalized_payload = payload
                .trim()
                .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
                .to_ascii_lowercase();
            if normalized_payload == "none" || normalized_payload == "nomatches" {
                return Vec::new();
            }
        }
    }

    if let Some(matches) = parsed.get("matches").and_then(Value::as_array) {
        for item in matches {
            if let Some(obj) = item.as_object() {
                let kind = obj
                    .get("type")
                    .or_else(|| obj.get("kind"))
                    .or_else(|| obj.get("category"))
                    .and_then(Value::as_str)
                    .and_then(parse_anonymise_kind);
                let value = obj
                    .get("value")
                    .or_else(|| obj.get("text"))
                    .or_else(|| obj.get("match"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if let (Some(kind), Some(value)) = (kind, value) {
                    out.push(AnonymiseMatch { kind, value });
                    continue;
                }
            }
            if let Some(item_text) = item.as_str() {
                if let Some((kind_token, value)) = item_text.split_once(':') {
                    if let Some(kind) = parse_anonymise_kind(kind_token) {
                        out.push(AnonymiseMatch {
                            kind,
                            value: value.trim().to_string(),
                        });
                    }
                }
            }
        }
    }

    if let Some(obj) = parsed.as_object() {
        for (key, value) in obj {
            if key.eq_ignore_ascii_case("matches") {
                continue;
            }
            if let Some(kind) = parse_anonymise_kind(key) {
                collect_anonymise_values(kind, value, &mut out);
            }
        }
    } else if let Some(arr) = parsed.as_array() {
        for value in arr {
            if let Some(obj) = value.as_object() {
                if let (Some(kind), Some(text)) = (
                    obj.get("type")
                        .and_then(Value::as_str)
                        .and_then(parse_anonymise_kind),
                    obj.get("value")
                        .or_else(|| obj.get("text"))
                        .and_then(Value::as_str),
                ) {
                    out.push(AnonymiseMatch {
                        kind,
                        value: text.to_string(),
                    });
                }
            }
        }
    }

    out
}

fn is_valid_empty_anonymise_json(parsed: &Value) -> bool {
    let Some(obj) = parsed.as_object() else {
        return false;
    };

    if obj.is_empty() {
        return true;
    }

    if obj
        .get("matches")
        .and_then(Value::as_array)
        .map(|arr| arr.is_empty())
        .unwrap_or(false)
    {
        return true;
    }

    let legacy_none = obj
        .get("anonymisation_extractions")
        .or_else(|| obj.get("anonymization_extractions"))
        .or_else(|| obj.get("extractions"))
        .and_then(Value::as_str)
        .map(|text| {
            let norm = text
                .trim()
                .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
                .to_ascii_lowercase();
            norm == "none" || norm == "nomatches" || norm == "nomatch"
        })
        .unwrap_or(false);
    if legacy_none {
        return true;
    }

    let mut saw_recognized = false;
    for (key, value) in obj {
        if parse_anonymise_kind(key).is_none() {
            continue;
        }
        saw_recognized = true;
        let empty_value = value.is_null()
            || value.as_array().map(|arr| arr.is_empty()).unwrap_or(false)
            || value
                .as_str()
                .map(|s| {
                    let norm = s
                        .trim()
                        .trim_matches(|ch: char| ch.is_ascii_punctuation() || ch.is_whitespace())
                        .to_ascii_lowercase();
                    norm.is_empty() || norm == "none"
                })
                .unwrap_or(false);
        if !empty_value {
            return false;
        }
    }

    saw_recognized
}

fn parse_anonymise_matches_from_key_value_text(raw: &str) -> Vec<AnonymiseMatch> {
    let mut out = Vec::new();
    for cap in anonymise_key_value_regex().captures_iter(raw) {
        let Some(kind) = cap
            .name("kind")
            .map(|m| m.as_str())
            .and_then(parse_anonymise_kind)
        else {
            continue;
        };
        let value = cap
            .name("value")
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        if value.is_empty() {
            continue;
        }
        out.push(AnonymiseMatch { kind, value });
    }
    out
}

fn collect_anonymise_values(
    kind: AnonymiseEntityKind,
    value: &Value,
    out: &mut Vec<AnonymiseMatch>,
) {
    if let Some(text) = value.as_str() {
        out.push(AnonymiseMatch {
            kind,
            value: text.to_string(),
        });
        return;
    }
    if let Some(arr) = value.as_array() {
        for entry in arr {
            collect_anonymise_values(kind, entry, out);
        }
        return;
    }
    if let Some(obj) = value.as_object() {
        if let Some(text) = obj
            .get("value")
            .or_else(|| obj.get("text"))
            .or_else(|| obj.get("match"))
            .and_then(Value::as_str)
        {
            out.push(AnonymiseMatch {
                kind,
                value: text.to_string(),
            });
        }
    }
}

fn extract_json_objects(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        if serde_json::from_str::<Value>(trimmed).is_ok() {
            out.push(trimmed.to_string());
            return out;
        }
    }

    let mut depth = 0usize;
    let mut start = None;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in raw.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    if let Some(start_idx) = start {
                        let end = idx + ch.len_utf8();
                        let candidate = &raw[start_idx..end];
                        if serde_json::from_str::<Value>(candidate).is_ok() {
                            out.push(candidate.to_string());
                        }
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }

    out
}

fn parse_anonymise_kind(raw: &str) -> Option<AnonymiseEntityKind> {
    let normalized = normalize_kind_token(raw);
    match normalized.as_str() {
        "name" | "names" | "person" | "people" => Some(AnonymiseEntityKind::Name),
        "location" | "locations" | "place" | "places" => Some(AnonymiseEntityKind::Location),
        "address" | "addresses" => Some(AnonymiseEntityKind::Address),
        "phone" | "phones" | "phonenumber" | "phonenumbers" | "telephone" | "telephonenumber"
        | "mobile" => Some(AnonymiseEntityKind::Phone),
        "email" | "emails" | "emailaddress" | "emailaddresses" => Some(AnonymiseEntityKind::Email),
        "number" | "numbers" => Some(AnonymiseEntityKind::Number),
        "organisation" | "organisations" | "organization" | "organizations" | "company"
        | "companies" | "institution" | "institutions" => Some(AnonymiseEntityKind::Organisation),
        "identifier" | "identifiers" | "id" | "ids" | "code" | "codes" => {
            Some(AnonymiseEntityKind::Identifier)
        }
        _ => None,
    }
}

fn normalize_kind_token(raw: &str) -> String {
    raw.trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
}

fn apply_anonymise_to_text(text: &str, matches: &[AnonymiseMatch]) -> (String, usize, usize) {
    let candidates = build_replacement_candidates(matches);
    apply_anonymise_to_text_with_candidates(text, &candidates)
}

fn apply_anonymise_to_text_with_candidates(
    text: &str,
    candidates: &[AnonymiseMatch],
) -> (String, usize, usize) {
    if text.is_empty() {
        return (String::new(), 0, 0);
    }
    let mut rewritten = text.to_string();
    let mut exact_total = 0usize;
    let mut fuzzy_total = 0usize;

    for candidate in candidates {
        let (after_exact, exact_count) = replace_exact_case_insensitive(
            &rewritten,
            &candidate.value,
            candidate.kind.placeholder(),
        );
        rewritten = after_exact;
        exact_total += exact_count;

        if candidate.kind.supports_fuzzy() {
            let (after_near, near_count) = replace_near_exact_phrase(
                &rewritten,
                &candidate.value,
                candidate.kind.placeholder(),
            );
            rewritten = after_near;
            fuzzy_total += near_count;

            let (after_fuzzy, fuzzy_count) =
                replace_fuzzy_phrase(&rewritten, &candidate.value, candidate.kind.placeholder());
            rewritten = after_fuzzy;
            fuzzy_total += fuzzy_count;
        }
    }

    (rewritten, exact_total, fuzzy_total)
}

fn build_replacement_candidates(matches: &[AnonymiseMatch]) -> Vec<AnonymiseMatch> {
    let mut out = Vec::new();
    let mut seen = HashSet::<String>::new();
    for entry in matches {
        let cleaned = sanitize_candidate_value(&entry.value);
        if !is_candidate_plausible(entry.kind, &cleaned) {
            continue;
        }
        let key = format!(
            "{}:{}",
            entry.kind.key(),
            candidate_key_for_dedupe(&cleaned)
        );
        if seen.insert(key) {
            out.push(AnonymiseMatch {
                kind: entry.kind,
                value: cleaned,
            });
        }
    }
    out.sort_by(|a, b| b.value.len().cmp(&a.value.len()));
    out
}

fn sanitize_candidate_value(raw: &str) -> String {
    let mut out = raw
        .trim()
        .trim_matches(|ch| matches!(ch, '"' | '\'' | '`'))
        .trim()
        .to_string();
    while out
        .chars()
        .next_back()
        .map(|ch| matches!(ch, '.' | ',' | ';' | ':' | '!' | '?'))
        .unwrap_or(false)
    {
        out.pop();
        out = out.trim_end().to_string();
    }
    out
}

fn candidate_key_for_dedupe(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn is_candidate_plausible(kind: AnonymiseEntityKind, value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 160 {
        return false;
    }
    let trimmed_lc = trimmed.to_ascii_lowercase();
    if trimmed.contains('<') || trimmed.contains('>') {
        return false;
    }
    if trimmed_lc.contains("from_transcript")
        || trimmed_lc.contains("placeholder")
        || trimmed_lc.contains("example")
    {
        return false;
    }
    if trimmed.starts_with('[') && trimmed.to_ascii_lowercase().contains(" omitted]") {
        return false;
    }
    let digits = trimmed.chars().filter(|ch| ch.is_ascii_digit()).count();
    match kind {
        AnonymiseEntityKind::Phone
        | AnonymiseEntityKind::Number
        | AnonymiseEntityKind::Identifier => digits >= 2 || trimmed.len() >= 4,
        _ => trimmed.chars().any(|ch| ch.is_alphabetic()) || digits >= 3,
    }
}

fn replace_exact_case_insensitive(text: &str, phrase: &str, replacement: &str) -> (String, usize) {
    if text.is_empty() || phrase.trim().is_empty() {
        return (text.to_string(), 0);
    }
    let pattern = build_exact_match_pattern(phrase);
    let Ok(re) = Regex::new(&pattern) else {
        return (text.to_string(), 0);
    };
    let count = re.find_iter(text).count();
    if count == 0 {
        return (text.to_string(), 0);
    }
    (re.replace_all(text, replacement).to_string(), count)
}

fn build_exact_match_pattern(phrase: &str) -> String {
    let escaped = regex::escape(phrase.trim());
    let starts_word = phrase
        .chars()
        .next()
        .map(|ch| ch.is_alphanumeric() || ch == '_')
        .unwrap_or(false);
    let ends_word = phrase
        .chars()
        .next_back()
        .map(|ch| ch.is_alphanumeric() || ch == '_')
        .unwrap_or(false);
    if starts_word && ends_word {
        format!(r"(?i)\b{escaped}\b")
    } else {
        format!(r"(?i){escaped}")
    }
}

fn replace_near_exact_phrase(text: &str, phrase: &str, replacement: &str) -> (String, usize) {
    if text.is_empty() || phrase.trim().is_empty() {
        return (text.to_string(), 0);
    }
    let phrase_norm = normalize_for_fuzzy(phrase);
    let phrase_compact = phrase_norm.replace(' ', "");
    let target_tokens = fuzzy_token_spans(phrase);
    if phrase_compact.len() < 4 || target_tokens.is_empty() || target_tokens.len() > 8 {
        return (text.to_string(), 0);
    }
    let spans = fuzzy_token_spans(text);
    if spans.is_empty() {
        return (text.to_string(), 0);
    }

    let omitted_spans = omitted_placeholder_regex()
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect::<Vec<_>>();
    let window = target_tokens.len();
    if window > spans.len() {
        return (text.to_string(), 0);
    }

    let mut near_hits = Vec::<(usize, usize)>::new();
    for start_idx in 0..=(spans.len() - window) {
        let first = spans[start_idx].0;
        let last = spans[start_idx + window - 1].1;
        if omitted_spans.iter().any(|(a, b)| first < *b && *a < last) {
            continue;
        }
        let segment = &text[first..last];
        let segment_compact = normalize_for_fuzzy(segment).replace(' ', "");
        if segment_compact.is_empty() {
            continue;
        }
        let max_len = phrase_compact.len().max(segment_compact.len());
        if max_len < 4 {
            continue;
        }
        let dist = strsim::levenshtein(&segment_compact, &phrase_compact);
        let near_enough = dist <= 1 || (dist == 2 && max_len >= 12);
        if near_enough {
            near_hits.push((first, last));
        }
    }

    if near_hits.is_empty() {
        return (text.to_string(), 0);
    }

    near_hits.sort_by(|a, b| b.0.cmp(&a.0));
    let mut selected = Vec::<(usize, usize)>::new();
    for (start, end) in near_hits {
        if selected.iter().any(|(a, b)| start < *b && *a < end) {
            continue;
        }
        selected.push((start, end));
    }
    if selected.is_empty() {
        return (text.to_string(), 0);
    }

    let mut rewritten = text.to_string();
    for (start, end) in &selected {
        rewritten.replace_range(*start..*end, replacement);
    }
    (rewritten, selected.len())
}

fn replace_fuzzy_phrase(text: &str, phrase: &str, replacement: &str) -> (String, usize) {
    if text.is_empty() || phrase.trim().is_empty() {
        return (text.to_string(), 0);
    }
    let phrase_norm = normalize_for_fuzzy(phrase);
    let target_tokens = fuzzy_token_spans(phrase);
    if phrase_norm.len() < 5 || target_tokens.is_empty() || target_tokens.len() > 8 {
        return (text.to_string(), 0);
    }

    let spans = fuzzy_token_spans(text);
    if spans.is_empty() {
        return (text.to_string(), 0);
    }

    let target_len = target_tokens.len();
    let min_window = target_len.saturating_sub(1).max(1);
    let max_window = (target_len + 1).min(target_len + 2);
    let threshold = fuzzy_similarity_threshold(target_len);
    let omitted_spans = omitted_placeholder_regex()
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect::<Vec<_>>();

    let mut fuzzy_hits = Vec::<(usize, usize, f64)>::new();
    for window in min_window..=max_window {
        if window > spans.len() {
            continue;
        }
        for start_idx in 0..=(spans.len() - window) {
            let first = spans[start_idx].0;
            let last = spans[start_idx + window - 1].1;
            if omitted_spans.iter().any(|(a, b)| first < *b && *a < last) {
                continue;
            }
            let segment = &text[first..last];
            let segment_norm = normalize_for_fuzzy(segment);
            if segment_norm.is_empty() {
                continue;
            }
            let ratio = segment_norm.len() as f64 / phrase_norm.len() as f64;
            if !(0.7..=1.4).contains(&ratio) {
                continue;
            }
            let score = strsim::jaro_winkler(&segment_norm, &phrase_norm);
            if score >= threshold {
                fuzzy_hits.push((first, last, score));
            }
        }
    }

    if fuzzy_hits.is_empty() {
        return (text.to_string(), 0);
    }

    fuzzy_hits.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut selected = Vec::<(usize, usize)>::new();
    for (start, end, _) in fuzzy_hits {
        if selected.iter().any(|(a, b)| start < *b && *a < end) {
            continue;
        }
        selected.push((start, end));
    }
    if selected.is_empty() {
        return (text.to_string(), 0);
    }

    selected.sort_by(|a, b| b.0.cmp(&a.0));
    let mut rewritten = text.to_string();
    for (start, end) in &selected {
        rewritten.replace_range(*start..*end, replacement);
    }
    (rewritten, selected.len())
}

fn fuzzy_similarity_threshold(token_count: usize) -> f64 {
    if token_count <= 1 {
        0.97
    } else if token_count <= 3 {
        0.94
    } else {
        0.91
    }
}

fn normalize_for_fuzzy(text: &str) -> String {
    fuzzy_token_regex()
        .find_iter(text)
        .map(|m| m.as_str().to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn fuzzy_token_spans(text: &str) -> Vec<(usize, usize)> {
    fuzzy_token_regex()
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect::<Vec<_>>()
}

fn fuzzy_token_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?u)[\p{L}\p{N}][\p{L}\p{N}'.\-]*").expect("valid regex"))
}

fn omitted_placeholder_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\[[a-z ]+ omitted\]").expect("valid regex"))
}

fn transcript_timing_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"\[\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\s*-\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?\]",
        )
        .expect("valid regex")
    })
}

fn anonymise_key_value_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?im)(?:^|[;\n\r\{\}\[\],]\s*|[-*]\s*)\b(?P<kind>name|names|location|locations|address|addresses|phone|phones|email|emails|number|numbers|organisation|organisations|organization|organizations|identifier|identifiers|id|ids)\b\s*[:=-]\s*(?P<value>[^;\n\r\}]+)"#,
        )
        .expect("valid regex")
    })
}

fn layouter_with_omitted_highlights(
    ui: &egui::Ui,
    text: &str,
    wrap_width: f32,
    active_line_range: Option<(usize, usize)>,
) -> Arc<egui::Galley> {
    let mut layout_job = egui::text::LayoutJob::default();
    layout_job.wrap.max_width = wrap_width;

    let mut normal = egui::TextFormat::default();
    normal.font_id = egui::TextStyle::Body.resolve(ui.style());
    normal.color = ui.visuals().text_color();
    let mut omitted = normal.clone();
    omitted.color = egui::Color32::from_rgb(160, 25, 25);
    let mut timing = normal.clone();
    timing.background = egui::Color32::from_rgb(255, 245, 200);
    timing.color = egui::Color32::from_rgb(75, 50, 0);

    let omitted_ranges = omitted_placeholder_regex()
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect::<Vec<_>>();
    let timing_ranges = if let Some((start_line, end_line)) = active_line_range {
        let mut ranges = Vec::<(usize, usize)>::new();
        let mut line_idx = 0usize;
        let mut line_offset = 0usize;
        for line in text.split_inclusive('\n') {
            if line_idx >= start_line && line_idx < end_line {
                for m in transcript_timing_regex().find_iter(line) {
                    ranges.push((line_offset + m.start(), line_offset + m.end()));
                }
            }
            line_offset += line.len();
            line_idx += 1;
        }
        ranges
    } else {
        Vec::new()
    };

    let mut boundaries =
        Vec::<usize>::with_capacity(2 + (omitted_ranges.len() + timing_ranges.len()) * 2);
    boundaries.push(0);
    boundaries.push(text.len());
    for (start, end) in omitted_ranges.iter().chain(timing_ranges.iter()) {
        boundaries.push(*start);
        boundaries.push(*end);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    for pair in boundaries.windows(2) {
        let start = pair[0];
        let end = pair[1];
        if start >= end {
            continue;
        }
        let format = if omitted_ranges.iter().any(|(a, b)| start < *b && *a < end) {
            omitted.clone()
        } else if timing_ranges.iter().any(|(a, b)| start < *b && *a < end) {
            timing.clone()
        } else {
            normal.clone()
        };
        layout_job.append(&text[start..end], 0.0, format);
    }

    ui.ctx().fonts_mut(|fonts| fonts.layout_job(layout_job))
}

fn secs_to_hms(seconds: f64) -> String {
    let mut total = seconds.max(0.0) as u64;
    let h = total / 3600;
    total %= 3600;
    let m = total / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

fn playback_total_time(state: &PlaybackState) -> f64 {
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(inner) = shared.try_lock() {
            if state.output_sample_rate > 0 {
                return inner.total_frames as f64 / state.output_sample_rate as f64;
            }
        }
    }
    0.0
}

fn playback_is_playing(state: &PlaybackState) -> bool {
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(inner) = shared.try_lock() {
            return inner.playing;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_braced_key_value_output() {
        let raw = "{location: Kabul; location: University of Oxford; number: 0750000000;}";
        let parsed = parse_anonymise_matches(raw).expect("should parse key:value output");
        assert!(parsed.len() >= 3);
        assert!(parsed.iter().any(|m| {
            m.kind == AnonymiseEntityKind::Location && m.value.eq_ignore_ascii_case("Kabul")
        }));
        assert!(parsed.iter().any(|m| {
            m.kind == AnonymiseEntityKind::Number && m.value.eq_ignore_ascii_case("0750000000")
        }));
    }

    #[test]
    fn merges_multiple_json_objects() {
        let raw = r#"{"location":["Kabul"]} {"name":["Anastasia"]} {"number":["007000000"]}"#;
        let parsed = parse_anonymise_matches(raw).expect("should merge multiple json objects");
        assert!(parsed
            .iter()
            .any(|m| m.kind == AnonymiseEntityKind::Location && m.value == "Kabul"));
        assert!(parsed
            .iter()
            .any(|m| m.kind == AnonymiseEntityKind::Name && m.value == "Anastasia"));
        assert!(parsed
            .iter()
            .any(|m| m.kind == AnonymiseEntityKind::Number && m.value == "007000000"));
    }

    #[test]
    fn parses_legacy_wrapper_key() {
        let raw = r#"{"anonymisation_extractions":"location: Kabul; name: Carol;"}"#;
        let parsed = parse_anonymise_matches(raw).expect("should parse wrapper payload");
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn prompt_contains_selected_examples() {
        let prompt = build_anonymise_prompt(
            "Anastasia called from Kabul.",
            &[AnonymiseEntityKind::Name, AnonymiseEntityKind::Location],
        );
        assert!(prompt.contains("name: <person_name_from_transcript>;"));
        assert!(prompt.contains("location: <location_from_transcript>;"));
        assert!(prompt.contains("`{key: value; key: value;}`"));
    }

    #[test]
    fn sanitize_candidate_strips_terminal_sentence_punctuation() {
        let cleaned = sanitize_candidate_value(r#" "Kabul...?!" "#);
        assert_eq!(cleaned, "Kabul");
    }

    #[test]
    fn fuzzy_replaces_when_surface_form_differs_but_tokens_match() {
        let text = "She lived in Afghanistan for several years.";
        let matches = vec![AnonymiseMatch {
            kind: AnonymiseEntityKind::Location,
            value: "(Afghanistan)".to_string(),
        }];
        let (rewritten, exact, fuzzy) = apply_anonymise_to_text(text, &matches);
        assert_eq!(exact, 0);
        assert_eq!(fuzzy, 1);
        assert!(rewritten.contains("[location omitted]"));
    }

    #[test]
    fn near_exact_handles_single_character_drift() {
        let text = "She lived in Afghanistann for several years.";
        let matches = vec![AnonymiseMatch {
            kind: AnonymiseEntityKind::Location,
            value: "Afghanistan".to_string(),
        }];
        let (rewritten, exact, fuzzy) = apply_anonymise_to_text(text, &matches);
        assert_eq!(exact, 0);
        assert_eq!(fuzzy, 1);
        assert!(rewritten.contains("[location omitted]"));
    }

    #[test]
    fn real_model_style_kv_output_parses_and_replaces() {
        let text = "President Bush met President Karzai in Afghanistan at the White House.";
        let raw = "{name: President Bush; name: President Karzai; location: Afghanistan; organisation: White House;}";
        let parsed = parse_anonymise_matches(raw).expect("should parse real model style output");
        assert_eq!(parsed.len(), 4);
        let (rewritten, exact, fuzzy) = apply_anonymise_to_text(text, &parsed);
        assert!(exact > 0 || fuzzy > 0);
        assert!(rewritten.contains("[name omitted]"));
        assert!(rewritten.contains("[location omitted]"));
        assert!(rewritten.contains("[organisation omitted]"));
    }

    #[test]
    fn selected_gpu_index_defaults_to_cpu_when_not_set() {
        let settings = AppSettings::default();
        assert_eq!(selected_gpu_index_from_settings(&settings), None);
    }

    #[test]
    fn selected_gpu_index_uses_explicit_device_value() {
        let mut settings = AppSettings::default();
        settings.devices = "2".to_string();
        assert_eq!(selected_gpu_index_from_settings(&settings), Some(2));
    }

    #[test]
    fn selected_gpu_index_uses_legacy_nonzero_main_gpu() {
        let mut settings = AppSettings::default();
        settings.main_gpu = 1;
        settings.whisper_no_gpu = false;
        settings.devices.clear();
        assert_eq!(selected_gpu_index_from_settings(&settings), Some(1));
    }

    #[test]
    fn shared_bridge_params_forces_cpu_when_no_gpu_selected() {
        let mut settings = AppSettings::default();
        settings.devices.clear();
        settings.main_gpu = 0;
        settings.whisper_no_gpu = false;
        let shared = shared_bridge_params(&settings);
        assert_eq!(shared.gpu, None);
        assert_eq!(shared.devices.as_deref(), Some("none"));
        assert_eq!(shared.n_gpu_layers, Some(0));
        assert_eq!(shared.main_gpu, Some(-1));
    }

    #[test]
    fn shared_bridge_params_keeps_selected_gpu() {
        let mut settings = AppSettings::default();
        settings.devices = "2".to_string();
        let shared = shared_bridge_params(&settings);
        assert_eq!(shared.gpu, Some(2));
        assert_eq!(shared.devices, None);
        assert_eq!(shared.n_gpu_layers, None);
        assert_eq!(shared.main_gpu, None);
    }

    #[test]
    fn shared_bridge_params_preserves_explicit_thread_settings() {
        let mut settings = AppSettings::default();
        settings.n_threads = 6;
        settings.n_threads_batch = 3;
        let shared = shared_bridge_params(&settings);
        assert_eq!(shared.n_threads, Some(6));
        assert_eq!(shared.n_threads_batch, Some(3));
    }

    #[test]
    fn resolve_audio_device_index_keeps_cpu_as_default() {
        let devices = vec![
            AudioDeviceOption {
                label: "CPU (no GPU)".to_string(),
                devices_value: "none".to_string(),
                main_gpu: 0,
                is_gpu: false,
                detail_line: "CPU".to_string(),
            },
            AudioDeviceOption {
                label: "GPU #0".to_string(),
                devices_value: "0".to_string(),
                main_gpu: 0,
                is_gpu: true,
                detail_line: "GPU".to_string(),
            },
        ];
        let settings = AppSettings::default();
        #[cfg(target_os = "macos")]
        assert_eq!(resolve_audio_device_index(&devices, &settings), 1);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(resolve_audio_device_index(&devices, &settings), 0);
    }

    #[test]
    fn resolve_runtime_backend_index_prefers_first_when_missing() {
        let options = vec!["vulkan".to_string(), "cuda".to_string()];
        assert_eq!(resolve_runtime_backend_index(&options, ""), 0);
        assert_eq!(resolve_runtime_backend_index(&options, "unknown"), 0);
    }

    #[test]
    fn resolve_runtime_backend_index_matches_case_insensitive() {
        let options = vec!["vulkan".to_string(), "cuda".to_string()];
        assert_eq!(resolve_runtime_backend_index(&options, "CUDA"), 1);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn detects_installed_windows_runtime_backend_cuda_from_dll_markers() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("to-runtime-backend-test-{stamp}"));
        let cuda_dir = root.join("vendor").join("cuda");
        fs::create_dir_all(&cuda_dir).expect("create cuda dir");
        fs::write(cuda_dir.join("cublas64_13.dll"), b"").expect("write marker dll");

        let detected = detect_installed_windows_runtime_backend(&root);
        assert_eq!(detected.as_deref(), Some("cuda"));

        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn detects_installed_windows_runtime_backend_vulkan_without_cuda_dlls() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("to-runtime-backend-test-{stamp}"));
        fs::create_dir_all(&root).expect("create runtime dir");
        fs::write(root.join(bridge_library_file_name()), b"").expect("write bridge marker");

        let detected = detect_installed_windows_runtime_backend(&root);
        assert_eq!(detected.as_deref(), Some("vulkan"));

        let _ = fs::remove_dir_all(&root);
    }
}

fn main() {
    configure_ui_startup();

    let paths = match app_paths().and_then(|p| {
        ensure_dirs(&p)?;
        Ok(p)
    }) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to init settings paths: {e}");
            return;
        }
    };
    install_panic_hook(paths.clone());

    let viewport = {
        let mut v = egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_app_id("transcribe-offline");
        if let Some(icon_data) = decode_app_icon_data() {
            v = v.with_icon(icon_data);
        }
        v
    };
    let options = NativeOptions {
        viewport,
        ..NativeOptions::default()
    };
    let _ = eframe::run_native(
        "Transcribe Offline",
        options,
        Box::new(move |_| Ok(Box::new(UiApp::new(paths)))),
    );
}

include!("main_backend_tail.rs");
