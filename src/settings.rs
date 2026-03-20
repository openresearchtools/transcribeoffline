use anyhow::{Context, Result};
use directories::{BaseDirs, ProjectDirs};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub whisper_models_dir: PathBuf,
    pub whisper_large_v3_models_dir: PathBuf,
    pub live_models_dir: PathBuf,
    pub chat_models_dir: PathBuf,
    pub diarization_models_dir: PathBuf,
    pub live_sessions_dir: PathBuf,
    pub settings_yaml: PathBuf,
    pub model_links_yaml: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLinks {
    pub chat_model: String,
    pub whisper_model: String,
    pub live_transcription_model: String,
    pub diarization_models_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub runtime_dir: String,
    pub audio_file: String,
    pub output_dir: String,
    pub whisper_model: String,
    pub live_transcription_model: String,
    pub live_input_device: String,
    pub diarization_models_dir: String,
    pub mode: String,
    pub custom_mode: String,
    pub subtitle_custom_mode: String,
    pub speech_custom_mode: String,
    pub diarization_enabled: bool,
    pub whisper_no_gpu: bool,
    pub ffmpeg_convert: bool,
    pub whisper_word_time_offset_sec: String,
    pub chat_model: String,
    pub chat_allow_thinking: bool,
    pub chat_prompt: String,
    pub chat_context_file: String,
    pub devices: String,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ui_font_size_px: f32,
    pub split_view: bool,
    pub edit_cursor_resync_delay_sec: f32,
    pub edit_autosave_delay_sec: f32,
    pub seek_step_sec: f32,
    pub seek_step_growth_sec: f32,
    pub playback_toggle_offset_sec: f32,
    pub runtime_download_backend: String,
    #[serde(default = "default_live_webrtc_enabled")]
    pub live_webrtc_enabled: bool,
    pub live_diarization_enabled: bool,
}

fn default_live_webrtc_enabled() -> bool {
    true
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            runtime_dir: default_runtime_dir().display().to_string(),
            audio_file: String::new(),
            output_dir: String::new(),
            whisper_model: String::new(),
            live_transcription_model: String::new(),
            live_input_device: String::new(),
            diarization_models_dir: String::new(),
            mode: "transcript".to_string(),
            custom_mode: "auto".to_string(),
            subtitle_custom_mode: "auto".to_string(),
            speech_custom_mode: "auto".to_string(),
            diarization_enabled: true,
            whisper_no_gpu: false,
            ffmpeg_convert: true,
            whisper_word_time_offset_sec: "0.73".to_string(),
            chat_model: String::new(),
            chat_allow_thinking: false,
            chat_prompt: String::new(),
            chat_context_file: String::new(),
            devices: String::new(),
            n_ctx: 32768,
            n_batch: 2048,
            n_ubatch: 2048,
            n_parallel: 1,
            n_gpu_layers: -1,
            main_gpu: 0,
            n_threads: 0,
            n_threads_batch: 0,
            ui_font_size_px: 12.0,
            split_view: true,
            edit_cursor_resync_delay_sec: 4.0,
            edit_autosave_delay_sec: 4.0,
            seek_step_sec: 5.0,
            seek_step_growth_sec: 2.0,
            playback_toggle_offset_sec: 0.0,
            runtime_download_backend: "vulkan".to_string(),
            live_webrtc_enabled: default_live_webrtc_enabled(),
            live_diarization_enabled: true,
        }
    }
}

pub fn default_runtime_dir() -> PathBuf {
    if let Some(dirs) = BaseDirs::new() {
        return dirs.data_dir().join("OpenResearchTools").join("engine");
    }

    if let Ok(app_data) = std::env::var("APPDATA") {
        return PathBuf::from(app_data)
            .join("OpenResearchTools")
            .join("engine");
    }

    PathBuf::from("engine")
}

pub fn normalize_runtime_dir_alias(raw: &str) -> String {
    let mut out = raw.trim().to_string();
    if out.is_empty() {
        return out;
    }

    if out.eq_ignore_ascii_case("engine-runtime")
        || out.eq_ignore_ascii_case("engine-vulkan")
        || out.eq_ignore_ascii_case("engine-vulcan")
        || out.eq_ignore_ascii_case("engine-cuda")
    {
        return "engine".to_string();
    }

    for (needle, replacement) in [
        ("\\engine-runtime\\", "\\engine\\"),
        ("/engine-runtime/", "/engine/"),
        ("\\engine-runtime", "\\engine"),
        ("/engine-runtime", "/engine"),
        ("\\engine-vulkan\\", "\\engine\\"),
        ("/engine-vulkan/", "/engine/"),
        ("\\engine-vulkan", "\\engine"),
        ("/engine-vulkan", "/engine"),
        ("\\engine-vulcan\\", "\\engine\\"),
        ("/engine-vulcan/", "/engine/"),
        ("\\engine-vulcan", "\\engine"),
        ("/engine-vulcan", "/engine"),
        ("\\engine-cuda\\", "\\engine\\"),
        ("/engine-cuda/", "/engine/"),
        ("\\engine-cuda", "\\engine"),
        ("/engine-cuda", "/engine"),
    ] {
        let needle_lc = needle.to_ascii_lowercase();
        loop {
            let out_lc = out.to_ascii_lowercase();
            let Some(pos) = out_lc.find(&needle_lc) else {
                break;
            };
            let end = pos + needle.len();
            out.replace_range(pos..end, replacement);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::normalize_runtime_dir_alias;

    #[test]
    fn normalizes_engine_runtime_suffix_windows() {
        let got = normalize_runtime_dir_alias(
            r"C:\Users\user\AppData\Roaming\OpenResearchTools\engine-runtime",
        );
        assert_eq!(
            got,
            r"C:\Users\user\AppData\Roaming\OpenResearchTools\engine"
        );
    }

    #[test]
    fn normalizes_engine_runtime_suffix_unix() {
        let got =
            normalize_runtime_dir_alias("/home/user/.local/share/OpenResearchTools/engine-runtime");
        assert_eq!(got, "/home/user/.local/share/OpenResearchTools/engine");
    }

    #[test]
    fn normalizes_engine_backend_suffixes() {
        assert_eq!(normalize_runtime_dir_alias("engine-vulkan"), "engine");
        assert_eq!(normalize_runtime_dir_alias("engine-vulcan"), "engine");
        assert_eq!(normalize_runtime_dir_alias("engine-cuda"), "engine");
    }
}

pub fn default_models_root_dir() -> PathBuf {
    if let Some(dirs) = BaseDirs::new() {
        return dirs.data_dir().join("OpenResearchTools").join("models");
    }

    if let Ok(app_data) = std::env::var("APPDATA") {
        return PathBuf::from(app_data)
            .join("OpenResearchTools")
            .join("models");
    }

    PathBuf::from("models")
}

pub const DEFAULT_WHISPER_MODEL_FILE: &str = "whisper-large-v3-turbo-GGML.bin";
pub const DEFAULT_LIVE_TRANSCRIPTION_MODEL_FILE: &str = "voxtral-mini-4b-realtime-q4_0.gguf";
pub const WHISPER_TURBO_REPO_DIR_NAME: &str = "openresearchtools__whisper-large-v3-turbo-GGML";
pub const WHISPER_LARGE_V3_REPO_DIR_NAME: &str = "openresearchtools__whisper-large-v3-GGML";
pub const LIVE_TRANSCRIPTION_REPO_DIR_NAME: &str =
    "openresearchtools__Voxtral-Mini-4B-Realtime-2602";
pub const CHAT_REPO_DIR_NAME: &str = "openresearchtools__Qwen3.5-9B-GGUF";
pub const DIARIZATION_REPO_DIR_NAME: &str =
    "openresearchtools__diar_streaming_sortformer_4spk-v2.1-gguf";

fn path_file_name_eq(path: &Path, expected: &str) -> bool {
    path.file_name()
        .map(|value| value.to_string_lossy().eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

fn normalize_repo_managed_file_path(raw: &str, repo_dir: &Path, legacy_dir_name: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let path = PathBuf::from(trimmed);
    let Some(file_name) = path.file_name() else {
        return trimmed.to_string();
    };
    let repo_path = repo_dir.join(file_name);

    if repo_path.exists() {
        return repo_path.display().to_string();
    }

    if path
        .parent()
        .map(|parent| path_file_name_eq(parent, legacy_dir_name))
        .unwrap_or(false)
    {
        return repo_path.display().to_string();
    }

    trimmed.to_string()
}

fn normalize_repo_managed_dir_path(raw: &str, repo_dir: &Path, legacy_dir_name: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let path = PathBuf::from(trimmed);
    if path == repo_dir || path_file_name_eq(&path, legacy_dir_name) {
        return repo_dir.display().to_string();
    }

    trimmed.to_string()
}

pub fn whisper_model_repo_dir(paths: &AppPaths, file_name: &str) -> PathBuf {
    if file_name.eq_ignore_ascii_case("whisper-large-v3-GGML.bin") {
        return paths.whisper_large_v3_models_dir.clone();
    }
    paths.whisper_models_dir.clone()
}

fn normalize_whisper_model_path(raw: &str, paths: &AppPaths) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let path = PathBuf::from(trimmed);
    let Some(file_name) = path.file_name() else {
        return trimmed.to_string();
    };
    let file_name = file_name.to_string_lossy();
    let repo_dir = whisper_model_repo_dir(paths, &file_name);
    let repo_path = repo_dir.join(file_name.as_ref());

    if repo_path.exists() {
        return repo_path.display().to_string();
    }

    let parent_matches = path.parent().map(|parent| {
        path_file_name_eq(parent, "Whisper")
            || path_file_name_eq(parent, "ggerganov__whisper.cpp")
            || path_file_name_eq(parent, WHISPER_TURBO_REPO_DIR_NAME)
            || path_file_name_eq(parent, WHISPER_LARGE_V3_REPO_DIR_NAME)
    });
    if parent_matches.unwrap_or(false) {
        return repo_path.display().to_string();
    }

    trimmed.to_string()
}

pub fn default_whisper_model_path(paths: &AppPaths) -> PathBuf {
    whisper_model_repo_dir(paths, DEFAULT_WHISPER_MODEL_FILE).join(DEFAULT_WHISPER_MODEL_FILE)
}

pub fn default_live_transcription_model_path(paths: &AppPaths) -> PathBuf {
    paths
        .live_models_dir
        .join(DEFAULT_LIVE_TRANSCRIPTION_MODEL_FILE)
}

pub fn default_diarization_models_dir(paths: &AppPaths) -> PathBuf {
    paths.diarization_models_dir.clone()
}

pub fn app_paths() -> Result<AppPaths> {
    let dirs = ProjectDirs::from("com", "OpenResearchTools", "TranscribeOffline")
        .context("failed to resolve user-space app directories")?;
    let config_dir = dirs.config_dir().to_path_buf();
    let data_dir = dirs.data_dir().to_path_buf();
    let models_dir = default_models_root_dir();
    let whisper_models_dir = models_dir.join(WHISPER_TURBO_REPO_DIR_NAME);
    let whisper_large_v3_models_dir = models_dir.join(WHISPER_LARGE_V3_REPO_DIR_NAME);
    let live_models_dir = models_dir.join(LIVE_TRANSCRIPTION_REPO_DIR_NAME);
    let chat_models_dir = models_dir.join(CHAT_REPO_DIR_NAME);
    let diarization_models_dir = models_dir.join(DIARIZATION_REPO_DIR_NAME);
    let live_sessions_dir = data_dir.join("live-sessions");
    Ok(AppPaths {
        settings_yaml: config_dir.join("settings.yaml"),
        model_links_yaml: config_dir.join("model-links.yaml"),
        config_dir,
        data_dir,
        models_dir,
        whisper_models_dir,
        whisper_large_v3_models_dir,
        live_models_dir,
        chat_models_dir,
        diarization_models_dir,
        live_sessions_dir,
    })
}

pub fn ensure_dirs(paths: &AppPaths) -> Result<()> {
    fs::create_dir_all(&paths.config_dir).with_context(|| {
        format!(
            "failed to create config directory '{}'",
            paths.config_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.data_dir).with_context(|| {
        format!(
            "failed to create data directory '{}'",
            paths.data_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.models_dir).with_context(|| {
        format!(
            "failed to create models directory '{}'",
            paths.models_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.whisper_models_dir).with_context(|| {
        format!(
            "failed to create whisper models directory '{}'",
            paths.whisper_models_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.whisper_large_v3_models_dir).with_context(|| {
        format!(
            "failed to create whisper models directory '{}'",
            paths.whisper_large_v3_models_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.live_models_dir).with_context(|| {
        format!(
            "failed to create realtime models directory '{}'",
            paths.live_models_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.chat_models_dir).with_context(|| {
        format!(
            "failed to create chat models directory '{}'",
            paths.chat_models_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.diarization_models_dir).with_context(|| {
        format!(
            "failed to create diarization models directory '{}'",
            paths.diarization_models_dir.display()
        )
    })?;
    fs::create_dir_all(&paths.live_sessions_dir).with_context(|| {
        format!(
            "failed to create live sessions directory '{}'",
            paths.live_sessions_dir.display()
        )
    })?;
    Ok(())
}

pub fn load_settings(paths: &AppPaths) -> Result<AppSettings> {
    if !paths.settings_yaml.exists() {
        let mut defaults = AppSettings::default();
        defaults.whisper_model = default_whisper_model_path(paths).display().to_string();
        defaults.live_transcription_model = default_live_transcription_model_path(paths)
            .display()
            .to_string();
        defaults.diarization_models_dir =
            default_diarization_models_dir(paths).display().to_string();
        save_settings(paths, &defaults)?;
        save_model_links(paths, &defaults.to_model_links())?;
        return Ok(defaults);
    }
    let raw = fs::read_to_string(&paths.settings_yaml)
        .with_context(|| format!("failed to read '{}'", paths.settings_yaml.display()))?;
    let migrated_raw = raw
        .replace("engine-runtime", "engine")
        .replace("engine-vulkan", "engine")
        .replace("engine-vulcan", "engine")
        .replace("engine-cuda", "engine");
    let effective_raw = if migrated_raw != raw {
        let _ = fs::write(&paths.settings_yaml, &migrated_raw);
        migrated_raw
    } else {
        raw
    };
    let mut parsed: AppSettings = serde_yaml::from_str(&effective_raw)
        .with_context(|| format!("failed to parse '{}'", paths.settings_yaml.display()))?;
    let original_runtime_dir = parsed.runtime_dir.clone();
    let original_whisper_model = parsed.whisper_model.clone();
    let original_live_model = parsed.live_transcription_model.clone();
    let original_diarization_dir = parsed.diarization_models_dir.clone();
    let original_live_webrtc_enabled = parsed.live_webrtc_enabled;
    parsed.runtime_dir = normalize_runtime_dir(parsed.runtime_dir);
    parsed.whisper_model = normalize_whisper_model_path(&parsed.whisper_model, paths);
    parsed.live_transcription_model = normalize_repo_managed_file_path(
        &parsed.live_transcription_model,
        &paths.live_models_dir,
        "Realtime",
    );
    parsed.diarization_models_dir = normalize_repo_managed_dir_path(
        &parsed.diarization_models_dir,
        &paths.diarization_models_dir,
        "Diarization",
    );
    if parsed.runtime_dir.trim().is_empty() {
        parsed.runtime_dir = default_runtime_dir().display().to_string();
    }
    if parsed.whisper_model.trim().is_empty() {
        parsed.whisper_model = default_whisper_model_path(paths).display().to_string();
    }
    if parsed.live_transcription_model.trim().is_empty() {
        parsed.live_transcription_model = default_live_transcription_model_path(paths)
            .display()
            .to_string();
    }
    if parsed.diarization_models_dir.trim().is_empty() {
        parsed.diarization_models_dir = default_diarization_models_dir(paths).display().to_string();
    }
    parsed.live_webrtc_enabled = true;
    if parsed.runtime_dir != original_runtime_dir
        || parsed.whisper_model != original_whisper_model
        || parsed.live_transcription_model != original_live_model
        || parsed.diarization_models_dir != original_diarization_dir
        || parsed.live_webrtc_enabled != original_live_webrtc_enabled
    {
        save_settings(paths, &parsed)?;
        save_model_links(paths, &parsed.to_model_links())?;
    }
    Ok(parsed)
}

fn normalize_runtime_dir(raw: String) -> String {
    let normalized = normalize_runtime_dir_alias(&raw);
    if is_disallowed_runtime_path(&normalized) {
        return String::new();
    }

    normalized
}

fn is_disallowed_runtime_path(raw: &str) -> bool {
    let path = Path::new(raw);
    if raw.trim().is_empty() {
        return false;
    }

    path.components().any(|c| {
        c.as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case("transcribeoffline")
    })
}

pub fn save_settings(paths: &AppPaths, settings: &AppSettings) -> Result<()> {
    let yaml = serde_yaml::to_string(settings).context("failed to serialize settings yaml")?;
    fs::write(&paths.settings_yaml, yaml)
        .with_context(|| format!("failed to write '{}'", paths.settings_yaml.display()))?;
    Ok(())
}

pub fn save_model_links(paths: &AppPaths, links: &ModelLinks) -> Result<()> {
    let yaml = serde_yaml::to_string(links).context("failed to serialize model-links yaml")?;
    fs::write(&paths.model_links_yaml, yaml)
        .with_context(|| format!("failed to write '{}'", paths.model_links_yaml.display()))?;
    Ok(())
}

impl AppSettings {
    pub fn to_model_links(&self) -> ModelLinks {
        ModelLinks {
            chat_model: self.chat_model.clone(),
            whisper_model: self.whisper_model.clone(),
            live_transcription_model: self.live_transcription_model.clone(),
            diarization_models_dir: self.diarization_models_dir.clone(),
        }
    }
}
