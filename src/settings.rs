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
    pub diarization_models_dir: PathBuf,
    pub settings_yaml: PathBuf,
    pub model_links_yaml: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLinks {
    pub chat_model: String,
    pub whisper_model: String,
    pub diarization_models_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub runtime_dir: String,
    pub audio_file: String,
    pub output_dir: String,
    pub whisper_model: String,
    pub diarization_models_dir: String,
    pub mode: String,
    pub custom_mode: String,
    pub subtitle_custom_mode: String,
    pub speech_custom_mode: String,
    pub speaker_count: String,
    pub diarization_enabled: bool,
    pub whisper_no_gpu: bool,
    pub ffmpeg_convert: bool,
    pub whisper_word_time_offset_sec: String,
    pub chat_model: String,
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
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            runtime_dir: default_runtime_dir().display().to_string(),
            audio_file: String::new(),
            output_dir: String::new(),
            whisper_model: String::new(),
            diarization_models_dir: String::new(),
            mode: "transcript".to_string(),
            custom_mode: "auto".to_string(),
            subtitle_custom_mode: "auto".to_string(),
            speech_custom_mode: "auto".to_string(),
            speaker_count: "auto".to_string(),
            diarization_enabled: true,
            whisper_no_gpu: false,
            ffmpeg_convert: true,
            whisper_word_time_offset_sec: "0.73".to_string(),
            chat_model: String::new(),
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
        return dirs.data_dir().join("OpenResearchTools").join("Models");
    }

    if let Ok(app_data) = std::env::var("APPDATA") {
        return PathBuf::from(app_data)
            .join("OpenResearchTools")
            .join("Models");
    }

    PathBuf::from("Models")
}

pub const DEFAULT_WHISPER_MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

pub fn default_whisper_model_path(paths: &AppPaths) -> PathBuf {
    paths.whisper_models_dir.join(DEFAULT_WHISPER_MODEL_FILE)
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
    let whisper_models_dir = models_dir.join("Whisper");
    let diarization_models_dir = models_dir.join("Diarization");
    Ok(AppPaths {
        settings_yaml: config_dir.join("settings.yaml"),
        model_links_yaml: config_dir.join("model-links.yaml"),
        config_dir,
        data_dir,
        models_dir,
        whisper_models_dir,
        diarization_models_dir,
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
    fs::create_dir_all(&paths.diarization_models_dir).with_context(|| {
        format!(
            "failed to create diarization models directory '{}'",
            paths.diarization_models_dir.display()
        )
    })?;
    Ok(())
}

pub fn load_settings(paths: &AppPaths) -> Result<AppSettings> {
    if !paths.settings_yaml.exists() {
        let mut defaults = AppSettings::default();
        defaults.whisper_model = default_whisper_model_path(paths).display().to_string();
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
    parsed.runtime_dir = normalize_runtime_dir(parsed.runtime_dir);
    if parsed.runtime_dir.trim().is_empty() {
        parsed.runtime_dir = default_runtime_dir().display().to_string();
    }
    if parsed.whisper_model.trim().is_empty() {
        parsed.whisper_model = default_whisper_model_path(paths).display().to_string();
    }
    if parsed.diarization_models_dir.trim().is_empty() {
        parsed.diarization_models_dir = default_diarization_models_dir(paths).display().to_string();
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
            diarization_models_dir: self.diarization_models_dir.clone(),
        }
    }
}
