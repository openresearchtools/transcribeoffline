fn run_transcription_with_progress<F>(
    settings: AppSettings,
    mut progress: F,
) -> Result<TranscriptionResult>
where
    F: FnMut(String),
{
    let started_at = Instant::now();
    progress("validating input".to_string());
    if settings.audio_file.trim().is_empty() {
        bail!("audio file is required");
    }
    if settings.whisper_model.trim().is_empty() {
        bail!("whisper model path is required");
    }

    let audio_path = PathBuf::from(settings.audio_file.trim());
    if !audio_path.exists() {
        bail!("audio file not found: '{}'", audio_path.display());
    }

    let prepare_started = Instant::now();
    // Parity rule: keep generated transcript files in the same folder as source audio.
    let output_dir = audio_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create output dir '{}'", output_dir.display()))?;

    let mode_norm = settings.mode.trim().to_ascii_lowercase();
    let mut custom_value = settings.custom_mode.trim().to_string();
    if mode_norm == "subtitle" {
        let subtitle_custom = settings.subtitle_custom_mode.trim();
        if !subtitle_custom.is_empty() {
            custom_value = subtitle_custom.to_string();
        }
        if !is_default_or_positive_float(&custom_value) {
            bail!("subtitle custom must be 'default'/'auto' or a positive number of seconds");
        }
    } else if mode_norm == "speech" {
        let speech_custom = settings.speech_custom_mode.trim();
        if !speech_custom.is_empty() {
            custom_value = speech_custom.to_string();
        }
        if !is_default_or_positive_float(&custom_value) {
            bail!("speech custom must be 'default'/'auto' or a positive number of seconds");
        }
    }
    if custom_value.is_empty() {
        custom_value = "default".to_string();
    }
    let custom_value_for_result = custom_value.clone();

    let mut metadata = json!({
        "mode": settings.mode,
        "custom": custom_value,
        "output_dir": output_dir.display().to_string(),
        "audio_source_path": audio_path.display().to_string(),
        "whisper_model": settings.whisper_model,
    });
    let selected_gpu = selected_gpu_index_from_settings(&settings);

    let diarization_enabled = mode_norm == "transcript";
    if diarization_enabled {
        if settings.diarization_models_dir.trim().is_empty() {
            bail!("diarization is enabled but diarization models dir is empty");
        }
        let diarization_model_path =
            PathBuf::from(settings.diarization_models_dir.trim()).join(SORTFORMER_MODEL_FILE);
        if !diarization_model_path.exists() {
            bail!(
                "diarization model not found: '{}'",
                diarization_model_path.display()
            );
        }
        metadata["diarization_model_path"] = json!(diarization_model_path.display().to_string());
        metadata["diarization_backend"] = json!("sortformer");
        metadata["diarization_feed_ms"] = json!(10_800_001.0);
    }
    if let Ok(v) = settings.whisper_word_time_offset_sec.trim().parse::<f64>() {
        metadata["whisper_word_time_offset_sec"] = json!(v);
    }
    metadata["transport"] = json!("audio_raw_bytes");
    let timing_prepare_ms = prepare_started.elapsed().as_millis();

    progress("loading runtime + bridge".to_string());
    let runtime_dir = resolve_runtime_dir(Path::new(settings.runtime_dir.trim()));
    configure_runtime_dll_search(&runtime_dir);
    let api = BridgeApi::load(&runtime_dir)?;
    if let Some(gpu_index) = selected_gpu {
        if !bridge_has_device_index(&api, gpu_index) {
            bail!(
                "selected GPU index {} is not available in runtime device list; choose CPU mode or re-select a valid GPU in Runtime settings",
                gpu_index
            );
        }
        // Force all audio modes to use selected dropdown GPU in runtime metadata.
        metadata["gpu"] = json!(gpu_index);
        metadata["whisper_gpu_device"] = json!(gpu_index);
        if diarization_enabled {
            if let Some(device_name) = resolve_bridge_device_name_by_index(&api, gpu_index) {
                // Keep diarization on same selected runtime device.
                metadata["diarization_device"] = json!(device_name);
            }
        }
    } else {
        // No explicit GPU selected: force CPU mode to avoid runtime auto-selecting an invalid backend.
        metadata["whisper_no_gpu"] = json!(true);
    }
    progress("reading audio bytes".to_string());
    let read_audio_started = Instant::now();
    let audio_bytes = fs::read(&audio_path)
        .with_context(|| format!("failed to read audio '{}'", audio_path.display()))?;
    let audio_bytes_len = audio_bytes.len();
    let timing_read_audio_ms = read_audio_started.elapsed().as_millis();
    let audio_format = infer_audio_format(&audio_path);
    let shared = shared_bridge_params(&settings);
    metadata["transport"] = json!("audio_raw_bytes");
    if diarization_enabled {
        progress("bridge pipeline stage: whisper transcription (starting)".to_string());
        progress("bridge pipeline stage: native diarization (starting)".to_string());
        progress("running whisper transcription + diarization".to_string());
    } else {
        progress("bridge pipeline stage: whisper transcription (starting)".to_string());
        progress("running whisper transcription".to_string());
    }
    let bridge_started = Instant::now();
    let response_json = api.run_audio_raw(
        &shared,
        &AudioRunParams {
            audio_bytes,
            audio_format,
            metadata_json: metadata,
            ffmpeg_convert: true,
        },
    )?;
    progress("bridge pipeline stage: complete".to_string());
    let timing_bridge_ms = bridge_started.elapsed().as_millis();

    progress("loading generated output text".to_string());
    let output_path = response_json
        .pointer("/output/path")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(|| fallback_output_path(&audio_path, &settings.mode, &output_dir));

    let read_output_started = Instant::now();
    let output_text = fs::read_to_string(&output_path)
        .with_context(|| format!("failed to read output '{}'", output_path.display()))?;
    let timing_read_output_ms = read_output_started.elapsed().as_millis();
    let preprocess_note =
        "Sending raw bytes; bridge-side FFmpeg conversion enabled (vendored runtime)."
            .to_string();
    let timing_total_ms = started_at.elapsed().as_millis();

    Ok(TranscriptionResult {
        _response_json: response_json,
        output_path,
        output_text,
        preprocess_note,
        mode: settings.mode,
        custom_value: custom_value_for_result,
        whisper_model: settings.whisper_model,
        diarization_enabled,
        timing_prepare_ms,
        timing_read_audio_ms,
        timing_bridge_ms,
        timing_read_output_ms,
        timing_total_ms,
        audio_bytes_len,
    })
}

fn run_chat(settings: AppSettings) -> Result<String> {
    if settings.chat_model.trim().is_empty() {
        bail!("chat model path is required");
    }
    if settings.chat_prompt.trim().is_empty() {
        bail!("chat prompt is required");
    }

    let mut prompt = settings.chat_prompt.clone();
    if !settings.chat_context_file.trim().is_empty() {
        let context_path = PathBuf::from(settings.chat_context_file.trim());
        let context = fs::read_to_string(&context_path)
            .with_context(|| format!("failed to read context file '{}'", context_path.display()))?;
        prompt = format!("{prompt}\n\nContext markdown:\n\n{context}");
    }

    let runtime_dir = resolve_runtime_dir(Path::new(settings.runtime_dir.trim()));
    configure_runtime_dll_search(&runtime_dir);
    let api = BridgeApi::load(&runtime_dir)?;
    if let Some(gpu_index) = selected_gpu_index_from_settings(&settings) {
        if !bridge_has_device_index(&api, gpu_index) {
            bail!(
                "selected GPU index {} is not available in runtime device list; choose CPU mode or re-select a valid GPU in Runtime settings",
                gpu_index
            );
        }
    }
    let shared = shared_bridge_params(&settings);
    let (reasoning, reasoning_budget, reasoning_format) = if settings.chat_allow_thinking {
        (Some("on".to_string()), -1, Some("deepseek".to_string()))
    } else {
        (Some("off".to_string()), REASONING_BUDGET_UNSET, None)
    };
    api.run_chat(
        &shared,
        &ChatRunParams {
            model_path: settings.chat_model,
            prompt,
            n_predict: 10_000,
            reasoning,
            reasoning_budget,
            reasoning_format,
        },
    )
}

fn shared_bridge_params(settings: &AppSettings) -> SharedBridgeParams {
    let selected_gpu = selected_gpu_index_from_settings(settings);
    // CPU mode is the default whenever no explicit GPU index is selected.
    let force_cpu = selected_gpu.is_none();
    let n_gpu_layers = if force_cpu { Some(0) } else { None };
    let main_gpu = if force_cpu { Some(-1) } else { None };
    let n_threads = positive_optional(settings.n_threads).or_else(default_runtime_threads);
    SharedBridgeParams {
        gpu: selected_gpu,
        devices: if force_cpu {
            Some("none".to_string())
        } else {
            None
        },
        tensor_split: None,
        split_mode: -1,
        n_ctx: settings.n_ctx,
        n_batch: settings.n_batch,
        n_ubatch: settings.n_ubatch,
        n_parallel: settings.n_parallel,
        n_threads,
        n_threads_batch: positive_optional(settings.n_threads_batch),
        n_gpu_layers,
        main_gpu,
    }
}

fn whisper_label_from_path(path: &Path) -> Option<&'static str> {
    let file = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    WHISPER_MODELS
        .iter()
        .find(|m| m.file_name.eq_ignore_ascii_case(&file))
        .map(|m| m.label)
}

fn whisper_model_dest_path(paths: &AppPaths, file_name: &str) -> PathBuf {
    whisper_model_repo_dir(paths, file_name).join(file_name)
}

fn live_model_dest_path(paths: &AppPaths, file_name: &str) -> PathBuf {
    paths.live_models_dir.join(file_name)
}

fn chat_model_dest_path(paths: &AppPaths, file_name: &str) -> PathBuf {
    paths.chat_models_dir.join(file_name)
}

pub(crate) fn resolve_chat_model_path_from_store(
    paths: &AppPaths,
    configured_path: &str,
) -> Option<PathBuf> {
    let trimmed = configured_path.trim();
    if trimmed.is_empty() {
        return None;
    }

    let configured = PathBuf::from(trimmed);
    if configured.exists() {
        return Some(configured);
    }

    let file_name = configured.file_name()?;
    let read_dir = fs::read_dir(&paths.models_dir).ok()?;
    for entry in read_dir.flatten() {
        let repo_dir = entry.path();
        if !repo_dir.is_dir() {
            continue;
        }
        let candidate = repo_dir.join(file_name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

pub(crate) fn sortformer_model_path_from_settings(
    paths: &AppPaths,
    settings: &AppSettings,
) -> PathBuf {
    let dir = if settings.diarization_models_dir.trim().is_empty() {
        default_diarization_models_dir(paths)
    } else {
        PathBuf::from(settings.diarization_models_dir.trim())
    };
    dir.join(SORTFORMER_MODEL_FILE)
}

fn whisper_combo_label(paths: &AppPaths, spec: &WhisperModelSpec) -> String {
    let model_path = whisper_model_dest_path(paths, spec.file_name);
    if model_path.exists() {
        format!(
            "{} ({}) [installed]",
            spec.label,
            human_model_size(spec.size_bytes)
        )
    } else {
        format!(
            "{} ({}) [not installed]",
            spec.label,
            human_model_size(spec.size_bytes)
        )
    }
}

fn live_model_combo_label(paths: &AppPaths, spec: &LiveModelSpec) -> String {
    let model_path = live_model_dest_path(paths, spec.file_name);
    if model_path.exists() {
        format!(
            "{} ({}) [installed]",
            spec.label,
            human_model_size(spec.size_bytes)
        )
    } else {
        format!(
            "{} ({}) [not installed]",
            spec.label,
            human_model_size(spec.size_bytes)
        )
    }
}

fn chat_model_combo_label(paths: &AppPaths, spec: &ChatModelSpec) -> String {
    let model_path = chat_model_dest_path(paths, spec.file_name);
    if model_path.exists() {
        format!(
            "{} ({}) [installed]",
            spec.label,
            human_model_size(spec.size_bytes)
        )
    } else {
        format!(
            "{} ({}) [not installed]",
            spec.label,
            human_model_size(spec.size_bytes)
        )
    }
}

fn is_managed_chat_model_path(paths: &AppPaths, raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return false;
    }
    Path::new(trimmed)
        .parent()
        .map(|parent| parent == paths.chat_models_dir.as_path())
        .unwrap_or(false)
}

fn missing_diarization_files(dir: &Path) -> Vec<&'static str> {
    if dir.join(SORTFORMER_MODEL_FILE).exists() {
        Vec::new()
    } else {
        vec![SORTFORMER_MODEL_FILE]
    }
}

fn first_installed_whisper_model(paths: &AppPaths) -> Option<&'static WhisperModelSpec> {
    WHISPER_MODELS
        .iter()
        .find(|spec| whisper_model_dest_path(paths, spec.file_name).exists())
}

fn has_any_installed_whisper_model(paths: &AppPaths) -> bool {
    first_installed_whisper_model(paths).is_some()
}

fn first_installed_live_model(paths: &AppPaths) -> Option<&'static LiveModelSpec> {
    LIVE_TRANSCRIPTION_MODELS
        .iter()
        .find(|spec| live_model_dest_path(paths, spec.file_name).exists())
}

fn has_any_installed_live_model(paths: &AppPaths) -> bool {
    first_installed_live_model(paths).is_some()
}

fn first_installed_chat_model(paths: &AppPaths) -> Option<&'static ChatModelSpec> {
    CHAT_MODELS
        .iter()
        .find(|spec| chat_model_dest_path(paths, spec.file_name).exists())
}

fn has_any_installed_chat_model(paths: &AppPaths) -> bool {
    first_installed_chat_model(paths).is_some()
}

fn backend_priority_for_platform(backend_norm: &str) -> i32 {
    #[cfg(target_os = "windows")]
    {
        if backend_norm.contains("cuda") {
            return 500;
        }
        if backend_norm.contains("vulkan") {
            return 400;
        }
        if backend_norm.contains("metal") {
            return 300;
        }
        return 100;
    }

    #[cfg(target_os = "macos")]
    {
        if backend_norm.contains("metal") {
            return 500;
        }
        if backend_norm.contains("vulkan") {
            return 400;
        }
        if backend_norm.contains("cuda") {
            return 200;
        }
        return 100;
    }

    #[cfg(target_os = "linux")]
    {
        if backend_norm.contains("vulkan") {
            return 500;
        }
        if backend_norm.contains("cuda") {
            return 400;
        }
        if backend_norm.contains("metal") {
            return 200;
        }
        return 100;
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        if backend_norm.contains("vulkan") {
            return 500;
        }
        if backend_norm.contains("cuda") {
            return 400;
        }
        if backend_norm.contains("metal") {
            return 300;
        }
        return 100;
    }
}

fn enumerate_audio_device_options(runtime_dir: &Path) -> Vec<AudioDeviceOption> {
    let mut options = vec![AudioDeviceOption {
        label: AUDIO_DEVICE_CPU_LABEL.to_string(),
        devices_value: "none".to_string(),
        main_gpu: 0,
        is_gpu: false,
        detail_line: AUDIO_DEVICE_CPU_LABEL.to_string(),
    }];
    let mut chosen_by_gpu = HashMap::<String, (i32, AudioDeviceOption)>::new();

    configure_runtime_dll_search(runtime_dir);
    let Ok(api) = BridgeApi::load(runtime_dir) else {
        return options;
    };
    let Ok(devices) = api.list_devices() else {
        return options;
    };

    for dev in devices {
        let backend = if dev.backend.trim().is_empty() {
            "unknown".to_string()
        } else {
            dev.backend.trim().to_string()
        };
        let name = if dev.name.trim().is_empty() {
            "unnamed".to_string()
        } else {
            dev.name.trim().to_string()
        };
        let description = dev.description.trim().to_string();
        let backend_norm = backend.to_ascii_lowercase();
        let name_norm = name.to_ascii_lowercase();
        if backend_norm.contains("cpu") || name_norm == "cpu" {
            continue;
        }
        let gpu_index = if dev.index < 0 { 0 } else { dev.index };
        let gpu_identity = if description.is_empty() {
            name_norm.clone()
        } else {
            description.to_ascii_lowercase()
        };
        let device_key = format!("{gpu_identity}#{gpu_index}");
        let label = if !description.is_empty() {
            format!("GPU #{} ({backend}): {description}", gpu_index)
        } else {
            format!("GPU #{} ({backend}): {name}", gpu_index)
        };
        let detail = format!(
            "{} (free {} / total {})",
            label,
            format_size(dev.memory_free),
            format_size(dev.memory_total)
        );

        let option = AudioDeviceOption {
            label,
            devices_value: gpu_index.to_string(),
            main_gpu: gpu_index,
            is_gpu: true,
            detail_line: detail,
        };

        let priority = backend_priority_for_platform(&backend_norm);
        match chosen_by_gpu.get(&device_key) {
            Some((current_prio, _)) if *current_prio >= priority => {}
            _ => {
                chosen_by_gpu.insert(device_key, (priority, option));
            }
        }
    }

    let mut selected = chosen_by_gpu
        .into_values()
        .map(|(_, option)| option)
        .collect::<Vec<_>>();
    selected.sort_by_key(|o| o.main_gpu);
    options.extend(selected);

    if options.len() == 1 {
        options[0].detail_line =
            "CPU (no GPU) - no compatible GPU backend detected, using CPU mode.".to_string();
    }

    options
}

fn resolve_audio_device_label(
    options: &[AudioDeviceOption],
    selected_gpu_index: Option<i32>,
) -> String {
    let Some(selected_gpu_index) = selected_gpu_index else {
        return AUDIO_DEVICE_CPU_LABEL.to_string();
    };

    if let Some(found) = options
        .iter()
        .find(|o| o.is_gpu && o.main_gpu == selected_gpu_index.max(0))
    {
        return found.label.clone();
    }

    options
        .iter()
        .find(|o| o.is_gpu)
        .map(|o| o.label.clone())
        .unwrap_or_else(|| AUDIO_DEVICE_CPU_LABEL.to_string())
}

fn resolve_bridge_device_name_by_index(api: &BridgeApi, gpu_index: i32) -> Option<String> {
    if gpu_index < 0 {
        return None;
    }
    let devices = api.list_devices().ok()?;
    devices
        .into_iter()
        .find(|d| d.index == gpu_index)
        .and_then(|d| {
            let trimmed = d.name.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
}

fn bridge_has_device_index(api: &BridgeApi, gpu_index: i32) -> bool {
    if gpu_index < 0 {
        return false;
    }
    let Ok(devices) = api.list_devices() else {
        return false;
    };
    devices.into_iter().any(|d| d.index == gpu_index)
}

fn format_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn download_file_with_progress(
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>, f64) + Send,
) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3600))
        .build()
        .context("failed to create HTTP client")?;
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("download request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("download request returned error status: {url}"))?;
    let total = response.content_length();

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create '{}'", parent.display()))?;
    }
    let tmp = dest.with_extension("download");
    let mut out = fs::File::create(&tmp)
        .with_context(|| format!("failed to create temporary file '{}'", tmp.display()))?;

    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;
    let start = Instant::now();
    let mut last_emit = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .unwrap_or_else(Instant::now);

    loop {
        let n = response
            .read(&mut buf)
            .with_context(|| format!("failed while reading response from '{url}'"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .with_context(|| format!("failed while writing '{}'", tmp.display()))?;
        downloaded += n as u64;

        if last_emit.elapsed() >= Duration::from_millis(350) {
            let elapsed = start.elapsed().as_secs_f64().max(0.01);
            let speed = downloaded as f64 / elapsed;
            on_progress(downloaded, total, speed);
            last_emit = Instant::now();
        }
    }

    out.flush()
        .with_context(|| format!("failed while flushing '{}'", tmp.display()))?;
    if dest.exists() {
        fs::remove_file(dest)
            .with_context(|| format!("failed to replace existing '{}'", dest.display()))?;
    }
    fs::rename(&tmp, dest).with_context(|| {
        format!(
            "failed to move temporary file '{}' to '{}'",
            tmp.display(),
            dest.display()
        )
    })?;

    let elapsed = start.elapsed().as_secs_f64().max(0.01);
    let speed = downloaded as f64 / elapsed;
    on_progress(downloaded, total, speed);
    Ok(())
}

fn start_model_download(
    tx: std::sync::mpsc::Sender<UiMessage>,
    runtime_state: Arc<Mutex<RuntimeState>>,
    family: String,
    display_name: String,
    url: String,
    dest: PathBuf,
) {
    let _ = tx.send(UiMessage::Log(format!(
        "Starting download: {family} model '{display_name}' -> {}",
        dest.display()
    )));
    let _ = tx.send(UiMessage::Status(format!("Downloading {display_name}...")));

    std::thread::spawn(move || {
        let tx_progress = tx.clone();
        let progress_label = display_name.clone();
        let result = download_file_with_progress(&url, &dest, move |done, total, speed| {
            let status = match total {
                Some(total_bytes) if total_bytes > 0 => format!(
                    "Downloading {}: {} / {} at {}/s",
                    progress_label,
                    format_size(done),
                    format_size(total_bytes),
                    format_size(speed as u64)
                ),
                _ => format!(
                    "Downloading {}: {} at {}/s",
                    progress_label,
                    format_size(done),
                    format_size(speed as u64)
                ),
            };
            let _ = tx_progress.send(UiMessage::DownloadStatus(Some(status)));
        });

        let _ = tx.send(UiMessage::DownloadStatus(None));
        match result {
            Ok(_) => {
                let _ = tx.send(UiMessage::WhisperInstalled(dest.clone()));
                let _ = tx.send(UiMessage::Status(format!("Installed {}", display_name)));
            }
            Err(e) => {
                let _ = tx.send(UiMessage::Log(format!(
                    "Download failed for {}: {}",
                    display_name, e
                )));
                let _ = tx.send(UiMessage::Status("Model download failed.".to_string()));
            }
        }
        let _ = tx.send(UiMessage::Refresh);
        drop(runtime_state);
    });
}

fn start_live_model_download(
    tx: std::sync::mpsc::Sender<UiMessage>,
    runtime_state: Arc<Mutex<RuntimeState>>,
    display_name: String,
    url: String,
    dest: PathBuf,
) {
    let _ = tx.send(UiMessage::Log(format!(
        "Starting download: realtime model '{display_name}' -> {}",
        dest.display()
    )));
    let _ = tx.send(UiMessage::Status(format!(
        "Downloading realtime model {display_name}..."
    )));

    std::thread::spawn(move || {
        let tx_progress = tx.clone();
        let progress_label = display_name.clone();
        let result = download_file_with_progress(&url, &dest, move |done, total, speed| {
            let status = match total {
                Some(total_bytes) if total_bytes > 0 => format!(
                    "Downloading {}: {} / {} at {}/s",
                    progress_label,
                    format_size(done),
                    format_size(total_bytes),
                    format_size(speed as u64)
                ),
                _ => format!(
                    "Downloading {}: {} at {}/s",
                    progress_label,
                    format_size(done),
                    format_size(speed as u64)
                ),
            };
            let _ = tx_progress.send(UiMessage::DownloadStatus(Some(status)));
        });

        let _ = tx.send(UiMessage::DownloadStatus(None));
        match result {
            Ok(_) => {
                let _ = tx.send(UiMessage::LiveModelInstalled(dest.clone()));
                let _ = tx.send(UiMessage::Status(format!(
                    "Installed realtime model {display_name}"
                )));
            }
            Err(e) => {
                let _ = tx.send(UiMessage::Log(format!(
                    "Realtime model download failed for {}: {}",
                    display_name, e
                )));
                let _ = tx.send(UiMessage::Status(
                    "Realtime model download failed.".to_string(),
                ));
            }
        }
        let _ = tx.send(UiMessage::Refresh);
        drop(runtime_state);
    });
}

fn start_chat_model_download(
    tx: std::sync::mpsc::Sender<UiMessage>,
    runtime_state: Arc<Mutex<RuntimeState>>,
    display_name: String,
    url: String,
    dest: PathBuf,
) {
    let _ = tx.send(UiMessage::Log(format!(
        "Starting download: chat model '{display_name}' -> {}",
        dest.display()
    )));
    let _ = tx.send(UiMessage::Status(format!(
        "Downloading chat model {display_name}..."
    )));

    std::thread::spawn(move || {
        let tx_progress = tx.clone();
        let progress_label = display_name.clone();
        let result = download_file_with_progress(&url, &dest, move |done, total, speed| {
            let status = match total {
                Some(total_bytes) if total_bytes > 0 => format!(
                    "Downloading {}: {} / {} at {}/s",
                    progress_label,
                    format_size(done),
                    format_size(total_bytes),
                    format_size(speed as u64)
                ),
                _ => format!(
                    "Downloading {}: {} at {}/s",
                    progress_label,
                    format_size(done),
                    format_size(speed as u64)
                ),
            };
            let _ = tx_progress.send(UiMessage::DownloadStatus(Some(status)));
        });

        let _ = tx.send(UiMessage::DownloadStatus(None));
        match result {
            Ok(_) => {
                let _ = tx.send(UiMessage::ChatModelInstalled(dest.clone()));
                let _ = tx.send(UiMessage::Status(format!("Installed chat model {display_name}")));
            }
            Err(e) => {
                let _ = tx.send(UiMessage::Log(format!(
                    "Chat model download failed for {}: {}",
                    display_name, e
                )));
                let _ = tx.send(UiMessage::Status("Chat model download failed.".to_string()));
            }
        }
        let _ = tx.send(UiMessage::Refresh);
        drop(runtime_state);
    });
}

fn start_diarization_download(
    tx: std::sync::mpsc::Sender<UiMessage>,
    runtime_state: Arc<Mutex<RuntimeState>>,
    dest_dir: PathBuf,
) {
    let _ = tx.send(UiMessage::Log(format!(
        "Starting diarization model download from {DIARIZATION_REPO} to {}",
        dest_dir.display()
    )));
    let _ = tx.send(UiMessage::Status(
        "Downloading Sortformer diarization model...".to_string(),
    ));

    std::thread::spawn(move || {
        let dest = dest_dir.join(SORTFORMER_MODEL_FILE);
        let tx_progress = tx.clone();
        let result =
            download_file_with_progress(DIARIZATION_MODEL_URL, &dest, move |done, total, speed| {
                let status = match total {
                    Some(total_bytes) if total_bytes > 0 => format!(
                        "Downloading Sortformer diarization model: {} / {} at {}/s",
                        format_size(done),
                        format_size(total_bytes),
                        format_size(speed as u64)
                    ),
                    _ => format!(
                        "Downloading Sortformer diarization model: {} at {}/s",
                        format_size(done),
                        format_size(speed as u64)
                    ),
                };
                let _ = tx_progress.send(UiMessage::DownloadStatus(Some(status)));
            });

        let _ = tx.send(UiMessage::DownloadStatus(None));
        match result {
            Ok(_) => {
                let _ = tx.send(UiMessage::DiarizationInstalled(dest_dir.clone()));
                let _ = tx.send(UiMessage::Status(
                    "Sortformer diarization model installed.".to_string(),
                ));
            }
            Err(e) => {
                let _ = tx.send(UiMessage::Log(format!(
                    "Sortformer diarization model download failed: {e}"
                )));
                let _ = tx.send(UiMessage::Status(
                    "Diarization download failed.".to_string(),
                ));
            }
        }
        let _ = tx.send(UiMessage::Refresh);
        drop(runtime_state);
    });
}

fn parse_transcript(text: &str) -> TranscriptState {
    let mut cues = Vec::new();
    let re_range = Regex::new(
        r"(?P<a>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\s*(?:-->|-)\s*(?P<b>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)",
    )
    .expect("valid regex");
    let re_bracket = Regex::new(
        r"\[(?P<a>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\s*-\s*(?P<b>\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,3})?)\]",
    )
    .expect("valid regex");

    for (line_index, raw) in text.lines().enumerate() {
        let line = raw.to_string();

        let mut found = None;
        if let Some(c) = re_range.captures(&line) {
            found = Some((c["a"].to_string(), c["b"].to_string()));
        } else if let Some(c) = re_bracket.captures(&line) {
            found = Some((c["a"].to_string(), c["b"].to_string()));
        }

        if let Some((a, b)) = found {
            if let (Some(start_sec), Some(end_sec)) = (parse_time_to_sec(&a), parse_time_to_sec(&b))
            {
                cues.push(TranscriptCue {
                    line_index,
                    start_sec,
                    end_sec,
                });
            }
        }
    }

    TranscriptState { cues }
}

fn parse_time_to_sec(token: &str) -> Option<f64> {
    let norm = token.trim().replace(',', ".");
    let mut parts = norm.split(':');
    let h = parts.next()?.parse::<f64>().ok()?;
    let m = parts.next()?.parse::<f64>().ok()?;
    let s = parts.next()?.parse::<f64>().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

fn cue_index_at_time(cues: &[TranscriptCue], t: f64) -> Option<usize> {
    if cues.is_empty() {
        return None;
    }
    for cue in cues {
        if t >= cue.start_sec && t <= cue.end_sec {
            return Some(cue.line_index);
        }
    }
    cues.iter()
        .rev()
        .find(|c| t >= c.start_sec)
        .map(|c| c.line_index)
}

fn cue_start_for_line(cues: &[TranscriptCue], line_idx: usize) -> Option<f64> {
    cues.iter()
        .rev()
        .find(|cue| cue.line_index <= line_idx)
        .or_else(|| cues.iter().find(|cue| cue.line_index >= line_idx))
        .map(|cue| cue.start_sec)
}

fn cue_line_range_at_time(
    cues: &[TranscriptCue],
    t: f64,
    total_lines: usize,
) -> Option<(usize, usize)> {
    if cues.is_empty() || total_lines == 0 {
        return None;
    }
    let pos = cues
        .iter()
        .position(|cue| t >= cue.start_sec && t <= cue.end_sec)
        .or_else(|| cues.iter().rposition(|cue| t >= cue.start_sec))
        .unwrap_or(0);
    let start = cues[pos].line_index.min(total_lines.saturating_sub(1));
    let end = cues
        .get(pos + 1)
        .map(|next| next.line_index.min(total_lines))
        .unwrap_or(total_lines)
        .max(start + 1);
    Some((start, end))
}

fn parse_f32_or(default_v: f32, text: &str) -> f32 {
    text.trim().parse::<f32>().unwrap_or(default_v)
}

fn configure_runtime_dll_search(runtime_dir: &Path) {
    #[cfg(target_os = "windows")]
    {
        use std::path::PathBuf;

        let mut candidates: Vec<PathBuf> = Vec::new();
        if !runtime_dir.as_os_str().is_empty() {
            candidates.push(runtime_dir.to_path_buf());
            candidates.push(runtime_dir.join("vendor").join("ffmpeg").join("bin"));
            candidates.push(runtime_dir.join("vendor").join("ffmpeg"));
            candidates.push(runtime_dir.join("vendor").join("pdfium"));
            let ffmpeg_root = runtime_dir.join("vendor").join("ffmpeg");
            if ffmpeg_root.exists() {
                let _ = env::set_var("FFMPEG_DIR", ffmpeg_root);
            }
        }

        let mut existing =
            env::split_paths(&env::var_os("PATH").unwrap_or_default()).collect::<Vec<_>>();
        let mut to_prepend: Vec<PathBuf> = Vec::new();
        let mut ffmpeg_env_set = false;
        for dir in candidates {
            if !dir.exists() {
                continue;
            }
            let dir_norm = dir.to_string_lossy().to_ascii_lowercase();
            let already = existing
                .iter()
                .any(|p| p.to_string_lossy().to_ascii_lowercase() == dir_norm)
                || to_prepend
                    .iter()
                    .any(|p| p.to_string_lossy().to_ascii_lowercase() == dir_norm);
            if !already {
                to_prepend.push(dir.clone());
            }

            if dir.file_name().is_some_and(|name| name == "ffmpeg") {
                let ffmpeg_root = dir.to_path_buf();
                if ffmpeg_root.join("include").exists() {
                    ffmpeg_env_set = true;
                    let ffmpeg_bin = ffmpeg_root.join("bin").join("ffmpeg.exe");
                    let _ = env::set_var("FFMPEG_DIR", ffmpeg_root.clone());
                    if ffmpeg_bin.exists() {
                        let _ = env::set_var("FFMPEG_DIR_EXE", ffmpeg_bin.clone());
                        let _ = env::set_var("FFMPEG_BINARY", ffmpeg_bin);
                    }
                    let _ = env::set_var("FFMPEG_DIR_BIN", ffmpeg_root.join("bin"));
                    let _ = env::set_var("PKG_CONFIG_PATH", ffmpeg_root.join("lib\\pkgconfig"));
                    let _ = env::set_var("PKG_CONFIG_ALLOW_SYSTEM_LIBS", "1");
                    let _ = env::set_var("PKG_CONFIG_ALLOW_SYSTEM_CFLAGS", "1");
                }
            }
        }

        if !to_prepend.is_empty() {
            let mut merged = to_prepend;
            merged.append(&mut existing);
            if let Ok(joined) = env::join_paths(merged) {
                env::set_var("PATH", joined);
            }
        }

        if ffmpeg_env_set {
            return;
        }
    }
}

fn configure_ui_startup() {
    // Keep startup hook present for parity with existing initialization flow,
    // but rely on eframe/winit defaults for DPI + renderer behavior.
}

fn resolve_runtime_dir(preferred: &Path) -> PathBuf {
    let candidates = runtime_dir_candidates(preferred);

    for dir in &candidates {
        if has_bridge_library(dir) {
            return dir.clone();
        }
    }
    for dir in &candidates {
        if dir.exists() {
            return dir.clone();
        }
    }

    if preferred.as_os_str().is_empty() {
        crate::settings::default_runtime_dir()
    } else {
        let mapped = crate::settings::normalize_runtime_dir_alias(&preferred.to_string_lossy());
        if mapped.trim().is_empty() {
            preferred.to_path_buf()
        } else {
            PathBuf::from(mapped)
        }
    }
}

fn runtime_dir_candidates(preferred: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let fallback = crate::settings::default_runtime_dir();
    let preferred_mapped =
        crate::settings::normalize_runtime_dir_alias(&preferred.to_string_lossy());

    if !preferred_mapped.trim().is_empty() {
        out.push(PathBuf::from(preferred_mapped));
    }

    out.push(fallback);

    dedupe_paths(out)
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in paths {
        if p.as_os_str().is_empty() {
            continue;
        }
        let norm = p.to_string_lossy().to_ascii_lowercase();
        if out
            .iter()
            .any(|x: &PathBuf| x.to_string_lossy().to_ascii_lowercase() == norm)
        {
            continue;
        }
        out.push(p);
    }
    out
}

fn has_bridge_library(runtime_dir: &Path) -> bool {
    if !runtime_dir.exists() {
        return false;
    }
    runtime_dir.join(bridge_library_file_name()).exists()
}

#[cfg(target_os = "windows")]
fn bridge_library_file_name() -> &'static str {
    "llama-server-bridge.dll"
}

#[cfg(target_os = "macos")]
fn bridge_library_file_name() -> &'static str {
    "libllama-server-bridge.dylib"
}

#[cfg(all(unix, not(target_os = "macos")))]
fn bridge_library_file_name() -> &'static str {
    "libllama-server-bridge.so"
}

fn is_default_or_positive_float(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed.eq_ignore_ascii_case("default") || trimmed.eq_ignore_ascii_case("auto") {
        return true;
    }
    trimmed
        .parse::<f64>()
        .map(|v| v.is_finite() && v > 0.0)
        .unwrap_or(false)
}

fn infer_audio_format(audio_path: &Path) -> String {
    audio_path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.to_ascii_lowercase())
        .unwrap_or_else(|| "bin".to_string())
}

fn fallback_output_path(audio_path: &Path, mode: &str, output_dir: &Path) -> PathBuf {
    let stem = audio_path
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("output");
    let ext = if mode.eq_ignore_ascii_case("subtitle") {
        "srt"
    } else {
        "md"
    };
    output_dir.join(format!("{stem}.{ext}"))
}

fn edited_file_path(original: &Path) -> PathBuf {
    let parent = original.parent().unwrap_or_else(|| Path::new("."));
    let stem = original
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("output");
    let ext = original
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("md");
    parent.join(format!("{stem}.edited.{ext}"))
}

fn chat_log_path_for_output(output_file: &Path) -> PathBuf {
    let parent = output_file.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_file
        .file_stem()
        .and_then(|x| x.to_str())
        .unwrap_or("chat");
    parent.join(format!("{stem}.log"))
}

fn positive_optional(v: i32) -> Option<i32> {
    if v > 0 {
        Some(v)
    } else {
        None
    }
}

fn default_runtime_threads() -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        let cpus = std::thread::available_parallelism().ok()?.get() as i32;
        if cpus <= 1 {
            return Some(1);
        }
        return Some((cpus - 1).max(1));
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn normalize_playback_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn playback_paths_equal(a: &Path, b: &Path) -> bool {
    let left = normalize_playback_path(a);
    let right = normalize_playback_path(b);
    if cfg!(windows) {
        left.to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy())
    } else {
        left == right
    }
}

fn playback_current_time(state: &PlaybackState) -> f64 {
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(inner) = shared.try_lock() {
            if state.output_sample_rate > 0 {
                return (inner.position_frames / state.output_sample_rate as f64).max(0.0);
            }
        }
    }
    0.0
}

fn playback_has_audio(state: &PlaybackState) -> bool {
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(inner) = shared.lock() {
            return inner.total_frames > 0;
        }
    }
    false
}

fn playback_is_ended(state: &PlaybackState) -> bool {
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(inner) = shared.lock() {
            return inner.ended;
        }
    }
    false
}

fn playback_apply_speed(state: &mut PlaybackState) {
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(mut inner) = shared.lock() {
            inner.speed = state.speed.clamp(0.1, 4.0);
        }
    }
}

fn playback_build_stream(
    shared: Arc<Mutex<PlaybackBuffer>>,
) -> Result<(Stream, StreamConfig, SampleFormat, u32, u16)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default audio output device available"))?;
    let supported = device
        .default_output_config()
        .context("failed to get default output config")?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.config();
    let channels = config.channels as usize;
    let shared_for_cb = shared.clone();

    let err_fn = |err| eprintln!("playback stream error: {err}");
    let stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            &config,
            move |data: &mut [f32], _| playback_write_output_f32(data, channels, &shared_for_cb),
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_output_stream(
            &config,
            move |data: &mut [i16], _| playback_write_output_i16(data, channels, &shared_for_cb),
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_output_stream(
            &config,
            move |data: &mut [u16], _| playback_write_output_u16(data, channels, &shared_for_cb),
            err_fn,
            None,
        )?,
        _ => bail!("unsupported output sample format"),
    };
    stream.play().context("failed to start playback stream")?;

    Ok((
        stream,
        config.clone(),
        sample_format,
        config.sample_rate.0,
        config.channels,
    ))
}

fn playback_write_output_f32(
    data: &mut [f32],
    out_channels: usize,
    shared: &Arc<Mutex<PlaybackBuffer>>,
) {
    if let Ok(mut guard) = shared.lock() {
        playback_fill_samples(data, out_channels, &mut guard, |dst, val| *dst = val);
    } else {
        for sample in data.iter_mut() {
            *sample = 0.0;
        }
    }
}

fn playback_write_output_i16(
    data: &mut [i16],
    out_channels: usize,
    shared: &Arc<Mutex<PlaybackBuffer>>,
) {
    if let Ok(mut guard) = shared.lock() {
        playback_fill_samples(data, out_channels, &mut guard, |dst, val| {
            *dst = (val.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        });
    } else {
        for sample in data.iter_mut() {
            *sample = 0;
        }
    }
}

fn playback_write_output_u16(
    data: &mut [u16],
    out_channels: usize,
    shared: &Arc<Mutex<PlaybackBuffer>>,
) {
    if let Ok(mut guard) = shared.lock() {
        playback_fill_samples(data, out_channels, &mut guard, |dst, val| {
            let s = ((val.clamp(-1.0, 1.0) * 0.5) + 0.5) * u16::MAX as f32;
            *dst = s as u16;
        });
    } else {
        for sample in data.iter_mut() {
            *sample = u16::MAX / 2;
        }
    }
}

fn playback_fill_samples<T, F>(
    data: &mut [T],
    out_channels: usize,
    guard: &mut PlaybackBuffer,
    mut assign: F,
) where
    F: FnMut(&mut T, f32),
{
    if !guard.playing || guard.total_frames == 0 || out_channels == 0 {
        for sample in data.iter_mut() {
            assign(sample, 0.0);
        }
        return;
    }

    for frame in data.chunks_mut(out_channels) {
        let frame_index = guard.position_frames.floor() as usize;
        if frame_index >= guard.total_frames {
            guard.playing = false;
            guard.ended = true;
            for sample in frame.iter_mut() {
                assign(sample, 0.0);
            }
            continue;
        }

        let src_i = frame_index * 2;
        let left = guard.samples_stereo_f32.get(src_i).copied().unwrap_or(0.0);
        let right = guard
            .samples_stereo_f32
            .get(src_i + 1)
            .copied()
            .unwrap_or(left);

        for (ch, sample) in frame.iter_mut().enumerate() {
            let val = if ch == 0 {
                left
            } else if ch == 1 {
                right
            } else if ch % 2 == 0 {
                left
            } else {
                right
            };
            assign(sample, val);
        }

        guard.position_frames += guard.speed as f64;
    }
}

fn decode_audio_to_stereo_f32(audio_path: &Path, dst_rate: u32) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::default::{get_codecs, get_probe};

    let file = fs::File::open(audio_path)
        .with_context(|| format!("failed to open audio '{}'", audio_path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = audio_path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| {
            format!(
                "unsupported or unrecognized media '{}'",
                audio_path.display()
            )
        })?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no audio track found in '{}'", audio_path.display()))?;
    let src_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("missing sample rate metadata in '{}'", audio_path.display()))?;
    let track_id = track.id;

    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("failed to initialize audio decoder")?;
    let mut out_stereo = Vec::<f32>::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("decoder reset required; unsupported stream state");
            }
            Err(err) => return Err(anyhow!("audio demux error: {err}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => {
                continue;
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("decoder reset required while decoding");
            }
            Err(err) => return Err(anyhow!("audio decode error: {err}")),
        };

        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }

        let mut sample_buf = SampleBuffer::<f32>::new(frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();
        for frame_idx in 0..frames {
            let base = frame_idx * channels;
            let left = samples.get(base).copied().unwrap_or(0.0);
            let right = if channels > 1 {
                samples.get(base + 1).copied().unwrap_or(left)
            } else {
                left
            };
            out_stereo.push(left);
            out_stereo.push(right);
        }
    }

    if out_stereo.is_empty() {
        bail!(
            "decoded audio is empty for '{}'; unsupported/corrupt input?",
            audio_path.display()
        );
    }

    Ok(resample_stereo_linear(
        &out_stereo,
        src_rate,
        dst_rate.max(8_000),
    ))
}

fn resample_stereo_linear(samples_stereo: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == 0 || dst_rate == 0 || samples_stereo.len() < 4 {
        return samples_stereo.to_vec();
    }
    if src_rate == dst_rate {
        return samples_stereo.to_vec();
    }

    let in_frames = samples_stereo.len() / 2;
    if in_frames == 0 {
        return Vec::new();
    }
    let out_frames =
        (((in_frames as f64) * (dst_rate as f64) / (src_rate as f64)).round() as usize).max(1);
    let mut out = Vec::with_capacity(out_frames * 2);

    for i in 0..out_frames {
        let src_pos = (i as f64) * (src_rate as f64) / (dst_rate as f64);
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(in_frames.saturating_sub(1));
        let frac = (src_pos - i0 as f64) as f32;

        let l0 = samples_stereo[i0 * 2];
        let r0 = samples_stereo[i0 * 2 + 1];
        let l1 = samples_stereo[i1 * 2];
        let r1 = samples_stereo[i1 * 2 + 1];

        out.push(l0 + (l1 - l0) * frac);
        out.push(r0 + (r1 - r0) * frac);
    }

    out
}

fn playback_ensure_stream(state: &mut PlaybackState) -> Result<()> {
    if state.stream.is_some() && state.shared.is_some() {
        return Ok(());
    }
    let shared = Arc::new(Mutex::new(PlaybackBuffer::default()));
    let (stream, config, format, rate, channels) = playback_build_stream(shared.clone())?;
    state.stream = Some(stream);
    state.stream_config = Some(config);
    state.sample_format = Some(format);
    state.shared = Some(shared);
    state.output_sample_rate = rate;
    state.output_channels = channels;
    Ok(())
}

fn playback_start_from(state: &mut PlaybackState, audio_path: &Path, start_sec: f64) -> Result<()> {
    let runtime_dir = resolve_runtime_dir(Path::new(""));
    configure_runtime_dll_search(&runtime_dir);
    playback_ensure_stream(state)?;
    let normalized_audio_path = normalize_playback_path(audio_path);
    let sample_rate = state.output_sample_rate.max(16_000);
    let start_frame = (start_sec.max(0.0) * sample_rate as f64).floor();

    if let Some(shared) = state.shared.as_ref() {
        if let Ok(mut inner) = shared.lock() {
            let already_loaded = inner.total_frames > 0
                && inner
                    .source_path
                    .as_ref()
                    .map(|p| playback_paths_equal(p, &normalized_audio_path))
                    .unwrap_or(false);
            if already_loaded {
                inner.position_frames = start_frame.min(inner.total_frames as f64);
                inner.playing = true;
                inner.ended = false;
                inner.speed = state.speed.clamp(0.1, 4.0);
                state.audio_path = Some(normalized_audio_path.clone());
                return Ok(());
            }
        }
    }

    let data = decode_audio_to_stereo_f32(&normalized_audio_path, sample_rate)?;
    let total_frames = data.len() / 2;

    if let Some(shared) = state.shared.as_ref() {
        if let Ok(mut inner) = shared.lock() {
            inner.samples_stereo_f32 = data;
            inner.total_frames = total_frames;
            inner.position_frames = start_frame.min(total_frames as f64);
            inner.playing = true;
            inner.ended = false;
            inner.speed = state.speed.clamp(0.1, 4.0);
            inner.source_path = Some(normalized_audio_path.clone());
        }
    }
    state.audio_path = Some(normalized_audio_path);
    Ok(())
}

fn playback_set_decoded_buffer(
    state: &mut PlaybackState,
    audio_path: &Path,
    data: Vec<f32>,
    start_sec: f64,
) -> Result<f64> {
    let runtime_dir = resolve_runtime_dir(Path::new(""));
    configure_runtime_dll_search(&runtime_dir);
    playback_ensure_stream(state)?;
    let normalized_audio_path = normalize_playback_path(audio_path);
    let sample_rate = state.output_sample_rate.max(16_000);
    let total_frames = data.len() / 2;
    let start_frame = (start_sec.max(0.0) * sample_rate as f64).floor();
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(mut inner) = shared.lock() {
            inner.samples_stereo_f32 = data;
            inner.total_frames = total_frames;
            inner.position_frames = start_frame.min(total_frames as f64);
            inner.playing = true;
            inner.ended = false;
            inner.speed = state.speed.clamp(0.1, 4.0);
            inner.source_path = Some(normalized_audio_path.clone());
        }
    }
    state.audio_path = Some(normalized_audio_path);
    Ok(playback_current_time(state))
}

fn playback_set_decoded_buffer_paused(
    state: &mut PlaybackState,
    audio_path: &Path,
    data: Vec<f32>,
) -> Result<()> {
    let runtime_dir = resolve_runtime_dir(Path::new(""));
    configure_runtime_dll_search(&runtime_dir);
    playback_ensure_stream(state)?;
    let normalized_audio_path = normalize_playback_path(audio_path);
    let total_frames = data.len() / 2;
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(mut inner) = shared.lock() {
            inner.samples_stereo_f32 = data;
            inner.total_frames = total_frames;
            inner.position_frames = 0.0;
            inner.playing = false;
            inner.ended = false;
            inner.speed = state.speed.clamp(0.1, 4.0);
            inner.source_path = Some(normalized_audio_path.clone());
        }
    }
    state.audio_path = Some(normalized_audio_path);
    Ok(())
}

fn playback_has_loaded_path(state: &PlaybackState, path: &Path) -> bool {
    let normalized_path = normalize_playback_path(path);
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(inner) = shared.lock() {
            return inner.total_frames > 0
                && inner
                    .source_path
                    .as_ref()
                    .map(|p| playback_paths_equal(p, &normalized_path))
                    .unwrap_or(false);
        }
    }
    false
}

fn playback_toggle_pause(state: &mut PlaybackState, path: &Path, start_sec: f64) -> Result<f64> {
    if !playback_has_audio(state) || playback_is_ended(state) {
        playback_start_from(state, path, start_sec)?;
        return Ok(playback_current_time(state));
    }
    if let Some(shared) = state.shared.as_ref() {
        if let Ok(mut inner) = shared.lock() {
            inner.playing = !inner.playing;
            return Ok(playback_current_time(state));
        }
    }
    bail!("playback state unavailable")
}

fn playback_seek_relative(state: &mut PlaybackState, delta: f64, growth: f64) -> Result<f64> {
    let path = if let Some(path) = state.audio_path.clone() {
        path
    } else {
        bail!("no active audio for seek");
    };
    let now = Instant::now();
    let step = if let Some(last) = state.last_seek_hotkey_at {
        if now.duration_since(last) < Duration::from_millis(800) {
            state.seek_repeat_count = state.seek_repeat_count.saturating_add(1);
            let mul = 1.0 + (state.seek_repeat_count as f64 * growth.max(0.0));
            delta * mul
        } else {
            state.seek_repeat_count = 0;
            delta
        }
    } else {
        delta
    };
    state.last_seek_hotkey_at = Some(now);

    let cur = playback_current_time(state);
    let target = (cur + step).max(0.0);
    playback_start_from(state, &path, target)?;
    Ok(target)
}
