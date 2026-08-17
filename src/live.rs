use anyhow::{anyhow, bail, Result};
use std::fs;
#[cfg(target_os = "linux")]
use std::fs::{File, OpenOptions};
#[cfg(target_os = "linux")]
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
#[cfg(target_os = "linux")]
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_os = "linux")]
use serde::{Deserialize, Serialize};

use crate::audio_capture_api::{AudioCaptureApi, AudioLiveConfig, AudioLivePaths};
use crate::audio_orchestrator::DiarizedTranscriptOrchestrator;
use crate::bridge::{
    AudioSessionEvent, BridgeApi, AUDIO_EVENT_DIARIZATION_SPAN_COMMIT,
    AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT, AUDIO_EVENT_ERROR, AUDIO_EVENT_NOTICE,
    AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT, AUDIO_EVENT_TRANSCRIPTION_STOPPED,
    AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT, REALTIME_BACKEND_SORTFORMER, REALTIME_BACKEND_VOXTRAL,
};
use crate::{
    bridge_has_device_index, resolve_bridge_device_name_by_index, selected_gpu_index_from_settings,
};
use crate::{AppPaths, AppSettings, RuntimeState, UiMessage};

const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
const TARGET_CHANNELS: u32 = 1;
const LIVE_PUSH_SAMPLES: u32 = 7_680;

#[derive(Debug, Clone)]
pub(crate) struct LiveInputDeviceOption {
    pub name: String,
    pub label: String,
}

pub(crate) struct ActiveLiveCapture {
    pub session_id: u64,
    pub recording_path: PathBuf,
    pub transcript_path: PathBuf,
    pub input_label: String,
    pub stop_requested: Arc<AtomicBool>,
}

impl Drop for ActiveLiveCapture {
    fn drop(&mut self) {
        self.stop_requested.store(true, Ordering::Relaxed);
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug, Serialize, Deserialize)]
struct LinuxLiveHelperRequest {
    session_id: u64,
    settings: AppSettings,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum LinuxLiveEvent {
    Status {
        text: String,
    },
    Started {
        session_id: u64,
        input_device: String,
        recording_path: PathBuf,
        transcript_path: PathBuf,
    },
    TextAppend {
        session_id: u64,
        chunk: String,
    },
    TextSet {
        session_id: u64,
        text: String,
    },
    Finished {
        session_id: u64,
        input_device: String,
        recording_path: PathBuf,
        transcript_path: PathBuf,
        transcript_text: String,
        preview_text: String,
    },
    Failed {
        session_id: u64,
        error: String,
    },
}

#[cfg(target_os = "linux")]
pub(crate) fn enumerate_input_device_options(_runtime_dir: &Path) -> Vec<LiveInputDeviceOption> {
    let default_option = LiveInputDeviceOption {
        name: String::new(),
        label: "Default input".to_string(),
    };
    // CPAL 0.15's ALSA iterator opens every PCM once for playback and once for
    // capture merely to enumerate it. That can block the GUI and produces an
    // error for every dmix/dsnoop entry. `arecord -L` reads ALSA's capture
    // hints without opening any device; alsa-utils is a declared Linux package
    // dependency.
    let Ok(output) = Command::new("arecord").arg("-L").output() else {
        return vec![default_option];
    };
    if !output.status.success() {
        return vec![default_option];
    }
    let mut names = parse_arecord_device_names(&String::from_utf8_lossy(&output.stdout));
    names.sort_by_key(|name| name.to_ascii_lowercase());
    names.dedup();

    let mut out = vec![default_option];
    for name in names {
        if name == "null" || name == "default" {
            continue;
        }
        let label = if name == "pipewire" || name == "pulse" {
            format!("{name} (sound server)")
        } else {
            name.clone()
        };
        out.push(LiveInputDeviceOption { name, label });
    }
    out
}

#[cfg(target_os = "linux")]
fn parse_arecord_device_names(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(char::is_whitespace))
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn enumerate_input_device_options(runtime_dir: &Path) -> Vec<LiveInputDeviceOption> {
    let default_option = LiveInputDeviceOption {
        name: String::new(),
        label: "Default input".to_string(),
    };

    let Ok(api) = AudioCaptureApi::load(runtime_dir) else {
        return vec![default_option];
    };
    let Ok(mut devices) = api.list_capture_devices() else {
        return vec![default_option];
    };

    devices.sort_by(|a, b| {
        b.is_default.cmp(&a.is_default).then_with(|| {
            a.name
                .to_ascii_lowercase()
                .cmp(&b.name.to_ascii_lowercase())
        })
    });

    let mut out = vec![default_option];
    for device in devices {
        let label = if device.is_default {
            format!("{} (default)", device.name)
        } else {
            device.name.clone()
        };
        out.push(LiveInputDeviceOption {
            name: device.name,
            label,
        });
    }
    out
}

pub(crate) fn resolve_input_device_index(
    devices: &[LiveInputDeviceOption],
    configured_name: &str,
) -> usize {
    let configured = configured_name.trim();
    if configured.is_empty() {
        return 0;
    }
    devices
        .iter()
        .position(|device| device.name == configured)
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
pub(crate) fn start_live_capture(
    paths: &AppPaths,
    settings: &AppSettings,
    _runtime_state: Arc<Mutex<RuntimeState>>,
    tx: mpsc::Sender<UiMessage>,
) -> Result<ActiveLiveCapture> {
    let live_model_path = PathBuf::from(settings.live_transcription_model.trim());
    if !live_model_path.is_file() {
        bail!(
            "live transcription model not found: '{}'",
            live_model_path.display()
        );
    }
    if settings.live_diarization_enabled {
        let _ = diarization_model_path(paths, settings)?;
    }

    let session_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    let session_name = format!("live-session-{session_id}");
    let output_dir = crate::settings::resolve_live_sessions_output_dir(
        &settings.live_sessions_output_dir,
        paths,
    );
    fs::create_dir_all(&output_dir).map_err(|error| {
        anyhow!(
            "failed to create live sessions output directory '{}': {error}",
            output_dir.display()
        )
    })?;
    let recording_path = output_dir.join(format!("{session_name}.clean.wav"));
    let transcript_path = output_dir.join(if settings.live_diarization_enabled {
        format!("{session_name}.transcript.md")
    } else {
        format!("{session_name}.transcript.txt")
    });
    let input_label = if settings.live_input_device.trim().is_empty() {
        "Default input".to_string()
    } else {
        settings.live_input_device.trim().to_string()
    };

    let ipc_dir = create_linux_live_ipc_dir(session_id)?;
    let request_path = ipc_dir.join("request.json");
    let events_path = ipc_dir.join("events.jsonl");
    let stop_path = ipc_dir.join("stop");
    let log_path = ipc_dir.join("engine-helper.log");
    let request = LinuxLiveHelperRequest {
        session_id,
        settings: settings.clone(),
    };
    fs::write(&request_path, serde_json::to_vec(&request)?)?;
    fs::write(&events_path, b"")?;
    let log_file = File::create(&log_path)?;
    let helper_exe = std::env::current_exe()
        .map_err(|error| anyhow!("failed to locate Transcribe Offline executable: {error}"))?;
    let mut command = Command::new(&helper_exe);
    command
        .arg("--linux-engine-live-helper")
        .arg(&request_path)
        .arg(&events_path)
        .arg(&stop_path)
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file));
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let _ = fs::remove_dir_all(&ipc_dir);
            return Err(anyhow!(
                "failed to launch isolated Linux Engine live helper '{}': {error}",
                helper_exe.display()
            ));
        }
    };

    let stop_requested = Arc::new(AtomicBool::new(false));
    let monitor_stop_requested = stop_requested.clone();
    std::thread::spawn(move || {
        let mut event_offset = 0u64;
        let mut stop_sent = false;
        let mut terminal_event_seen = false;
        loop {
            if monitor_stop_requested.load(Ordering::Relaxed) && !stop_sent {
                if let Err(error) = fs::write(&stop_path, b"stop\n") {
                    let _ = tx.send(UiMessage::Status(format!(
                        "Failed to signal live Engine helper to stop: {error}"
                    )));
                }
                stop_sent = true;
            }

            match forward_new_linux_live_events(&events_path, &mut event_offset, &tx) {
                Ok(saw_terminal) => terminal_event_seen |= saw_terminal,
                Err(error) => {
                    let _ = tx.send(UiMessage::Status(format!(
                        "Failed to read live Engine helper events: {error}"
                    )));
                }
            }
            if terminal_event_seen {
                let _ = child.wait();
                break;
            }

            match child.try_wait() {
                Ok(Some(status)) => {
                    if let Ok(saw_terminal) =
                        forward_new_linux_live_events(&events_path, &mut event_offset, &tx)
                    {
                        terminal_event_seen |= saw_terminal;
                    }
                    if !terminal_event_seen {
                        let log = fs::read(&log_path)
                            .map(|bytes| output_tail_for_live_helper(&bytes))
                            .unwrap_or_default();
                        let detail = if log.is_empty() {
                            format!("live Engine helper exited with {status}")
                        } else {
                            format!("live Engine helper exited with {status}: {log}")
                        };
                        let _ = tx.send(UiMessage::LiveSessionFailed {
                            session_id,
                            error: detail,
                        });
                    }
                    break;
                }
                Ok(None) => {}
                Err(error) => {
                    let _ = tx.send(UiMessage::LiveSessionFailed {
                        session_id,
                        error: format!("failed to monitor live Engine helper: {error}"),
                    });
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = fs::remove_dir_all(&ipc_dir);
    });

    Ok(ActiveLiveCapture {
        session_id,
        recording_path,
        transcript_path,
        input_label,
        stop_requested,
    })
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn start_live_capture(
    paths: &AppPaths,
    settings: &AppSettings,
    runtime_state: Arc<Mutex<RuntimeState>>,
    tx: mpsc::Sender<UiMessage>,
) -> Result<ActiveLiveCapture> {
    start_live_capture_in_process(paths, settings, runtime_state, tx, None)
}

fn start_live_capture_in_process(
    paths: &AppPaths,
    settings: &AppSettings,
    runtime_state: Arc<Mutex<RuntimeState>>,
    tx: mpsc::Sender<UiMessage>,
    requested_session_id: Option<u64>,
) -> Result<ActiveLiveCapture> {
    let runtime_dir = PathBuf::from(settings.runtime_dir.trim());
    let bridge_api = BridgeApi::load(&runtime_dir)?;
    let backend_name = resolve_runtime_backend_name(&bridge_api, settings)?;
    let output_dir = crate::settings::resolve_live_sessions_output_dir(
        &settings.live_sessions_output_dir,
        paths,
    );
    fs::create_dir_all(&output_dir).map_err(|err| {
        anyhow!(
            "failed to create live sessions output directory '{}': {err}",
            output_dir.display()
        )
    })?;

    let live_model_path = PathBuf::from(settings.live_transcription_model.trim());
    if !live_model_path.exists() {
        bail!(
            "live transcription model not found: '{}'",
            live_model_path.display()
        );
    }

    let diarization_model_path = if settings.live_diarization_enabled {
        let model_path = diarization_model_path(paths, settings)?;
        Some(model_path)
    } else {
        None
    };

    let session_id = requested_session_id.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    });
    let session_name = format!("live-session-{session_id}");
    let recording_path = output_dir.join(format!("{session_name}.clean.wav"));
    let transcript_path = output_dir.join(if settings.live_diarization_enabled {
            format!("{session_name}.transcript.md")
        } else {
            format!("{session_name}.transcript.txt")
        });
    let input_label = if settings.live_input_device.trim().is_empty() {
        "Default input".to_string()
    } else {
        settings.live_input_device.trim().to_string()
    };

    let stop_requested = Arc::new(AtomicBool::new(false));
    let worker_stop_requested = stop_requested.clone();
    let worker_runtime_dir = runtime_dir.clone();
    let worker_tx = tx.clone();
    let worker_recording_path = recording_path.clone();
    let worker_transcript_path = transcript_path.clone();
    let worker_input_label = input_label.clone();
    let worker_output_dir = output_dir.clone();
    let worker_session_name = session_name.clone();
    let worker_capture_device_name = if settings.live_input_device.trim().is_empty() {
        None
    } else {
        Some(settings.live_input_device.trim().to_string())
    };
    let worker_live_model_path = live_model_path.display().to_string();
    let worker_diarization_model_path = diarization_model_path
        .as_ref()
        .map(|path| path.display().to_string());
    let worker_backend_name = backend_name.clone();
    let worker_live_diarization_enabled = settings.live_diarization_enabled;
    std::thread::spawn(move || {
        if let Err(err) = live_worker(
            worker_runtime_dir,
            session_id,
            worker_output_dir,
            worker_session_name,
            worker_capture_device_name,
            worker_live_model_path,
            worker_diarization_model_path,
            worker_backend_name,
            true,
            worker_live_diarization_enabled,
            runtime_state,
            worker_tx.clone(),
            worker_recording_path,
            worker_transcript_path,
            worker_input_label,
            worker_stop_requested,
        ) {
            let _ = worker_tx.send(UiMessage::LiveSessionFailed {
                session_id,
                error: err.to_string(),
            });
        }
    });

    Ok(ActiveLiveCapture {
        session_id,
        recording_path,
        transcript_path,
        input_label,
        stop_requested,
    })
}

#[cfg(target_os = "linux")]
pub(crate) fn maybe_run_linux_live_helper(args: &[String]) -> Option<i32> {
    if args.get(1).map(String::as_str) != Some("--linux-engine-live-helper") {
        return None;
    }
    let request_path = args.get(2).map(PathBuf::from);
    let events_path = args.get(3).map(PathBuf::from);
    let stop_path = args.get(4).map(PathBuf::from);
    let result = match (request_path, events_path.as_ref(), stop_path) {
        (Some(request_path), Some(events_path), Some(stop_path)) => {
            run_linux_live_helper(&request_path, events_path, &stop_path)
        }
        _ => Err(anyhow!(
            "live helper requires request, events, and stop paths"
        )),
    };
    match result {
        Ok(()) => Some(0),
        Err(error) => {
            if let Some(events_path) = events_path {
                let session_id = args
                    .get(2)
                    .and_then(|path| fs::read(path).ok())
                    .and_then(|raw| serde_json::from_slice::<LinuxLiveHelperRequest>(&raw).ok())
                    .map(|request| request.session_id)
                    .unwrap_or(0);
                let _ = append_linux_live_event(
                    &events_path,
                    &LinuxLiveEvent::Failed {
                        session_id,
                        error: error.to_string(),
                    },
                );
            }
            eprintln!("Linux Engine live helper failed: {error}");
            Some(1)
        }
    }
}

#[cfg(target_os = "linux")]
fn run_linux_live_helper(
    request_path: &Path,
    events_path: &Path,
    stop_path: &Path,
) -> Result<()> {
    let request: LinuxLiveHelperRequest = serde_json::from_slice(
        &fs::read(request_path)
            .map_err(|error| anyhow!("failed to read live helper request: {error}"))?,
    )
    .map_err(|error| anyhow!("failed to parse live helper request: {error}"))?;
    let paths = crate::settings::app_paths()?;
    crate::settings::ensure_dirs(&paths)?;
    let (tx, rx) = mpsc::channel();
    let runtime_state = Arc::new(Mutex::new(RuntimeState::default()));
    let capture = start_live_capture_in_process(
        &paths,
        &request.settings,
        runtime_state,
        tx,
        Some(request.session_id),
    )?;
    let mut stop_forwarded = false;

    loop {
        if stop_path.exists() && !stop_forwarded {
            capture.stop_requested.store(true, Ordering::Relaxed);
            stop_forwarded = true;
        }
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(message) => {
                if let Some((event, terminal)) = linux_live_event_from_ui_message(message) {
                    append_linux_live_event(events_path, &event)?;
                    if terminal {
                        return Ok(());
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("live Engine helper event channel disconnected unexpectedly");
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_live_event_from_ui_message(message: UiMessage) -> Option<(LinuxLiveEvent, bool)> {
    match message {
        UiMessage::Status(text) | UiMessage::Log(text) => {
            Some((LinuxLiveEvent::Status { text }, false))
        }
        UiMessage::LiveSessionStarted {
            session_id,
            input_device,
            recording_path,
            transcript_path,
        } => Some((
            LinuxLiveEvent::Started {
                session_id,
                input_device,
                recording_path,
                transcript_path,
            },
            false,
        )),
        UiMessage::LiveTextAppend { session_id, chunk } => Some((
            LinuxLiveEvent::TextAppend { session_id, chunk },
            false,
        )),
        UiMessage::LiveTextSet { session_id, text } => {
            Some((LinuxLiveEvent::TextSet { session_id, text }, false))
        }
        UiMessage::LiveSessionFinished {
            session_id,
            input_device,
            recording_path,
            transcript_path,
            transcript_text,
            preview_text,
        } => Some((
            LinuxLiveEvent::Finished {
                session_id,
                input_device,
                recording_path,
                transcript_path,
                transcript_text,
                preview_text,
            },
            true,
        )),
        UiMessage::LiveSessionFailed { session_id, error } => {
            Some((LinuxLiveEvent::Failed { session_id, error }, true))
        }
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn append_linux_live_event(path: &Path, event: &LinuxLiveEvent) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, event)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn forward_new_linux_live_events(
    path: &Path,
    offset: &mut u64,
    tx: &mpsc::Sender<UiMessage>,
) -> Result<bool> {
    let mut file = OpenOptions::new().read(true).open(path)?;
    file.seek(SeekFrom::Start(*offset))?;
    let mut reader = BufReader::new(file);
    let mut terminal = false;
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        if !line.ends_with('\n') {
            break;
        }
        *offset += bytes_read as u64;
        let event: LinuxLiveEvent = serde_json::from_str(line.trim_end())?;
        terminal |= forward_linux_live_event(event, tx);
    }
    Ok(terminal)
}

#[cfg(target_os = "linux")]
fn forward_linux_live_event(
    event: LinuxLiveEvent,
    tx: &mpsc::Sender<UiMessage>,
) -> bool {
    let (message, terminal) = match event {
        LinuxLiveEvent::Status { text } => (UiMessage::Status(text), false),
        LinuxLiveEvent::Started {
            session_id,
            input_device,
            recording_path,
            transcript_path,
        } => (
            UiMessage::LiveSessionStarted {
                session_id,
                input_device,
                recording_path,
                transcript_path,
            },
            false,
        ),
        LinuxLiveEvent::TextAppend { session_id, chunk } => {
            (UiMessage::LiveTextAppend { session_id, chunk }, false)
        }
        LinuxLiveEvent::TextSet { session_id, text } => {
            (UiMessage::LiveTextSet { session_id, text }, false)
        }
        LinuxLiveEvent::Finished {
            session_id,
            input_device,
            recording_path,
            transcript_path,
            transcript_text,
            preview_text,
        } => (
            UiMessage::LiveSessionFinished {
                session_id,
                input_device,
                recording_path,
                transcript_path,
                transcript_text,
                preview_text,
            },
            true,
        ),
        LinuxLiveEvent::Failed { session_id, error } => {
            (UiMessage::LiveSessionFailed { session_id, error }, true)
        }
    };
    let _ = tx.send(message);
    terminal
}

#[cfg(target_os = "linux")]
fn create_linux_live_ipc_dir(session_id: u64) -> Result<PathBuf> {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .unwrap_or_else(std::env::temp_dir);
    for attempt in 0..100u32 {
        let path = base.join(format!(
            "transcribe-offline-live-{}-{session_id}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(anyhow!(
                    "failed to create live helper IPC directory '{}': {error}",
                    path.display()
                ));
            }
        }
    }
    bail!("failed to allocate unique live helper IPC directory")
}

#[cfg(target_os = "linux")]
fn output_tail_for_live_helper(raw: &[u8]) -> String {
    const MAX_CHARS: usize = 2_000;
    let text = String::from_utf8_lossy(raw);
    let char_count = text.chars().count();
    if char_count <= MAX_CHARS {
        return text.trim().to_string();
    }
    text.chars()
        .skip(char_count - MAX_CHARS)
        .collect::<String>()
        .trim()
        .to_string()
}

fn live_worker(
    runtime_dir: PathBuf,
    session_id: u64,
    output_dir: PathBuf,
    session_name: String,
    capture_device_name: Option<String>,
    transcription_model_path: String,
    diarization_model_path: Option<String>,
    backend_name: String,
    webrtc_enabled: bool,
    diarization_enabled: bool,
    runtime_state: Arc<Mutex<RuntimeState>>,
    tx: mpsc::Sender<UiMessage>,
    fallback_recording_path: PathBuf,
    fallback_transcript_path: PathBuf,
    input_label: String,
    stop_requested: Arc<AtomicBool>,
) -> Result<()> {
    let bridge_api = BridgeApi::load(&runtime_dir)?;
    let mut session_params = bridge_api.default_audio_session_params_native();
    session_params.expected_input_sample_rate_hz = TARGET_SAMPLE_RATE_HZ;
    session_params.expected_input_channels = TARGET_CHANNELS;
    session_params.max_buffered_audio_samples = 0;
    session_params.event_queue_capacity = 0;

    let mut transcription_params = bridge_api.default_audio_transcription_params_native();
    transcription_params.mode = crate::bridge::AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE;
    transcription_params.realtime_params =
        bridge_api.default_realtime_params_native_for_backend(REALTIME_BACKEND_VOXTRAL);
    transcription_params.realtime_params.backend_kind = REALTIME_BACKEND_VOXTRAL;
    transcription_params.realtime_params.expected_sample_rate_hz = TARGET_SAMPLE_RATE_HZ;

    let diarization_params = if diarization_enabled {
        let mut params =
            bridge_api.default_realtime_params_native_for_backend(REALTIME_BACKEND_SORTFORMER);
        params.backend_kind = REALTIME_BACKEND_SORTFORMER;
        params.expected_sample_rate_hz = TARGET_SAMPLE_RATE_HZ;
        Some(params)
    } else {
        None
    };

    let live_config = AudioLiveConfig {
        output_dir,
        session_name,
        capture_device_name,
        bridge_push_samples: LIVE_PUSH_SAMPLES,
        enable_webrtc: webrtc_enabled,
        enable_transcription: true,
        enable_diarization: diarization_enabled,
        write_clean_wav: true,
        write_preview_file: diarization_enabled,
        event_queue_capacity: 0,
        session_params,
        transcription_params,
        transcription_model_path,
        transcription_backend_name: backend_name.clone(),
        diarization_params,
        diarization_model_path,
        diarization_backend_name: if diarization_enabled {
            Some(backend_name)
        } else {
            None
        },
    };

    let api = AudioCaptureApi::load(&runtime_dir)?;
    let live = api.create_live(&live_config)?;
    live.start()?;

    let output_paths = live.output_paths().unwrap_or_else(|_| {
        fallback_output_paths(
            &live_config,
            &fallback_recording_path,
            &fallback_transcript_path,
        )
    });
    let recording_path = if output_paths.cleaned_wav_path.as_os_str().is_empty() {
        fallback_recording_path
    } else {
        output_paths.cleaned_wav_path.clone()
    };
    let transcript_path = if output_paths.transcript_path.as_os_str().is_empty() {
        fallback_transcript_path
    } else {
        output_paths.transcript_path.clone()
    };

    let _ = tx.send(UiMessage::LiveSessionStarted {
        session_id,
        input_device: input_label.clone(),
        recording_path: recording_path.clone(),
        transcript_path: transcript_path.clone(),
    });
    let _ = tx.send(UiMessage::Status(format!(
        "Live transcription started from '{input_label}' using engine audio runtime."
    )));
    if live_config.enable_diarization {
        let _ = tx.send(UiMessage::Status(
            "Live diarization: backend preview markdown will replace the live view as it updates."
                .to_string(),
        ));
    }

    let mut transcript_text = String::new();
    let mut preview_text = String::new();
    let mut diarized_orchestrator =
        diarization_enabled.then(|| DiarizedTranscriptOrchestrator::new(TARGET_SAMPLE_RATE_HZ));
    let mut diarized_preview_active = false;
    let mut stop_called = false;
    let mut idle_after_stop = 0usize;
    let mut terminal_error = None::<String>;

    loop {
        if stop_requested.load(Ordering::Relaxed) && !stop_called {
            live.stop()?;
            stop_called = true;
            let _ = tx.send(UiMessage::Status(
                "Stopping live transcription...".to_string(),
            ));
        }

        let pending = live.wait_events(100)?;
        let events = if pending > 0 {
            live.drain_events(256)?
        } else {
            Vec::new()
        };

        if events.is_empty() {
            if stop_called || terminal_error.is_some() {
                idle_after_stop += 1;
                if idle_after_stop >= 5 {
                    break;
                }
            }
            continue;
        }
        idle_after_stop = 0;

        for event in events {
            handle_live_event(
                &tx,
                session_id,
                &event,
                live_config.enable_diarization,
                diarized_orchestrator.as_mut(),
                &mut transcript_text,
                &mut preview_text,
                &mut diarized_preview_active,
                stop_called,
                &mut terminal_error,
            );
        }

        if terminal_error.is_some() && stop_called {
            break;
        }
    }

    if transcript_path.exists() {
        if let Ok(on_disk) = fs::read_to_string(&transcript_path) {
            if !on_disk.trim().is_empty() {
                transcript_text = on_disk.clone();
                preview_text = on_disk;
            }
        }
    } else if let Some(preview_path) = output_paths.preview_path.as_ref() {
        if let Ok(on_disk) = fs::read_to_string(preview_path) {
            if !on_disk.trim().is_empty() {
                preview_text = on_disk.clone();
                transcript_text = on_disk;
            }
        }
    }

    if transcript_text.trim().is_empty() && !preview_text.trim().is_empty() {
        transcript_text = preview_text.clone();
    }
    if let Some(orchestrator) = diarized_orchestrator.as_ref() {
        let snapshot = orchestrator.final_snapshot();
        if !snapshot.markdown.trim().is_empty() {
            if let Some(preview_path) = output_paths.preview_path.as_ref() {
                let _ = fs::write(preview_path, &snapshot.markdown);
            }
            let _ = fs::write(&transcript_path, &snapshot.markdown);
            preview_text = snapshot.markdown.clone();
            if diarization_enabled || transcript_text.trim().is_empty() {
                transcript_text = snapshot.markdown;
            }
        }
    }

    if let Some(message) = terminal_error {
        let _ = tx.send(UiMessage::Status(format!("Live session error: {message}")));
    }

    if let Ok(mut state) = runtime_state.lock() {
        crate::ensure_output_entry(&mut state.output_entries, transcript_path.clone(), false);
        state.active_audio_path = Some(recording_path.clone());
        if !state
            .media_entries
            .iter()
            .any(|entry| entry.path == recording_path)
        {
            state.media_entries.push(crate::MediaEntry {
                path: recording_path.clone(),
                selected: false,
            });
        }
    }

    let _ = tx.send(UiMessage::LiveSessionFinished {
        session_id,
        input_device: input_label,
        recording_path,
        transcript_path,
        transcript_text,
        preview_text,
    });
    Ok(())
}

fn handle_live_event(
    tx: &mpsc::Sender<UiMessage>,
    session_id: u64,
    event: &AudioSessionEvent,
    diarization_enabled: bool,
    diarized_orchestrator: Option<&mut DiarizedTranscriptOrchestrator>,
    transcript_text: &mut String,
    preview_text: &mut String,
    diarized_preview_active: &mut bool,
    stop_called: bool,
    terminal_error: &mut Option<String>,
) {
    if diarization_enabled {
        if let Some(orchestrator) = diarized_orchestrator {
            let orchestrator_changed = matches!(
                event.kind,
                AUDIO_EVENT_DIARIZATION_SPAN_COMMIT
                    | AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT
                    | AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT
            ) && orchestrator.ingest_event(event);
            if orchestrator_changed {
                let snapshot = orchestrator.snapshot();
                if !snapshot.markdown.trim().is_empty() {
                    *diarized_preview_active = true;
                    *preview_text = snapshot.markdown.clone();
                    *transcript_text = snapshot.markdown.clone();
                    let _ = tx.send(UiMessage::LiveTextSet {
                        session_id,
                        text: snapshot.markdown,
                    });
                }
            }
        }
    }

    match event.kind {
        AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT if diarization_enabled => {
            if !event.text.trim().is_empty() && !*diarized_preview_active {
                *diarized_preview_active = true;
                *transcript_text = event.text.clone();
                *preview_text = event.text.clone();
                let _ = tx.send(UiMessage::LiveTextSet {
                    session_id,
                    text: event.text.clone(),
                });
            }
        }
        AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT if !diarization_enabled => {
            if let Some(chunk) = preview_chunk_text(event) {
                append_live_preview(preview_text, &chunk);
                *transcript_text = preview_text.clone();
                let _ = tx.send(UiMessage::LiveTextAppend { session_id, chunk });
            }
        }
        AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT if diarization_enabled => {
            if !*diarized_preview_active {
                if let Some(chunk) = preview_chunk_text(event) {
                    append_live_preview(preview_text, &chunk);
                    *transcript_text = preview_text.clone();
                    let _ = tx.send(UiMessage::LiveTextAppend { session_id, chunk });
                }
            }
        }
        AUDIO_EVENT_TRANSCRIPTION_STOPPED if !stop_called => {
            let detail = event.detail.trim();
            if !detail.is_empty() {
                let _ = tx.send(UiMessage::Status(format!(
                    "Live runtime transcription stopped unexpectedly: {detail}"
                )));
            } else {
                let _ = tx.send(UiMessage::Status(
                    "Live runtime transcription stopped unexpectedly.".to_string(),
                ));
            }
        }
        AUDIO_EVENT_ERROR => {
            let message = if event.detail.trim().is_empty() {
                event.text.trim().to_string()
            } else {
                event.detail.trim().to_string()
            };
            if !message.is_empty() {
                *terminal_error = Some(message);
            }
        }
        AUDIO_EVENT_NOTICE => {
            if !event.text.trim().is_empty() && !event.detail.trim().is_empty() {
                let _ = tx.send(UiMessage::Status(format!(
                    "Live runtime: {} -> {}",
                    event.text.trim(),
                    event.detail.trim()
                )));
            }
        }
        _ => {}
    }
}

fn fallback_output_paths(
    config: &AudioLiveConfig,
    recording_path: &Path,
    transcript_path: &Path,
) -> AudioLivePaths {
    AudioLivePaths {
        output_dir: config.output_dir.clone(),
        cleaned_wav_path: recording_path.to_path_buf(),
        transcript_path: transcript_path.to_path_buf(),
        preview_path: if config.enable_diarization {
            Some(
                config
                    .output_dir
                    .join(format!("{}.preview.md", config.session_name)),
            )
        } else {
            None
        },
    }
}

fn diarization_model_path(paths: &AppPaths, settings: &AppSettings) -> Result<PathBuf> {
    let model_path = crate::sortformer_model_path_from_settings(paths, settings);
    if !model_path.exists() {
        bail!("diarization model not found: '{}'", model_path.display());
    }
    Ok(model_path)
}

fn resolve_runtime_backend_name(api: &BridgeApi, settings: &AppSettings) -> Result<String> {
    if let Some(gpu_index) = selected_gpu_index_from_settings(settings) {
        if !bridge_has_device_index(api, gpu_index) {
            bail!(
                "selected GPU index {} is not available in runtime device list",
                gpu_index
            );
        }
        return resolve_bridge_device_name_by_index(api, gpu_index)
            .ok_or_else(|| anyhow!("failed to resolve runtime backend device name"));
    }
    Ok("CPU".to_string())
}

fn preview_chunk_text(event: &AudioSessionEvent) -> Option<String> {
    let text = event.text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn append_live_preview(target: &mut String, chunk: &str) {
    let chunk = chunk.trim();
    if chunk.is_empty() {
        return;
    }

    let needs_space = if target.is_empty() || target.ends_with(char::is_whitespace) {
        false
    } else {
        !matches!(
            chunk.chars().next(),
            Some('.' | ',' | ';' | '?' | '!' | ':' | ')' | ']' | '}' | '\'' | '"')
        )
    };

    if needs_space {
        target.push(' ');
    }
    target.push_str(chunk);
}

#[cfg(all(test, target_os = "linux"))]
mod linux_device_tests {
    use super::parse_arecord_device_names;

    #[test]
    fn parses_capture_hints_without_treating_descriptions_as_devices() {
        let output = "pipewire\n    PipeWire Sound Server\ndefault\n    Default ALSA Output\nhw:CARD=sofhdadsp,DEV=0\n    Direct hardware device\n";
        assert_eq!(
            parse_arecord_device_names(output),
            vec!["pipewire", "default", "hw:CARD=sofhdadsp,DEV=0"]
        );
    }
}
