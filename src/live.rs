use anyhow::{anyhow, bail, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

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

pub(crate) fn start_live_capture(
    paths: &AppPaths,
    settings: &AppSettings,
    runtime_state: Arc<Mutex<RuntimeState>>,
    tx: mpsc::Sender<UiMessage>,
) -> Result<ActiveLiveCapture> {
    let runtime_dir = PathBuf::from(settings.runtime_dir.trim());
    let bridge_api = BridgeApi::load(&runtime_dir)?;
    let backend_name = resolve_runtime_backend_name(&bridge_api, settings)?;

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

    let session_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    let session_name = format!("live-session-{session_id}");
    let recording_path = paths
        .live_sessions_dir
        .join(format!("{session_name}.clean.wav"));
    let transcript_path = paths
        .live_sessions_dir
        .join(if settings.live_diarization_enabled {
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
    let worker_output_dir = paths.live_sessions_dir.clone();
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
        let snapshot = orchestrator.snapshot();
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
        crate::ensure_output_entry(
            &mut state.output_entries,
            crate::edited_file_path(&transcript_path),
            false,
        );
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
