use anyhow::{anyhow, bail, Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream, StreamConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum PlaybackCommand {
    Load {
        path: PathBuf,
        start_sec: f64,
        autoplay: bool,
        speed: f64,
    },
    PlayAt {
        path: PathBuf,
        start_sec: f64,
    },
    Toggle,
    SeekRelative {
        delta_sec: f64,
    },
    SetSpeed {
        speed: f64,
    },
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum PlaybackEvent {
    State(PlaybackSnapshot),
    Loaded {
        path: PathBuf,
        start_sec: f64,
        autoplay: bool,
    },
    Error {
        path: Option<PathBuf>,
        message: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct PlaybackSnapshot {
    pub path: Option<PathBuf>,
    pub current_sec: f64,
    pub total_sec: f64,
    pub playing: bool,
    pub ended: bool,
    pub loaded: bool,
    pub decoding: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum PlaybackNotice {
    Loaded {
        path: PathBuf,
        start_sec: f64,
        autoplay: bool,
    },
    Error {
        path: Option<PathBuf>,
        message: String,
    },
}

pub(crate) struct PlaybackController {
    command_tx: mpsc::SyncSender<PlaybackCommand>,
    snapshot: Arc<Mutex<PlaybackSnapshot>>,
    notices: Arc<Mutex<VecDeque<PlaybackNotice>>>,
    pub speed: f64,
    pub last_seek_hotkey_at: Option<Instant>,
    pub seek_repeat_count: u32,
}

impl Default for PlaybackController {
    fn default() -> Self {
        let (command_tx, command_rx) = mpsc::sync_channel(64);
        let snapshot = Arc::new(Mutex::new(PlaybackSnapshot::default()));
        let notices = Arc::new(Mutex::new(VecDeque::new()));
        let worker_snapshot = snapshot.clone();
        let worker_notices = notices.clone();
        std::thread::Builder::new()
            .name("linux-playback-frontend".to_string())
            .spawn(move || playback_frontend_worker(command_rx, worker_snapshot, worker_notices))
            .expect("failed to create Linux playback frontend thread");
        Self {
            command_tx,
            snapshot,
            notices,
            speed: 1.0,
            last_seek_hotkey_at: None,
            seek_repeat_count: 0,
        }
    }
}

impl Drop for PlaybackController {
    fn drop(&mut self) {
        let _ = self.command_tx.try_send(PlaybackCommand::Shutdown);
    }
}

impl PlaybackController {
    pub(crate) fn snapshot(&self) -> PlaybackSnapshot {
        self.snapshot
            .try_lock()
            .map(|state| state.clone())
            .unwrap_or_default()
    }

    pub(crate) fn take_notices(&self) -> Vec<PlaybackNotice> {
        let Ok(mut notices) = self.notices.try_lock() else {
            return Vec::new();
        };
        notices.drain(..).collect()
    }

    pub(crate) fn load(&self, path: PathBuf, start_sec: f64, autoplay: bool) -> Result<()> {
        if let Ok(mut snapshot) = self.snapshot.lock() {
            snapshot.decoding = true;
        }
        self.send(PlaybackCommand::Load {
            path,
            start_sec: start_sec.max(0.0),
            autoplay,
            speed: self.speed.clamp(0.1, 4.0),
        })
    }

    pub(crate) fn play_at(&self, path: PathBuf, start_sec: f64) -> Result<()> {
        self.send(PlaybackCommand::PlayAt {
            path,
            start_sec: start_sec.max(0.0),
        })
    }

    pub(crate) fn toggle(&self) -> Result<()> {
        self.send(PlaybackCommand::Toggle)
    }

    pub(crate) fn seek_relative(&self, delta_sec: f64) -> Result<()> {
        self.send(PlaybackCommand::SeekRelative { delta_sec })
    }

    pub(crate) fn set_speed(&self, speed: f64) -> Result<()> {
        self.send(PlaybackCommand::SetSpeed {
            speed: speed.clamp(0.1, 4.0),
        })
    }

    fn send(&self, command: PlaybackCommand) -> Result<()> {
        self.command_tx.try_send(command).map_err(|error| match error {
            mpsc::TrySendError::Full(_) => anyhow!("Linux playback helper command queue is busy"),
            mpsc::TrySendError::Disconnected(_) => {
                anyhow!("Linux playback helper is unavailable")
            }
        })
    }
}

fn playback_frontend_worker(
    commands: mpsc::Receiver<PlaybackCommand>,
    snapshot: Arc<Mutex<PlaybackSnapshot>>,
    notices: Arc<Mutex<VecDeque<PlaybackNotice>>>,
) {
    let helper_exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            push_notice(
                &notices,
                PlaybackNotice::Error {
                    path: None,
                    message: format!("failed to locate playback helper executable: {error}"),
                },
            );
            return;
        }
    };
    let mut child = match Command::new(&helper_exe)
        .arg("--linux-playback-helper")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            push_notice(
                &notices,
                PlaybackNotice::Error {
                    path: None,
                    message: format!(
                        "failed to launch isolated playback helper '{}': {error}",
                        helper_exe.display()
                    ),
                },
            );
            return;
        }
    };

    let Some(mut helper_stdin) = child.stdin.take() else {
        push_notice(
            &notices,
            PlaybackNotice::Error {
                path: None,
                message: "playback helper stdin was unavailable".to_string(),
            },
        );
        let _ = child.kill();
        return;
    };
    let Some(helper_stdout) = child.stdout.take() else {
        push_notice(
            &notices,
            PlaybackNotice::Error {
                path: None,
                message: "playback helper stdout was unavailable".to_string(),
            },
        );
        let _ = child.kill();
        return;
    };
    let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    if let Some(helper_stderr) = child.stderr.take() {
        let stderr_tail_for_thread = stderr_tail.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(helper_stderr).lines().map_while(Result::ok) {
                if let Ok(mut tail) = stderr_tail_for_thread.lock() {
                    tail.push_back(line);
                    while tail.len() > 20 {
                        tail.pop_front();
                    }
                }
            }
        });
    }

    let (event_tx, event_rx) = mpsc::sync_channel(64);
    std::thread::spawn(move || {
        for line in BufReader::new(helper_stdout).lines() {
            let event = match line {
                Ok(line) => serde_json::from_str::<PlaybackEvent>(&line)
                    .map_err(|error| format!("invalid playback helper event: {error}")),
                Err(error) => Err(format!("failed reading playback helper event: {error}")),
            };
            match event {
                Ok(PlaybackEvent::State(state)) => {
                    let _ = event_tx.try_send(Ok(PlaybackEvent::State(state)));
                }
                important => {
                    if event_tx.send(important).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let mut shutdown_requested = false;
    loop {
        while let Ok(event) = event_rx.try_recv() {
            match event {
                Ok(PlaybackEvent::State(next)) => {
                    if let Ok(mut state) = snapshot.lock() {
                        *state = next;
                    }
                }
                Ok(PlaybackEvent::Loaded {
                    path,
                    start_sec,
                    autoplay,
                }) => push_notice(
                    &notices,
                    PlaybackNotice::Loaded {
                        path,
                        start_sec,
                        autoplay,
                    },
                ),
                Ok(PlaybackEvent::Error { path, message }) => {
                    if let Ok(mut state) = snapshot.lock() {
                        state.decoding = false;
                        state.playing = false;
                    }
                    push_notice(&notices, PlaybackNotice::Error { path, message });
                }
                Err(message) => push_notice(
                    &notices,
                    PlaybackNotice::Error {
                        path: None,
                        message,
                    },
                ),
            }
        }

        match commands.recv_timeout(Duration::from_millis(20)) {
            Ok(command) => {
                let is_shutdown = matches!(command, PlaybackCommand::Shutdown);
                let write_result = serde_json::to_writer(&mut helper_stdin, &command)
                    .and_then(|_| helper_stdin.write_all(b"\n").map_err(serde_json::Error::io))
                    .and_then(|_| helper_stdin.flush().map_err(serde_json::Error::io));
                if let Err(error) = write_result {
                    push_notice(
                        &notices,
                        PlaybackNotice::Error {
                            path: None,
                            message: format!("failed to send command to playback helper: {error}"),
                        },
                    );
                    break;
                }
                if is_shutdown {
                    shutdown_requested = true;
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                shutdown_requested = true;
                break;
            }
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                let detail = stderr_tail
                    .lock()
                    .ok()
                    .map(|tail| tail.iter().cloned().collect::<Vec<_>>().join(" | "))
                    .unwrap_or_default();
                let message = if detail.is_empty() {
                    format!("playback helper exited unexpectedly ({status})")
                } else {
                    format!("playback helper exited unexpectedly ({status}): {detail}")
                };
                push_notice(
                    &notices,
                    PlaybackNotice::Error {
                        path: None,
                        message,
                    },
                );
                return;
            }
            Ok(None) => {}
            Err(error) => {
                push_notice(
                    &notices,
                    PlaybackNotice::Error {
                        path: None,
                        message: format!("failed to monitor playback helper: {error}"),
                    },
                );
                break;
            }
        }
    }

    if shutdown_requested {
        let _ = child.wait();
    } else {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn push_notice(notices: &Arc<Mutex<VecDeque<PlaybackNotice>>>, notice: PlaybackNotice) {
    if let Ok(mut queue) = notices.lock() {
        queue.push_back(notice);
        while queue.len() > 32 {
            queue.pop_front();
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

#[derive(Default)]
struct PlaybackEngine {
    stream: Option<Stream>,
    shared: Option<Arc<Mutex<PlaybackBuffer>>>,
    output_sample_rate: u32,
}

impl PlaybackEngine {
    fn ensure_stream(&mut self, audio_errors: mpsc::Sender<String>) -> Result<()> {
        if self.stream.is_some() && self.shared.is_some() {
            return Ok(());
        }
        let shared = Arc::new(Mutex::new(PlaybackBuffer::default()));
        let (stream, rate) = build_stream(shared.clone(), audio_errors)?;
        self.stream = Some(stream);
        self.shared = Some(shared);
        self.output_sample_rate = rate;
        Ok(())
    }

    fn load(
        &mut self,
        path: &Path,
        start_sec: f64,
        autoplay: bool,
        speed: f64,
        audio_errors: mpsc::Sender<String>,
    ) -> Result<()> {
        self.ensure_stream(audio_errors)?;
        let path = normalize_path(path);
        let sample_rate = self.output_sample_rate.max(16_000);
        let data = decode_audio_to_stereo_f32(&path, sample_rate)?;
        let total_frames = data.len() / 2;
        let start_frame = (start_sec.max(0.0) * sample_rate as f64)
            .floor()
            .min(total_frames as f64);
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow!("playback buffer was unavailable"))?;
        let mut inner = shared
            .lock()
            .map_err(|_| anyhow!("playback buffer lock was poisoned"))?;
        inner.samples_stereo_f32 = data;
        inner.total_frames = total_frames;
        inner.position_frames = start_frame;
        inner.playing = autoplay;
        inner.ended = false;
        inner.speed = speed.clamp(0.1, 4.0);
        inner.source_path = Some(path);
        Ok(())
    }

    fn play_at(
        &mut self,
        path: &Path,
        start_sec: f64,
        audio_errors: mpsc::Sender<String>,
    ) -> Result<bool> {
        let path = normalize_path(path);
        if let Some(shared) = self.shared.as_ref() {
            if let Ok(mut inner) = shared.lock() {
                let matches = inner
                    .source_path
                    .as_ref()
                    .map(|loaded| normalize_path(loaded) == path)
                    .unwrap_or(false);
                if matches && inner.total_frames > 0 {
                    let frame = (start_sec.max(0.0) * self.output_sample_rate as f64)
                        .floor()
                        .min(inner.total_frames as f64);
                    inner.position_frames = frame;
                    inner.playing = true;
                    inner.ended = false;
                    return Ok(false);
                }
            }
        }
        self.load(&path, start_sec, true, 1.0, audio_errors)?;
        Ok(true)
    }

    fn toggle(&mut self) -> Result<()> {
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow!("no audio is loaded"))?;
        let mut inner = shared
            .lock()
            .map_err(|_| anyhow!("playback buffer lock was poisoned"))?;
        if inner.total_frames == 0 {
            bail!("no audio is loaded");
        }
        if inner.ended {
            inner.position_frames = 0.0;
            inner.ended = false;
        }
        inner.playing = !inner.playing;
        Ok(())
    }

    fn seek_relative(&mut self, delta_sec: f64) -> Result<()> {
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow!("no audio is loaded"))?;
        let mut inner = shared
            .lock()
            .map_err(|_| anyhow!("playback buffer lock was poisoned"))?;
        if inner.total_frames == 0 {
            bail!("no audio is loaded");
        }
        let delta_frames = delta_sec * self.output_sample_rate as f64;
        inner.position_frames =
            (inner.position_frames + delta_frames).clamp(0.0, inner.total_frames as f64);
        inner.ended = false;
        Ok(())
    }

    fn set_speed(&mut self, speed: f64) -> Result<()> {
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow!("no audio is loaded"))?;
        let mut inner = shared
            .lock()
            .map_err(|_| anyhow!("playback buffer lock was poisoned"))?;
        inner.speed = speed.clamp(0.1, 4.0);
        Ok(())
    }

    fn snapshot(&self, decoding: bool) -> PlaybackSnapshot {
        let Some(shared) = self.shared.as_ref() else {
            return PlaybackSnapshot {
                decoding,
                ..PlaybackSnapshot::default()
            };
        };
        let Ok(inner) = shared.try_lock() else {
            return PlaybackSnapshot {
                decoding,
                ..PlaybackSnapshot::default()
            };
        };
        let rate = self.output_sample_rate.max(1) as f64;
        PlaybackSnapshot {
            path: inner.source_path.clone(),
            current_sec: (inner.position_frames / rate).max(0.0),
            total_sec: inner.total_frames as f64 / rate,
            playing: inner.playing,
            ended: inner.ended,
            loaded: inner.total_frames > 0,
            decoding,
        }
    }
}

pub(crate) fn maybe_run_helper(args: &[String]) -> Option<i32> {
    if args.get(1).map(String::as_str) != Some("--linux-playback-helper") {
        return None;
    }
    match run_helper() {
        Ok(()) => Some(0),
        Err(error) => {
            eprintln!("Linux playback helper failed: {error:#}");
            Some(1)
        }
    }
}

fn run_helper() -> Result<()> {
    let (command_tx, command_rx) = mpsc::channel::<PlaybackCommand>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { break };
            match serde_json::from_str::<PlaybackCommand>(&line) {
                Ok(command) => {
                    if command_tx.send(command).is_err() {
                        break;
                    }
                }
                Err(error) => eprintln!("invalid playback command: {error}"),
            }
        }
    });

    let (audio_error_tx, audio_error_rx) = mpsc::channel::<String>();
    let mut engine = PlaybackEngine::default();
    let mut decoding = false;
    let mut last_state_sent = Instant::now() - Duration::from_secs(1);
    loop {
        while let Ok(message) = audio_error_rx.try_recv() {
            emit_event(&PlaybackEvent::Error {
                path: engine.snapshot(decoding).path,
                message: format!("audio output stream error: {message}"),
            })?;
        }

        match command_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(PlaybackCommand::Load {
                path,
                start_sec,
                autoplay,
                speed,
            }) => {
                decoding = true;
                emit_event(&PlaybackEvent::State(engine.snapshot(decoding)))?;
                let result = engine.load(&path, start_sec, autoplay, speed, audio_error_tx.clone());
                decoding = false;
                match result {
                    Ok(()) => emit_event(&PlaybackEvent::Loaded {
                        path: normalize_path(&path),
                        start_sec,
                        autoplay,
                    })?,
                    Err(error) => emit_event(&PlaybackEvent::Error {
                        path: Some(path),
                        message: error.to_string(),
                    })?,
                }
                emit_event(&PlaybackEvent::State(engine.snapshot(decoding)))?;
            }
            Ok(PlaybackCommand::PlayAt { path, start_sec }) => {
                decoding = true;
                emit_event(&PlaybackEvent::State(engine.snapshot(decoding)))?;
                let result = engine.play_at(&path, start_sec, audio_error_tx.clone());
                decoding = false;
                match result {
                    Ok(loaded) => {
                        if loaded {
                            emit_event(&PlaybackEvent::Loaded {
                                path: normalize_path(&path),
                                start_sec,
                                autoplay: true,
                            })?;
                        }
                    }
                    Err(error) => emit_event(&PlaybackEvent::Error {
                        path: Some(path),
                        message: error.to_string(),
                    })?,
                }
            }
            Ok(PlaybackCommand::Toggle) => {
                if let Err(error) = engine.toggle() {
                    emit_event(&PlaybackEvent::Error {
                        path: engine.snapshot(decoding).path,
                        message: error.to_string(),
                    })?;
                }
            }
            Ok(PlaybackCommand::SeekRelative { delta_sec }) => {
                if let Err(error) = engine.seek_relative(delta_sec) {
                    emit_event(&PlaybackEvent::Error {
                        path: engine.snapshot(decoding).path,
                        message: error.to_string(),
                    })?;
                }
            }
            Ok(PlaybackCommand::SetSpeed { speed }) => {
                if let Err(error) = engine.set_speed(speed) {
                    emit_event(&PlaybackEvent::Error {
                        path: engine.snapshot(decoding).path,
                        message: error.to_string(),
                    })?;
                }
            }
            Ok(PlaybackCommand::Shutdown) => return Ok(()),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
        }

        if last_state_sent.elapsed() >= Duration::from_millis(80) {
            emit_event(&PlaybackEvent::State(engine.snapshot(decoding)))?;
            last_state_sent = Instant::now();
        }
    }
}

fn emit_event(event: &PlaybackEvent) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, event)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn build_stream(
    shared: Arc<Mutex<PlaybackBuffer>>,
    audio_errors: mpsc::Sender<String>,
) -> Result<(Stream, u32)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default audio output device is configured"))?;
    let supported = device
        .default_output_config()
        .context("failed to query the default audio output format")?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.config();
    let channels = config.channels as usize;
    let rate = config.sample_rate.0;
    let err_fn = move |error: cpal::StreamError| {
        let _ = audio_errors.send(error.to_string());
    };
    let stream = match sample_format {
        SampleFormat::F32 => {
            let callback_shared = shared.clone();
            device.build_output_stream(
                &config,
                move |data: &mut [f32], _| write_output_f32(data, channels, &callback_shared),
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let callback_shared = shared.clone();
            device.build_output_stream(
                &config,
                move |data: &mut [i16], _| write_output_i16(data, channels, &callback_shared),
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let callback_shared = shared.clone();
            device.build_output_stream(
                &config,
                move |data: &mut [u16], _| write_output_u16(data, channels, &callback_shared),
                err_fn,
                None,
            )?
        }
        other => bail!("unsupported default output sample format: {other:?}"),
    };
    stream
        .play()
        .context("failed to start the default audio output stream")?;
    Ok((stream, rate))
}

fn write_output_f32(data: &mut [f32], channels: usize, shared: &Arc<Mutex<PlaybackBuffer>>) {
    if let Ok(mut buffer) = shared.try_lock() {
        fill_samples(data, channels, &mut buffer, |dst, value| *dst = value);
    } else {
        data.fill(0.0);
    }
}

fn write_output_i16(data: &mut [i16], channels: usize, shared: &Arc<Mutex<PlaybackBuffer>>) {
    if let Ok(mut buffer) = shared.try_lock() {
        fill_samples(data, channels, &mut buffer, |dst, value| {
            *dst = (value.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        });
    } else {
        data.fill(0);
    }
}

fn write_output_u16(data: &mut [u16], channels: usize, shared: &Arc<Mutex<PlaybackBuffer>>) {
    if let Ok(mut buffer) = shared.try_lock() {
        fill_samples(data, channels, &mut buffer, |dst, value| {
            *dst = (((value.clamp(-1.0, 1.0) * 0.5) + 0.5) * u16::MAX as f32) as u16;
        });
    } else {
        data.fill(u16::MAX / 2);
    }
}

fn fill_samples<T, F>(
    data: &mut [T],
    output_channels: usize,
    buffer: &mut PlaybackBuffer,
    mut assign: F,
) where
    F: FnMut(&mut T, f32),
{
    if !buffer.playing || buffer.total_frames == 0 || output_channels == 0 {
        for sample in data {
            assign(sample, 0.0);
        }
        return;
    }
    for frame in data.chunks_mut(output_channels) {
        let frame_index = buffer.position_frames.floor() as usize;
        if frame_index >= buffer.total_frames {
            buffer.playing = false;
            buffer.ended = true;
            for sample in frame {
                assign(sample, 0.0);
            }
            continue;
        }
        let source = frame_index * 2;
        let left = buffer
            .samples_stereo_f32
            .get(source)
            .copied()
            .unwrap_or(0.0);
        let right = buffer
            .samples_stereo_f32
            .get(source + 1)
            .copied()
            .unwrap_or(left);
        for (channel, sample) in frame.iter_mut().enumerate() {
            assign(sample, if channel % 2 == 0 { left } else { right });
        }
        buffer.position_frames += buffer.speed;
    }
}

fn decode_audio_to_stereo_f32(audio_path: &Path, destination_rate: u32) -> Result<Vec<f32>> {
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
    let media = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = audio_path.extension().and_then(|value| value.to_str()) {
        hint.with_extension(extension);
    }
    let probed = get_probe().format(
        &hint,
        media,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no audio track found in '{}'", audio_path.display()))?;
    let source_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("audio sample rate metadata is missing"))?;
    let track_id = track.id;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut stereo = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => bail!("audio decoder reset was required"),
            Err(error) => return Err(anyhow!("audio demux error: {error}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => bail!("audio decoder reset was required"),
            Err(error) => return Err(anyhow!("audio decode error: {error}")),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let frames = decoded.frames();
        if frames == 0 {
            continue;
        }
        let mut samples = SampleBuffer::<f32>::new(frames as u64, spec);
        samples.copy_interleaved_ref(decoded);
        for frame in samples.samples().chunks(channels) {
            let left = frame.first().copied().unwrap_or(0.0);
            let right = frame.get(1).copied().unwrap_or(left);
            stereo.push(left);
            stereo.push(right);
        }
    }
    if stereo.is_empty() {
        bail!("decoded audio was empty for '{}'", audio_path.display());
    }
    Ok(resample_stereo_linear(
        &stereo,
        source_rate,
        destination_rate.max(8_000),
    ))
}

fn resample_stereo_linear(samples: &[f32], source_rate: u32, destination_rate: u32) -> Vec<f32> {
    if source_rate == 0 || destination_rate == 0 || samples.len() < 4 {
        return samples.to_vec();
    }
    if source_rate == destination_rate {
        return samples.to_vec();
    }
    let input_frames = samples.len() / 2;
    let output_frames = ((input_frames as f64 * destination_rate as f64 / source_rate as f64)
        .round() as usize)
        .max(1);
    let mut output = Vec::with_capacity(output_frames * 2);
    for output_index in 0..output_frames {
        let source_position = output_index as f64 * source_rate as f64 / destination_rate as f64;
        let first = source_position.floor() as usize;
        let second = (first + 1).min(input_frames.saturating_sub(1));
        let fraction = (source_position - first as f64) as f32;
        output.push(samples[first * 2] + (samples[second * 2] - samples[first * 2]) * fraction);
        output.push(
            samples[first * 2 + 1] + (samples[second * 2 + 1] - samples[first * 2 + 1]) * fraction,
        );
    }
    output
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_keeps_stereo_channels_separate() {
        let input = vec![0.0, 1.0, 0.5, 0.5, 1.0, 0.0];
        let output = resample_stereo_linear(&input, 3, 6);
        assert_eq!(output.len(), 12);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 1.0);
        assert!(output[4] > output[0]);
        assert!(output[5] < output[1]);
    }

    #[test]
    fn command_protocol_round_trips_paths_and_playback_mode() {
        let command = PlaybackCommand::Load {
            path: PathBuf::from("/tmp/example audio.mp3"),
            start_sec: 12.5,
            autoplay: true,
            speed: 1.25,
        };
        let encoded = serde_json::to_string(&command).expect("serialize command");
        let decoded: PlaybackCommand = serde_json::from_str(&encoded).expect("parse command");
        match decoded {
            PlaybackCommand::Load {
                path,
                start_sec,
                autoplay,
                speed,
            } => {
                assert_eq!(path, PathBuf::from("/tmp/example audio.mp3"));
                assert_eq!(start_sec, 12.5);
                assert!(autoplay);
                assert_eq!(speed, 1.25);
            }
            _ => panic!("wrong command variant"),
        }
    }
}
