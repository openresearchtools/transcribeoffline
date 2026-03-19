use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use crate::bridge::{
    llama_server_bridge_audio_event, llama_server_bridge_audio_session_params,
    llama_server_bridge_audio_transcription_params, llama_server_bridge_realtime_params,
    AudioSessionEvent,
};

#[repr(C)]
pub struct llama_server_audio_live {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_audio_capture_device_info {
    pub index: i32,
    pub is_default: i32,
    pub name: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_audio_output_paths {
    pub output_dir: *mut c_char,
    pub cleaned_wav_path: *mut c_char,
    pub transcript_path: *mut c_char,
    pub preview_path: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_audio_live_params {
    pub output_dir: *const c_char,
    pub session_name: *const c_char,
    pub capture_device_name: *const c_char,
    pub capture_device_index: i32,
    pub bridge_push_samples: u32,
    pub enable_webrtc: i32,
    pub enable_transcription: i32,
    pub enable_diarization: i32,
    pub write_clean_wav: i32,
    pub write_preview_file: i32,
    pub event_queue_capacity: u32,
    pub session_params: llama_server_bridge_audio_session_params,
    pub transcription_params: llama_server_bridge_audio_transcription_params,
    pub diarization_params: llama_server_bridge_realtime_params,
}

type FnAudioDefaultLiveParams = unsafe extern "C" fn() -> llama_server_audio_live_params;
type FnAudioEmptyOutputPaths = unsafe extern "C" fn() -> llama_server_audio_output_paths;
type FnAudioListCaptureDevices =
    unsafe extern "C" fn(*mut *mut llama_server_audio_capture_device_info, *mut usize) -> i32;
type FnAudioFreeCaptureDevices =
    unsafe extern "C" fn(*mut llama_server_audio_capture_device_info, usize);
type FnAudioLiveCreate =
    unsafe extern "C" fn(*const llama_server_audio_live_params) -> *mut llama_server_audio_live;
type FnAudioLiveDestroy = unsafe extern "C" fn(*mut llama_server_audio_live);
type FnAudioLiveStart = unsafe extern "C" fn(*mut llama_server_audio_live) -> i32;
type FnAudioLiveStop = unsafe extern "C" fn(*mut llama_server_audio_live) -> i32;
type FnAudioLiveWaitEvents = unsafe extern "C" fn(*mut llama_server_audio_live, u32) -> i32;
type FnAudioLiveDrainEvents = unsafe extern "C" fn(
    *mut llama_server_audio_live,
    *mut *mut llama_server_bridge_audio_event,
    *mut usize,
    usize,
) -> i32;
type FnAudioLiveFreeEvents = unsafe extern "C" fn(*mut llama_server_bridge_audio_event, usize);
type FnAudioLiveGetOutputPaths = unsafe extern "C" fn(
    *const llama_server_audio_live,
    *mut llama_server_audio_output_paths,
) -> i32;
type FnAudioOutputPathsFree = unsafe extern "C" fn(*mut llama_server_audio_output_paths);
type FnAudioLiveLastError = unsafe extern "C" fn(*const llama_server_audio_live) -> *const c_char;

#[derive(Debug, Clone)]
pub struct LiveCaptureDeviceInfo {
    pub index: i32,
    pub is_default: bool,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AudioLivePaths {
    pub output_dir: PathBuf,
    pub cleaned_wav_path: PathBuf,
    pub transcript_path: PathBuf,
    pub preview_path: Option<PathBuf>,
}

#[derive(Clone)]
pub struct AudioLiveConfig {
    pub output_dir: PathBuf,
    pub session_name: String,
    pub capture_device_name: Option<String>,
    pub bridge_push_samples: u32,
    pub enable_webrtc: bool,
    pub enable_transcription: bool,
    pub enable_diarization: bool,
    pub write_clean_wav: bool,
    pub write_preview_file: bool,
    pub event_queue_capacity: u32,
    pub session_params: llama_server_bridge_audio_session_params,
    pub transcription_params: llama_server_bridge_audio_transcription_params,
    pub transcription_model_path: String,
    pub transcription_backend_name: String,
    pub diarization_params: Option<llama_server_bridge_realtime_params>,
    pub diarization_model_path: Option<String>,
    pub diarization_backend_name: Option<String>,
}

pub struct AudioCaptureApi {
    runtime_dir: PathBuf,
    _lib: Library,
    default_live_params: FnAudioDefaultLiveParams,
    empty_output_paths: FnAudioEmptyOutputPaths,
    list_capture_devices: FnAudioListCaptureDevices,
    free_capture_devices: FnAudioFreeCaptureDevices,
    live_create: FnAudioLiveCreate,
    live_destroy: FnAudioLiveDestroy,
    live_start: FnAudioLiveStart,
    live_stop: FnAudioLiveStop,
    live_wait_events: FnAudioLiveWaitEvents,
    live_drain_events: FnAudioLiveDrainEvents,
    live_free_events: FnAudioLiveFreeEvents,
    live_get_output_paths: FnAudioLiveGetOutputPaths,
    output_paths_free: FnAudioOutputPathsFree,
    live_last_error: FnAudioLiveLastError,
}

pub struct AudioLiveHandle<'a> {
    api: &'a AudioCaptureApi,
    ptr: *mut llama_server_audio_live,
    _output_dir_c: CString,
    _session_name_c: CString,
    _capture_device_name_c: Option<CString>,
    _transcription_model_path_c: CString,
    _transcription_backend_name_c: CString,
    _diarization_model_path_c: Option<CString>,
    _diarization_backend_name_c: Option<CString>,
}

impl Drop for AudioLiveHandle<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.api.live_destroy)(self.ptr);
            }
        }
    }
}

impl AudioCaptureApi {
    pub fn load(runtime_dir: &Path) -> Result<Self> {
        let runtime_dir = runtime_dir
            .canonicalize()
            .unwrap_or_else(|_| runtime_dir.to_path_buf());
        configure_runtime_loader_paths(&runtime_dir);
        let library_path = audio_library_path(&runtime_dir);
        if !library_path.exists() {
            bail!(
                "missing audio capture library: '{}'",
                library_path.display()
            );
        }

        let lib = unsafe { Library::new(&library_path) }
            .with_context(|| format!("failed to load '{}'", library_path.display()))?;

        unsafe {
            let default_live_params = *lib
                .get::<FnAudioDefaultLiveParams>(b"llama_server_audio_default_live_params\0")
                .context("missing symbol llama_server_audio_default_live_params")?;
            let empty_output_paths = *lib
                .get::<FnAudioEmptyOutputPaths>(b"llama_server_audio_empty_output_paths\0")
                .context("missing symbol llama_server_audio_empty_output_paths")?;
            let list_capture_devices = *lib
                .get::<FnAudioListCaptureDevices>(b"llama_server_audio_list_capture_devices\0")
                .context("missing symbol llama_server_audio_list_capture_devices")?;
            let free_capture_devices = *lib
                .get::<FnAudioFreeCaptureDevices>(b"llama_server_audio_free_capture_devices\0")
                .context("missing symbol llama_server_audio_free_capture_devices")?;
            let live_create = *lib
                .get::<FnAudioLiveCreate>(b"llama_server_audio_live_create\0")
                .context("missing symbol llama_server_audio_live_create")?;
            let live_destroy = *lib
                .get::<FnAudioLiveDestroy>(b"llama_server_audio_live_destroy\0")
                .context("missing symbol llama_server_audio_live_destroy")?;
            let live_start = *lib
                .get::<FnAudioLiveStart>(b"llama_server_audio_live_start\0")
                .context("missing symbol llama_server_audio_live_start")?;
            let live_stop = *lib
                .get::<FnAudioLiveStop>(b"llama_server_audio_live_stop\0")
                .context("missing symbol llama_server_audio_live_stop")?;
            let live_wait_events = *lib
                .get::<FnAudioLiveWaitEvents>(b"llama_server_audio_live_wait_events\0")
                .context("missing symbol llama_server_audio_live_wait_events")?;
            let live_drain_events = *lib
                .get::<FnAudioLiveDrainEvents>(b"llama_server_audio_live_drain_events\0")
                .context("missing symbol llama_server_audio_live_drain_events")?;
            let live_free_events = *lib
                .get::<FnAudioLiveFreeEvents>(b"llama_server_audio_live_free_events\0")
                .context("missing symbol llama_server_audio_live_free_events")?;
            let live_get_output_paths = *lib
                .get::<FnAudioLiveGetOutputPaths>(b"llama_server_audio_live_get_output_paths\0")
                .context("missing symbol llama_server_audio_live_get_output_paths")?;
            let output_paths_free = *lib
                .get::<FnAudioOutputPathsFree>(b"llama_server_audio_output_paths_free\0")
                .context("missing symbol llama_server_audio_output_paths_free")?;
            let live_last_error = *lib
                .get::<FnAudioLiveLastError>(b"llama_server_audio_live_last_error\0")
                .context("missing symbol llama_server_audio_live_last_error")?;

            Ok(Self {
                runtime_dir,
                _lib: lib,
                default_live_params,
                empty_output_paths,
                list_capture_devices,
                free_capture_devices,
                live_create,
                live_destroy,
                live_start,
                live_stop,
                live_wait_events,
                live_drain_events,
                live_free_events,
                live_get_output_paths,
                output_paths_free,
                live_last_error,
            })
        }
    }

    pub fn list_capture_devices(&self) -> Result<Vec<LiveCaptureDeviceInfo>> {
        let mut ptr_devices = ptr::null_mut();
        let mut count = 0usize;
        let rc = self.with_runtime_cwd(|| unsafe {
            (self.list_capture_devices)(&mut ptr_devices, &mut count)
        })?;
        if rc != 0 {
            bail!("llama_server_audio_list_capture_devices failed (rc={rc})");
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let dev = unsafe { &*ptr_devices.add(i) };
            out.push(LiveCaptureDeviceInfo {
                index: dev.index,
                is_default: dev.is_default != 0,
                name: cstr_from_mut(dev.name),
            });
        }
        if !ptr_devices.is_null() {
            unsafe {
                (self.free_capture_devices)(ptr_devices, count);
            }
        }
        Ok(out)
    }

    pub fn create_live<'a>(&'a self, config: &AudioLiveConfig) -> Result<AudioLiveHandle<'a>> {
        let output_dir_c = CString::new(config.output_dir.to_string_lossy().as_ref())
            .context("live output_dir contains NUL byte")?;
        let session_name_c = CString::new(config.session_name.as_str())
            .context("live session_name contains NUL byte")?;
        let capture_device_name_c = match config.capture_device_name.as_deref() {
            Some(value) if !value.trim().is_empty() => {
                Some(CString::new(value).context("live capture_device_name contains NUL byte")?)
            }
            _ => None,
        };
        let transcription_model_path_c = CString::new(config.transcription_model_path.as_str())
            .context("live transcription model path contains NUL byte")?;
        let transcription_backend_name_c = CString::new(config.transcription_backend_name.as_str())
            .context("live transcription backend name contains NUL byte")?;
        let diarization_model_path_c = match config.diarization_model_path.as_deref() {
            Some(value) if !value.trim().is_empty() => {
                Some(CString::new(value).context("live diarization model path contains NUL byte")?)
            }
            _ => None,
        };
        let diarization_backend_name_c = match config.diarization_backend_name.as_deref() {
            Some(value) if !value.trim().is_empty() => Some(
                CString::new(value).context("live diarization backend name contains NUL byte")?,
            ),
            _ => None,
        };

        let mut params = unsafe { (self.default_live_params)() };
        params.output_dir = output_dir_c.as_ptr();
        params.session_name = session_name_c.as_ptr();
        params.capture_device_name = capture_device_name_c
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());
        params.capture_device_index = -1;
        if config.bridge_push_samples > 0 {
            params.bridge_push_samples = config.bridge_push_samples;
        }
        params.enable_webrtc = if config.enable_webrtc { 1 } else { 0 };
        params.enable_transcription = if config.enable_transcription { 1 } else { 0 };
        params.enable_diarization = if config.enable_diarization { 1 } else { 0 };
        params.write_clean_wav = if config.write_clean_wav { 1 } else { 0 };
        params.write_preview_file = if config.write_preview_file { 1 } else { 0 };
        params.event_queue_capacity = config.event_queue_capacity;
        params.session_params = config.session_params;
        params.transcription_params = config.transcription_params;
        params.transcription_params.realtime_params.model_path =
            transcription_model_path_c.as_ptr();
        params.transcription_params.realtime_params.backend_name =
            transcription_backend_name_c.as_ptr();

        if let Some(diarization_params) = config.diarization_params {
            params.diarization_params = diarization_params;
        }
        params.diarization_params.model_path = diarization_model_path_c
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());
        params.diarization_params.backend_name = diarization_backend_name_c
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null());

        let ptr = self.with_runtime_cwd(|| unsafe { (self.live_create)(&params) })?;
        if ptr.is_null() {
            bail!("llama_server_audio_live_create returned null");
        }

        Ok(AudioLiveHandle {
            api: self,
            ptr,
            _output_dir_c: output_dir_c,
            _session_name_c: session_name_c,
            _capture_device_name_c: capture_device_name_c,
            _transcription_model_path_c: transcription_model_path_c,
            _transcription_backend_name_c: transcription_backend_name_c,
            _diarization_model_path_c: diarization_model_path_c,
            _diarization_backend_name_c: diarization_backend_name_c,
        })
    }

    fn with_runtime_cwd<T>(&self, f: impl FnOnce() -> T) -> Result<T> {
        static CWD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = CWD_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock
            .lock()
            .map_err(|_| anyhow!("failed to lock runtime working directory guard"))?;

        let prev = std::env::current_dir().context("failed to read current working directory")?;
        std::env::set_current_dir(&self.runtime_dir).with_context(|| {
            format!(
                "failed to set working directory to runtime dir '{}'",
                self.runtime_dir.display()
            )
        })?;

        struct CwdResetGuard {
            prev: PathBuf,
        }
        impl Drop for CwdResetGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.prev);
            }
        }
        let _reset = CwdResetGuard { prev };

        Ok(f())
    }
}

impl AudioLiveHandle<'_> {
    pub fn start(&self) -> Result<()> {
        let rc = self
            .api
            .with_runtime_cwd(|| unsafe { (self.api.live_start)(self.ptr) })?;
        if rc != 0 {
            bail!(
                "audio live start failed rc={} err='{}'",
                rc,
                self.last_error()
            );
        }
        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        let rc = self
            .api
            .with_runtime_cwd(|| unsafe { (self.api.live_stop)(self.ptr) })?;
        if rc != 0 {
            bail!(
                "audio live stop failed rc={} err='{}'",
                rc,
                self.last_error()
            );
        }
        Ok(())
    }

    pub fn wait_events(&self, timeout_ms: u32) -> Result<i32> {
        let rc = self
            .api
            .with_runtime_cwd(|| unsafe { (self.api.live_wait_events)(self.ptr, timeout_ms) })?;
        if rc < 0 {
            bail!(
                "audio live wait failed rc={} err='{}'",
                rc,
                self.last_error()
            );
        }
        Ok(rc)
    }

    pub fn drain_events(&self, max_events: usize) -> Result<Vec<AudioSessionEvent>> {
        let mut ptr_events = ptr::null_mut();
        let mut count = 0usize;
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.live_drain_events)(self.ptr, &mut ptr_events, &mut count, max_events)
        })?;
        if rc != 0 {
            bail!(
                "audio live drain failed rc={} err='{}'",
                rc,
                self.last_error()
            );
        }

        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let ev = unsafe { &*ptr_events.add(i) };
            out.push(AudioSessionEvent {
                seq_no: ev.seq_no,
                kind: ev.kind,
                flags: ev.flags,
                start_sample: ev.start_sample,
                end_sample: ev.end_sample,
                speaker_id: ev.speaker_id,
                item_id: ev.item_id,
                text: cstr_from_mut(ev.text),
                detail: cstr_from_mut(ev.detail),
            });
        }
        if !ptr_events.is_null() {
            unsafe {
                (self.api.live_free_events)(ptr_events, count);
            }
        }
        Ok(out)
    }

    pub fn output_paths(&self) -> Result<AudioLivePaths> {
        let mut native_paths = unsafe { (self.api.empty_output_paths)() };
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.live_get_output_paths)(self.ptr, &mut native_paths)
        })?;
        if rc != 0 {
            bail!(
                "audio live output path query failed rc={} err='{}'",
                rc,
                self.last_error()
            );
        }

        let paths = AudioLivePaths {
            output_dir: PathBuf::from(cstr_from_mut(native_paths.output_dir)),
            cleaned_wav_path: PathBuf::from(cstr_from_mut(native_paths.cleaned_wav_path)),
            transcript_path: PathBuf::from(cstr_from_mut(native_paths.transcript_path)),
            preview_path: {
                let preview = cstr_from_mut(native_paths.preview_path);
                if preview.trim().is_empty() {
                    None
                } else {
                    Some(PathBuf::from(preview))
                }
            },
        };

        unsafe {
            (self.api.output_paths_free)(&mut native_paths);
        }
        Ok(paths)
    }

    pub fn last_error(&self) -> String {
        cstr_from_const(unsafe { (self.api.live_last_error)(self.ptr) })
    }
}

fn audio_library_path(runtime_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        runtime_dir.join("llama-server-audio.dll")
    } else if cfg!(target_os = "macos") {
        runtime_dir.join("libllama-server-audio.dylib")
    } else {
        runtime_dir.join("libllama-server-audio.so")
    }
}

#[cfg(target_os = "windows")]
fn configure_runtime_loader_paths(runtime_dir: &Path) {
    use std::env;

    let mut dirs = vec![runtime_dir.to_path_buf()];
    dirs.push(runtime_dir.join("vendor").join("ffmpeg").join("bin"));
    dirs.push(runtime_dir.join("vendor").join("ffmpeg"));
    dirs.push(runtime_dir.join("vendor").join("pdfium"));
    dirs.push(runtime_dir.join("vendor").join("cuda"));
    dirs.push(runtime_dir.join("vendor").join("webrtc-audio-processing"));
    dirs.push(
        runtime_dir
            .join("vendor")
            .join("webrtc-audio-processing")
            .join("bin"),
    );

    let mut existing =
        env::split_paths(&env::var_os("PATH").unwrap_or_default()).collect::<Vec<_>>();
    let mut to_prepend = Vec::<PathBuf>::new();
    for dir in dirs {
        if !dir.exists() {
            continue;
        }
        let norm = dir.to_string_lossy().to_ascii_lowercase();
        let already = existing
            .iter()
            .any(|path| path.to_string_lossy().to_ascii_lowercase() == norm)
            || to_prepend
                .iter()
                .any(|path| path.to_string_lossy().to_ascii_lowercase() == norm);
        if !already {
            to_prepend.push(dir);
        }
    }

    if !to_prepend.is_empty() {
        let mut merged = to_prepend;
        merged.append(&mut existing);
        if let Ok(joined) = env::join_paths(merged) {
            env::set_var("PATH", joined);
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn configure_runtime_loader_paths(_runtime_dir: &Path) {}

fn cstr_from_const(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn cstr_from_mut(ptr: *mut c_char) -> String {
    cstr_from_const(ptr as *const c_char)
}
