use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use regex::Regex;
use serde_json::{json, Value};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

#[repr(C)]
pub struct llama_server_bridge {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_server_bridge_audio_session {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_params {
    pub model_path: *const c_char,
    pub mmproj_path: *const c_char,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub gpu: i32,
    pub no_kv_offload: i32,
    pub mmproj_use_gpu: i32,
    pub cache_ram_mib: i32,
    pub seed: i32,
    pub ctx_shift: i32,
    pub kv_unified: i32,
    pub devices: *const c_char,
    pub tensor_split: *const c_char,
    pub split_mode: i32,
    pub embedding: i32,
    pub reranking: i32,
    pub pooling_type: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_chat_request {
    pub prompt: *const c_char,
    pub n_predict: i32,
    pub id_slot: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub seed: i32,
    pub repeat_last_n: i32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_raw_request {
    pub audio_bytes: *const u8,
    pub audio_bytes_len: usize,
    pub audio_format: *const c_char,
    pub metadata_json: *const c_char,
    pub ffmpeg_convert: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_realtime_params {
    pub backend_kind: i32,
    pub model_path: *const c_char,
    pub backend_name: *const c_char,
    pub expected_sample_rate_hz: u32,
    pub audio_ring_capacity_samples: u32,
    pub capture_debug: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_session_params {
    pub expected_input_sample_rate_hz: u32,
    pub expected_input_channels: u32,
    pub max_buffered_audio_samples: u32,
    pub event_queue_capacity: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_transcription_params {
    pub bridge_params: llama_server_bridge_params,
    pub metadata_json: *const c_char,
    pub mode: i32,
    pub realtime_params: llama_server_bridge_realtime_params,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_audio_event {
    pub seq_no: u64,
    pub kind: i32,
    pub flags: u32,
    pub start_sample: u64,
    pub end_sample: u64,
    pub speaker_id: i32,
    pub item_id: u32,
    pub text: *mut c_char,
    pub detail: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_vlm_result {
    pub ok: i32,
    pub truncated: i32,
    pub stop: i32,
    pub n_decoded: i32,
    pub n_prompt_tokens: i32,
    pub n_tokens_cached: i32,
    pub eos_reached: i32,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub text: *mut c_char,
    pub error_json: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_json_result {
    pub ok: i32,
    pub status: i32,
    pub json: *mut c_char,
    pub error_json: *mut c_char,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_server_bridge_device_info {
    pub index: i32,
    pub r#type: i32,
    pub memory_free: u64,
    pub memory_total: u64,
    pub backend: *mut c_char,
    pub name: *mut c_char,
    pub description: *mut c_char,
}

type FnDefaultParams = unsafe extern "C" fn() -> llama_server_bridge_params;
type FnDefaultChatRequest = unsafe extern "C" fn() -> llama_server_bridge_chat_request;
type FnDefaultAudioRawRequest = unsafe extern "C" fn() -> llama_server_bridge_audio_raw_request;
type FnDefaultAudioSessionParams =
    unsafe extern "C" fn() -> llama_server_bridge_audio_session_params;
type FnDefaultAudioTranscriptionParams =
    unsafe extern "C" fn() -> llama_server_bridge_audio_transcription_params;
type FnDefaultRealtimeParamsForBackend =
    unsafe extern "C" fn(i32) -> llama_server_bridge_realtime_params;
type FnEmptyVlmResult = unsafe extern "C" fn() -> llama_server_bridge_vlm_result;
type FnEmptyJsonResult = unsafe extern "C" fn() -> llama_server_bridge_json_result;
type FnCreate = unsafe extern "C" fn(*const llama_server_bridge_params) -> *mut llama_server_bridge;
type FnDestroy = unsafe extern "C" fn(*mut llama_server_bridge);
type FnChatComplete = unsafe extern "C" fn(
    *mut llama_server_bridge,
    *const llama_server_bridge_chat_request,
    *mut llama_server_bridge_vlm_result,
) -> i32;
type FnAudioRaw = unsafe extern "C" fn(
    *mut llama_server_bridge,
    *const llama_server_bridge_audio_raw_request,
    *mut llama_server_bridge_json_result,
) -> i32;
type FnAudioSessionCreate = unsafe extern "C" fn(
    *const llama_server_bridge_audio_session_params,
) -> *mut llama_server_bridge_audio_session;
type FnAudioSessionDestroy = unsafe extern "C" fn(*mut llama_server_bridge_audio_session);
type FnAudioSessionPushAudio = unsafe extern "C" fn(
    *mut llama_server_bridge_audio_session,
    *const core::ffi::c_void,
    usize,
    u32,
    u32,
    i32,
) -> i32;
type FnAudioSessionPushEncoded = unsafe extern "C" fn(
    *mut llama_server_bridge_audio_session,
    *const u8,
    usize,
    *const c_char,
) -> i32;
type FnAudioSessionFlushAudio = unsafe extern "C" fn(*mut llama_server_bridge_audio_session) -> i32;
type FnAudioSessionStartDiarization = unsafe extern "C" fn(
    *mut llama_server_bridge_audio_session,
    *const llama_server_bridge_realtime_params,
) -> i32;
type FnAudioSessionStopDiarization =
    unsafe extern "C" fn(*mut llama_server_bridge_audio_session) -> i32;
type FnAudioSessionStartTranscription = unsafe extern "C" fn(
    *mut llama_server_bridge_audio_session,
    *const llama_server_bridge_audio_transcription_params,
) -> i32;
type FnAudioSessionWaitEvents =
    unsafe extern "C" fn(*mut llama_server_bridge_audio_session, u32) -> i32;
type FnAudioSessionDrainEvents = unsafe extern "C" fn(
    *mut llama_server_bridge_audio_session,
    *mut *mut llama_server_bridge_audio_event,
    *mut usize,
    usize,
) -> i32;
type FnAudioSessionFreeEvents = unsafe extern "C" fn(*mut llama_server_bridge_audio_event, usize);
type FnAudioSessionLastError =
    unsafe extern "C" fn(*const llama_server_bridge_audio_session) -> *const c_char;
type FnResultFree = unsafe extern "C" fn(*mut llama_server_bridge_vlm_result);
type FnJsonResultFree = unsafe extern "C" fn(*mut llama_server_bridge_json_result);
type FnLastError = unsafe extern "C" fn(*const llama_server_bridge) -> *const c_char;
type FnListDevices =
    unsafe extern "C" fn(*mut *mut llama_server_bridge_device_info, *mut usize) -> i32;
type FnFreeDevices = unsafe extern "C" fn(*mut llama_server_bridge_device_info, usize);

#[derive(Debug, Clone)]
pub struct SharedBridgeParams {
    pub gpu: Option<i32>,
    pub devices: Option<String>,
    pub tensor_split: Option<String>,
    pub split_mode: i32,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub n_gpu_layers: Option<i32>,
    pub main_gpu: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct ChatRunParams {
    pub model_path: String,
    pub prompt: String,
    pub n_predict: i32,
}

#[derive(Debug, Clone)]
pub struct AudioRunParams {
    pub audio_bytes: Vec<u8>,
    pub audio_format: String,
    pub metadata_json: Value,
    pub ffmpeg_convert: bool,
}

#[derive(Debug, Clone)]
pub struct RealtimeRunParams {
    pub backend_kind: i32,
    pub model_path: String,
    pub backend_name: String,
    pub expected_sample_rate_hz: u32,
}

#[derive(Debug, Clone)]
pub struct AudioSessionParams {
    pub expected_input_sample_rate_hz: u32,
    pub expected_input_channels: u32,
    pub max_buffered_audio_samples: u32,
    pub event_queue_capacity: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AudioSessionEvent {
    pub seq_no: u64,
    pub kind: i32,
    pub flags: u32,
    pub start_sample: u64,
    pub end_sample: u64,
    pub speaker_id: i32,
    pub item_id: u32,
    pub text: String,
    pub detail: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: i32,
    pub backend: String,
    pub name: String,
    pub description: String,
    pub memory_free: u64,
    pub memory_total: u64,
}

pub const REALTIME_BACKEND_SORTFORMER: i32 = 1;
pub const REALTIME_BACKEND_VOXTRAL: i32 = 2;
pub const AUDIO_SAMPLE_FORMAT_S16: i32 = 2;
pub const AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE: i32 = 0;
pub const AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE: i32 = 1;
pub const AUDIO_EVENT_NOTICE: i32 = 0;
pub const AUDIO_EVENT_DIARIZATION_SPAN_COMMIT: i32 = 3;
pub const AUDIO_EVENT_DIARIZATION_STOPPED: i32 = 2;
pub const AUDIO_EVENT_DIARIZATION_TRANSCRIPT_COMMIT: i32 = 4;
pub const AUDIO_EVENT_TRANSCRIPTION_PIECE_COMMIT: i32 = 8;
pub const AUDIO_EVENT_TRANSCRIPTION_WORD_COMMIT: i32 = 9;
pub const AUDIO_EVENT_TRANSCRIPTION_RESULT_JSON: i32 = 10;
pub const AUDIO_EVENT_TRANSCRIPTION_STOPPED: i32 = 11;
pub const AUDIO_EVENT_STREAM_FLUSHED: i32 = 12;
pub const AUDIO_EVENT_ERROR: i32 = 13;
pub const AUDIO_EVENT_FLAG_FINAL: u32 = 1u32 << 0;
pub const AUDIO_EVENT_FLAG_FROM_BUFFER_REPLAY: u32 = 1u32 << 1;
pub const AUDIO_EVENT_FLAG_PREVIEW: u32 = 1u32 << 2;
pub const AUDIO_EVENT_FLAG_SNAPSHOT_START: u32 = 1u32 << 3;
pub const AUDIO_EVENT_FLAG_SNAPSHOT_END: u32 = 1u32 << 4;

pub struct BridgeApi {
    runtime_dir: PathBuf,
    _lib: Library,
    default_params: FnDefaultParams,
    default_chat_request: FnDefaultChatRequest,
    default_audio_raw_request: FnDefaultAudioRawRequest,
    default_audio_session_params: FnDefaultAudioSessionParams,
    default_audio_transcription_params: FnDefaultAudioTranscriptionParams,
    default_realtime_params_for_backend: FnDefaultRealtimeParamsForBackend,
    empty_vlm_result: FnEmptyVlmResult,
    empty_json_result: FnEmptyJsonResult,
    create: FnCreate,
    destroy: FnDestroy,
    chat_complete: FnChatComplete,
    audio_raw: FnAudioRaw,
    audio_session_create: FnAudioSessionCreate,
    audio_session_destroy: FnAudioSessionDestroy,
    audio_session_push_audio: FnAudioSessionPushAudio,
    audio_session_push_encoded: FnAudioSessionPushEncoded,
    audio_session_flush_audio: FnAudioSessionFlushAudio,
    audio_session_start_diarization: FnAudioSessionStartDiarization,
    audio_session_stop_diarization: FnAudioSessionStopDiarization,
    audio_session_start_transcription: FnAudioSessionStartTranscription,
    audio_session_wait_events: FnAudioSessionWaitEvents,
    audio_session_drain_events: FnAudioSessionDrainEvents,
    audio_session_free_events: FnAudioSessionFreeEvents,
    audio_session_last_error: FnAudioSessionLastError,
    result_free: FnResultFree,
    json_result_free: FnJsonResultFree,
    last_error: FnLastError,
    #[allow(dead_code)]
    list_devices: FnListDevices,
    #[allow(dead_code)]
    free_devices: FnFreeDevices,
}

struct BridgeHandle<'a> {
    api: &'a BridgeApi,
    ptr: *mut llama_server_bridge,
}

pub struct AudioSessionHandle<'a> {
    api: &'a BridgeApi,
    ptr: *mut llama_server_bridge_audio_session,
}

impl Drop for BridgeHandle<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.api.destroy)(self.ptr);
            }
        }
    }
}

impl Drop for AudioSessionHandle<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.api.audio_session_destroy)(self.ptr);
            }
        }
    }
}

impl BridgeApi {
    pub fn default_audio_session_params_native(&self) -> llama_server_bridge_audio_session_params {
        unsafe { (self.default_audio_session_params)() }
    }

    pub fn default_audio_transcription_params_native(
        &self,
    ) -> llama_server_bridge_audio_transcription_params {
        unsafe { (self.default_audio_transcription_params)() }
    }

    pub fn default_realtime_params_native_for_backend(
        &self,
        backend_kind: i32,
    ) -> llama_server_bridge_realtime_params {
        unsafe { (self.default_realtime_params_for_backend)(backend_kind) }
    }

    pub fn load(runtime_dir: &Path) -> Result<Self> {
        let runtime_dir = runtime_dir
            .canonicalize()
            .unwrap_or_else(|_| runtime_dir.to_path_buf());
        configure_runtime_loader_paths(&runtime_dir);
        let library_path = bridge_library_path(&runtime_dir);
        if !library_path.exists() {
            bail!("missing bridge library: '{}'", library_path.display());
        }

        let lib = unsafe { Library::new(&library_path) }
            .with_context(|| format!("failed to load '{}'", library_path.display()))?;

        unsafe {
            let default_params = *lib
                .get::<FnDefaultParams>(b"llama_server_bridge_default_params\0")
                .context("missing symbol llama_server_bridge_default_params")?;
            let default_chat_request = *lib
                .get::<FnDefaultChatRequest>(b"llama_server_bridge_default_chat_request\0")
                .context("missing symbol llama_server_bridge_default_chat_request")?;
            let default_audio_raw_request = *lib
                .get::<FnDefaultAudioRawRequest>(b"llama_server_bridge_default_audio_raw_request\0")
                .context("missing symbol llama_server_bridge_default_audio_raw_request")?;
            let default_audio_session_params = *lib
                .get::<FnDefaultAudioSessionParams>(
                    b"llama_server_bridge_default_audio_session_params\0",
                )
                .context("missing symbol llama_server_bridge_default_audio_session_params")?;
            let default_audio_transcription_params = *lib
                .get::<FnDefaultAudioTranscriptionParams>(
                    b"llama_server_bridge_default_audio_transcription_params\0",
                )
                .context("missing symbol llama_server_bridge_default_audio_transcription_params")?;
            let default_realtime_params_for_backend = *lib
                .get::<FnDefaultRealtimeParamsForBackend>(
                    b"llama_server_bridge_default_realtime_params_for_backend\0",
                )
                .context(
                    "missing symbol llama_server_bridge_default_realtime_params_for_backend",
                )?;
            let empty_vlm_result = *lib
                .get::<FnEmptyVlmResult>(b"llama_server_bridge_empty_vlm_result\0")
                .context("missing symbol llama_server_bridge_empty_vlm_result")?;
            let empty_json_result = *lib
                .get::<FnEmptyJsonResult>(b"llama_server_bridge_empty_json_result\0")
                .context("missing symbol llama_server_bridge_empty_json_result")?;
            let create = *lib
                .get::<FnCreate>(b"llama_server_bridge_create\0")
                .context("missing symbol llama_server_bridge_create")?;
            let destroy = *lib
                .get::<FnDestroy>(b"llama_server_bridge_destroy\0")
                .context("missing symbol llama_server_bridge_destroy")?;
            let chat_complete = *lib
                .get::<FnChatComplete>(b"llama_server_bridge_chat_complete\0")
                .context("missing symbol llama_server_bridge_chat_complete")?;
            let audio_raw = *lib
                .get::<FnAudioRaw>(b"llama_server_bridge_audio_transcriptions_raw\0")
                .context("missing symbol llama_server_bridge_audio_transcriptions_raw")?;
            let audio_session_create = *lib
                .get::<FnAudioSessionCreate>(b"llama_server_bridge_audio_session_create\0")
                .context("missing symbol llama_server_bridge_audio_session_create")?;
            let audio_session_destroy = *lib
                .get::<FnAudioSessionDestroy>(b"llama_server_bridge_audio_session_destroy\0")
                .context("missing symbol llama_server_bridge_audio_session_destroy")?;
            let audio_session_push_audio = *lib
                .get::<FnAudioSessionPushAudio>(b"llama_server_bridge_audio_session_push_audio\0")
                .context("missing symbol llama_server_bridge_audio_session_push_audio")?;
            let audio_session_push_encoded = *lib
                .get::<FnAudioSessionPushEncoded>(
                    b"llama_server_bridge_audio_session_push_encoded\0",
                )
                .context("missing symbol llama_server_bridge_audio_session_push_encoded")?;
            let audio_session_flush_audio = *lib
                .get::<FnAudioSessionFlushAudio>(b"llama_server_bridge_audio_session_flush_audio\0")
                .context("missing symbol llama_server_bridge_audio_session_flush_audio")?;
            let audio_session_start_diarization = *lib
                .get::<FnAudioSessionStartDiarization>(
                    b"llama_server_bridge_audio_session_start_diarization\0",
                )
                .context("missing symbol llama_server_bridge_audio_session_start_diarization")?;
            let audio_session_stop_diarization = *lib
                .get::<FnAudioSessionStopDiarization>(
                    b"llama_server_bridge_audio_session_stop_diarization\0",
                )
                .context("missing symbol llama_server_bridge_audio_session_stop_diarization")?;
            let audio_session_start_transcription = *lib
                .get::<FnAudioSessionStartTranscription>(
                    b"llama_server_bridge_audio_session_start_transcription\0",
                )
                .context("missing symbol llama_server_bridge_audio_session_start_transcription")?;
            let audio_session_wait_events = *lib
                .get::<FnAudioSessionWaitEvents>(b"llama_server_bridge_audio_session_wait_events\0")
                .context("missing symbol llama_server_bridge_audio_session_wait_events")?;
            let audio_session_drain_events = *lib
                .get::<FnAudioSessionDrainEvents>(
                    b"llama_server_bridge_audio_session_drain_events\0",
                )
                .context("missing symbol llama_server_bridge_audio_session_drain_events")?;
            let audio_session_free_events = *lib
                .get::<FnAudioSessionFreeEvents>(b"llama_server_bridge_audio_session_free_events\0")
                .context("missing symbol llama_server_bridge_audio_session_free_events")?;
            let audio_session_last_error = *lib
                .get::<FnAudioSessionLastError>(b"llama_server_bridge_audio_session_last_error\0")
                .context("missing symbol llama_server_bridge_audio_session_last_error")?;
            let result_free = *lib
                .get::<FnResultFree>(b"llama_server_bridge_result_free\0")
                .context("missing symbol llama_server_bridge_result_free")?;
            let json_result_free = *lib
                .get::<FnJsonResultFree>(b"llama_server_bridge_json_result_free\0")
                .context("missing symbol llama_server_bridge_json_result_free")?;
            let last_error = *lib
                .get::<FnLastError>(b"llama_server_bridge_last_error\0")
                .context("missing symbol llama_server_bridge_last_error")?;
            let list_devices = *lib
                .get::<FnListDevices>(b"llama_server_bridge_list_devices\0")
                .context("missing symbol llama_server_bridge_list_devices")?;
            let free_devices = *lib
                .get::<FnFreeDevices>(b"llama_server_bridge_free_devices\0")
                .context("missing symbol llama_server_bridge_free_devices")?;

            Ok(Self {
                runtime_dir,
                _lib: lib,
                default_params,
                default_chat_request,
                default_audio_raw_request,
                default_audio_session_params,
                default_audio_transcription_params,
                default_realtime_params_for_backend,
                empty_vlm_result,
                empty_json_result,
                create,
                destroy,
                chat_complete,
                audio_raw,
                audio_session_create,
                audio_session_destroy,
                audio_session_push_audio,
                audio_session_push_encoded,
                audio_session_flush_audio,
                audio_session_start_diarization,
                audio_session_stop_diarization,
                audio_session_start_transcription,
                audio_session_wait_events,
                audio_session_drain_events,
                audio_session_free_events,
                audio_session_last_error,
                result_free,
                json_result_free,
                last_error,
                list_devices,
                free_devices,
            })
        }
    }

    #[allow(dead_code)]
    pub fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
        let mut ptr_devices = ptr::null_mut();
        let mut count = 0usize;
        let rc =
            self.with_runtime_cwd(|| unsafe { (self.list_devices)(&mut ptr_devices, &mut count) })?;
        if rc != 0 {
            bail!("llama_server_bridge_list_devices failed (rc={rc})");
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let dev = unsafe { &*ptr_devices.add(i) };
            out.push(DeviceInfo {
                index: dev.index,
                backend: cstr_from_mut(dev.backend),
                name: cstr_from_mut(dev.name),
                description: cstr_from_mut(dev.description),
                memory_free: dev.memory_free,
                memory_total: dev.memory_total,
            });
        }
        unsafe {
            (self.free_devices)(ptr_devices, count);
        }
        Ok(out)
    }

    pub fn run_chat(&self, shared: &SharedBridgeParams, run: &ChatRunParams) -> Result<String> {
        if run.model_path.trim().is_empty() {
            bail!("chat model path is empty");
        }
        if run.prompt.trim().is_empty() {
            bail!("chat prompt is empty");
        }

        let bridge = self.create_bridge(shared, &run.model_path, false)?;
        let prompt_c =
            CString::new(run.prompt.as_str()).context("chat prompt contains NUL byte")?;

        let mut req = unsafe { (self.default_chat_request)() };
        req.prompt = prompt_c.as_ptr();
        req.n_predict = run.n_predict;
        req.id_slot = -1;
        req.temperature = 0.0;
        req.top_p = 1.0;
        req.top_k = -1;
        req.min_p = 0.0;
        req.seed = -1;

        let mut out = unsafe { (self.empty_vlm_result)() };
        let rc = unsafe { (self.chat_complete)(bridge.ptr, &req, &mut out) };
        let text = cstr_from_mut(out.text);
        let out_err = cstr_from_mut(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err = cstr_from_const(unsafe { (self.last_error)(bridge.ptr) });
            unsafe {
                (self.result_free)(&mut out);
            }
            bail!(
                "chat failed rc={} ok={} bridge_err='{}' out_err='{}'",
                rc,
                out.ok,
                bridge_err,
                out_err
            );
        }

        unsafe {
            (self.result_free)(&mut out);
        }
        Ok(text)
    }

    pub fn run_audio_raw(
        &self,
        shared: &SharedBridgeParams,
        run: &AudioRunParams,
    ) -> Result<Value> {
        if run.audio_bytes.is_empty() {
            bail!("audio bytes are empty");
        }
        if run.audio_format.trim().is_empty() {
            bail!("audio format is empty");
        }
        let bridge = self.create_bridge(shared, "", true)?;
        let format_c =
            CString::new(run.audio_format.as_str()).context("audio format contains NUL byte")?;
        let metadata_c = CString::new(run.metadata_json.to_string())
            .context("audio metadata contains NUL byte")?;

        let mut req = unsafe { (self.default_audio_raw_request)() };
        req.audio_bytes = run.audio_bytes.as_ptr();
        req.audio_bytes_len = run.audio_bytes.len();
        req.audio_format = format_c.as_ptr();
        req.metadata_json = metadata_c.as_ptr();
        req.ffmpeg_convert = if run.ffmpeg_convert { 1 } else { 0 };

        let mut out = unsafe { (self.empty_json_result)() };
        let rc = unsafe { (self.audio_raw)(bridge.ptr, &req, &mut out) };
        let json_text = cstr_from_mut(out.json);
        let out_err = cstr_from_mut(out.error_json);
        if rc != 0 || out.ok == 0 {
            let bridge_err = cstr_from_const(unsafe { (self.last_error)(bridge.ptr) });
            unsafe {
                (self.json_result_free)(&mut out);
            }
            bail!(
                "audio failed rc={} status={} bridge_err='{}' out_err='{}'",
                rc,
                out.status,
                bridge_err,
                out_err
            );
        }
        unsafe {
            (self.json_result_free)(&mut out);
        }
        match serde_json::from_str::<Value>(&json_text) {
            Ok(parsed) => Ok(parsed),
            Err(parse_err) => {
                // Keep transcription flow alive even if the runtime returns malformed/non-UTF JSON.
                // The caller already has deterministic fallback output path logic.
                let mut fallback = serde_json::Map::new();
                fallback.insert(
                    "_bridge_json_parse_warning".to_string(),
                    Value::String(format!(
                        "audio response JSON parse failed; fallback mode enabled: {parse_err}"
                    )),
                );
                if let Some(path) = extract_output_path_from_jsonish(&json_text) {
                    fallback.insert("output".to_string(), json!({ "path": path }));
                }
                Ok(Value::Object(fallback))
            }
        }
    }

    pub fn create_audio_session<'a>(
        &'a self,
        params: &AudioSessionParams,
    ) -> Result<AudioSessionHandle<'a>> {
        let mut session_params = unsafe { (self.default_audio_session_params)() };
        session_params.expected_input_sample_rate_hz = params.expected_input_sample_rate_hz;
        session_params.expected_input_channels = params.expected_input_channels;
        session_params.max_buffered_audio_samples = params.max_buffered_audio_samples;
        session_params.event_queue_capacity = params.event_queue_capacity;

        let ptr =
            self.with_runtime_cwd(|| unsafe { (self.audio_session_create)(&session_params) })?;
        if ptr.is_null() {
            bail!("llama_server_bridge_audio_session_create returned null");
        }
        Ok(AudioSessionHandle { api: self, ptr })
    }

    fn make_realtime_params(
        &self,
        run: &RealtimeRunParams,
    ) -> Result<(llama_server_bridge_realtime_params, CString, CString)> {
        if run.model_path.trim().is_empty() {
            bail!("realtime model path is empty");
        }
        if run.backend_name.trim().is_empty() {
            bail!("realtime backend name is empty");
        }
        let model_c = CString::new(run.model_path.as_str())
            .context("realtime model path contains NUL byte")?;
        let backend_c = CString::new(run.backend_name.as_str())
            .context("realtime backend name contains NUL byte")?;
        let mut params = unsafe { (self.default_realtime_params_for_backend)(run.backend_kind) };
        params.backend_kind = run.backend_kind;
        params.model_path = model_c.as_ptr();
        params.backend_name = backend_c.as_ptr();
        params.expected_sample_rate_hz = run.expected_sample_rate_hz;
        Ok((params, model_c, backend_c))
    }

    fn create_bridge<'a>(
        &'a self,
        shared: &SharedBridgeParams,
        model_path: &str,
        audio_only_mode: bool,
    ) -> Result<BridgeHandle<'a>> {
        let model_c = CString::new(model_path).context("model path contains NUL byte")?;
        let devices_c = match shared.devices.as_deref() {
            Some(v) if !v.trim().is_empty() => {
                Some(CString::new(v).context("devices contains NUL byte")?)
            }
            _ => None,
        };
        let tensor_split_c = match shared.tensor_split.as_deref() {
            Some(v) if !v.trim().is_empty() => {
                Some(CString::new(v).context("tensor_split contains NUL byte")?)
            }
            _ => None,
        };
        if shared.gpu.is_some() && devices_c.is_some() {
            bail!("invalid bridge params: 'gpu' and 'devices' cannot both be set");
        }

        let mut params = unsafe { (self.default_params)() };
        params.model_path = model_c.as_ptr();
        params.mmproj_path = ptr::null();
        params.n_ctx = shared.n_ctx;
        params.n_batch = shared.n_batch;
        params.n_ubatch = shared.n_ubatch;
        params.n_parallel = shared.n_parallel;
        if let Some(v) = shared.n_threads {
            params.n_threads = v;
        }
        if let Some(v) = shared.n_threads_batch {
            params.n_threads_batch = v;
        }
        if let Some(v) = shared.n_gpu_layers {
            params.n_gpu_layers = v;
        }
        if let Some(v) = shared.main_gpu {
            params.main_gpu = v;
        }
        if let Some(v) = shared.gpu {
            params.gpu = v;
        }
        params.devices = devices_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
        params.tensor_split = tensor_split_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
        params.split_mode = shared.split_mode;
        params.embedding = 0;
        params.reranking = 0;

        let prev_audio_only_env = if audio_only_mode {
            std::env::var("LLAMA_SERVER_AUDIO_ONLY").ok()
        } else {
            None
        };
        if audio_only_mode {
            std::env::set_var("LLAMA_SERVER_AUDIO_ONLY", "1");
        }
        let ptr = self.with_runtime_cwd(|| unsafe { (self.create)(&params) })?;
        if audio_only_mode {
            if let Some(v) = prev_audio_only_env {
                std::env::set_var("LLAMA_SERVER_AUDIO_ONLY", v);
            } else {
                std::env::remove_var("LLAMA_SERVER_AUDIO_ONLY");
            }
        }
        if ptr.is_null() {
            bail!("llama_server_bridge_create returned null");
        }
        Ok(BridgeHandle { api: self, ptr })
    }

    fn with_runtime_cwd<T>(&self, f: impl FnOnce() -> T) -> Result<T> {
        use std::sync::{Mutex, OnceLock};

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

impl AudioSessionHandle<'_> {
    pub fn start_diarization(&self, run: &RealtimeRunParams) -> Result<()> {
        let (params, _model_c, _backend_c) = self.api.make_realtime_params(run)?;
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_start_diarization)(self.ptr, &params)
        })?;
        if rc != 0 {
            bail!(
                "audio session diarization start failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn stop_diarization(&self) -> Result<()> {
        let rc = self
            .api
            .with_runtime_cwd(|| unsafe { (self.api.audio_session_stop_diarization)(self.ptr) })?;
        if rc != 0 {
            bail!(
                "audio session diarization stop failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn start_transcription_realtime(&self, run: &RealtimeRunParams) -> Result<()> {
        let (realtime_params, _model_c, _backend_c) = self.api.make_realtime_params(run)?;
        let mut params = unsafe { (self.api.default_audio_transcription_params)() };
        params.mode = AUDIO_TRANSCRIPTION_MODE_REALTIME_NATIVE;
        params.realtime_params = realtime_params;
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_start_transcription)(self.ptr, &params)
        })?;
        if rc != 0 {
            bail!(
                "audio session transcription start failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn start_transcription_offline(
        &self,
        shared: &SharedBridgeParams,
        metadata_json: &Value,
    ) -> Result<()> {
        let devices_c = match shared.devices.as_deref() {
            Some(v) if !v.trim().is_empty() => {
                Some(CString::new(v).context("devices contains NUL byte")?)
            }
            _ => None,
        };
        let tensor_split_c = match shared.tensor_split.as_deref() {
            Some(v) if !v.trim().is_empty() => {
                Some(CString::new(v).context("tensor_split contains NUL byte")?)
            }
            _ => None,
        };
        if shared.gpu.is_some() && devices_c.is_some() {
            bail!("invalid bridge params: 'gpu' and 'devices' cannot both be set");
        }
        let metadata_c = CString::new(metadata_json.to_string())
            .context("audio transcription metadata contains NUL byte")?;

        let mut params = unsafe { (self.api.default_audio_transcription_params)() };
        params.mode = AUDIO_TRANSCRIPTION_MODE_OFFLINE_ROUTE;
        params.metadata_json = metadata_c.as_ptr();
        params.bridge_params.n_ctx = shared.n_ctx;
        params.bridge_params.n_batch = shared.n_batch;
        params.bridge_params.n_ubatch = shared.n_ubatch;
        params.bridge_params.n_parallel = shared.n_parallel;
        if let Some(v) = shared.n_threads {
            params.bridge_params.n_threads = v;
        }
        if let Some(v) = shared.n_threads_batch {
            params.bridge_params.n_threads_batch = v;
        }
        if let Some(v) = shared.n_gpu_layers {
            params.bridge_params.n_gpu_layers = v;
        }
        if let Some(v) = shared.main_gpu {
            params.bridge_params.main_gpu = v;
        }
        if let Some(v) = shared.gpu {
            params.bridge_params.gpu = v;
        }
        params.bridge_params.devices = devices_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
        params.bridge_params.tensor_split = tensor_split_c
            .as_ref()
            .map(|v| v.as_ptr())
            .unwrap_or(ptr::null());
        params.bridge_params.split_mode = shared.split_mode;
        params.bridge_params.embedding = 0;
        params.bridge_params.reranking = 0;

        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_start_transcription)(self.ptr, &params)
        })?;
        if rc != 0 {
            bail!(
                "audio session transcription start failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn push_audio_s16(
        &self,
        samples: &[i16],
        sample_rate_hz: u32,
        channels: u32,
    ) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let frame_count = samples.len() / channels.max(1) as usize;
        if frame_count == 0 {
            return Ok(());
        }
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_push_audio)(
                self.ptr,
                samples.as_ptr() as *const core::ffi::c_void,
                frame_count,
                sample_rate_hz,
                channels,
                AUDIO_SAMPLE_FORMAT_S16,
            )
        })?;
        if rc != 0 {
            bail!(
                "audio session push failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn push_encoded(&self, audio_bytes: &[u8], audio_format: &str) -> Result<()> {
        if audio_bytes.is_empty() {
            bail!("encoded audio bytes are empty");
        }
        if audio_format.trim().is_empty() {
            bail!("encoded audio format is empty");
        }
        let audio_format_c =
            CString::new(audio_format).context("encoded audio format contains NUL byte")?;
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_push_encoded)(
                self.ptr,
                audio_bytes.as_ptr(),
                audio_bytes.len(),
                audio_format_c.as_ptr(),
            )
        })?;
        if rc != 0 {
            bail!(
                "audio session encoded push failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn flush_audio(&self) -> Result<()> {
        let rc = self
            .api
            .with_runtime_cwd(|| unsafe { (self.api.audio_session_flush_audio)(self.ptr) })?;
        if rc != 0 {
            bail!(
                "audio session flush failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(())
    }

    pub fn wait_events(&self, timeout_ms: u32) -> Result<i32> {
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_wait_events)(self.ptr, timeout_ms)
        })?;
        if rc < 0 {
            bail!(
                "audio session wait failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
            );
        }
        Ok(rc)
    }

    pub fn drain_events(&self, max_events: usize) -> Result<Vec<AudioSessionEvent>> {
        let mut ptr_events = ptr::null_mut();
        let mut count = 0usize;
        let rc = self.api.with_runtime_cwd(|| unsafe {
            (self.api.audio_session_drain_events)(self.ptr, &mut ptr_events, &mut count, max_events)
        })?;
        if rc != 0 {
            bail!(
                "audio session drain failed rc={} err='{}'",
                rc,
                cstr_from_const(unsafe { (self.api.audio_session_last_error)(self.ptr) })
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
                (self.api.audio_session_free_events)(ptr_events, count);
            }
        }
        Ok(out)
    }
}

fn bridge_library_path(runtime_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        runtime_dir.join("llama-server-bridge.dll")
    } else if cfg!(target_os = "macos") {
        runtime_dir.join("libllama-server-bridge.dylib")
    } else {
        runtime_dir.join("libllama-server-bridge.so")
    }
}

fn cstr_from_const(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn cstr_from_mut(ptr: *mut c_char) -> String {
    cstr_from_const(ptr as *const c_char)
}

fn extract_output_path_from_jsonish(raw: &str) -> Option<String> {
    static OUTPUT_PATH_RE: OnceLock<Regex> = OnceLock::new();
    static PATH_RE: OnceLock<Regex> = OnceLock::new();

    let output_path_re = OUTPUT_PATH_RE.get_or_init(|| {
        Regex::new(r#""output"\s*:\s*\{[^{}]*"path"\s*:\s*"(?P<path>(?:\\.|[^"\\])*)""#)
            .expect("valid output path regex")
    });
    if let Some(path) = capture_json_string_group(output_path_re, raw, "path") {
        return Some(path);
    }

    let path_re = PATH_RE.get_or_init(|| {
        Regex::new(r#""path"\s*:\s*"(?P<path>(?:\\.|[^"\\])*)""#).expect("valid path regex")
    });
    capture_json_string_group(path_re, raw, "path")
}

fn capture_json_string_group(re: &Regex, raw: &str, group: &str) -> Option<String> {
    let caps = re.captures(raw)?;
    let escaped = caps.name(group)?.as_str();
    serde_json::from_str::<String>(&format!("\"{escaped}\"")).ok()
}

#[cfg(test)]
mod tests {
    use super::extract_output_path_from_jsonish;

    #[test]
    fn extracts_nested_output_path() {
        let raw = r#"{"ok":1,"output":{"path":"/tmp/output-file.md"}}"#;
        let path = extract_output_path_from_jsonish(raw).expect("path should parse");
        assert_eq!(path, "/tmp/output-file.md");
    }

    #[test]
    fn extracts_flat_path_when_output_object_missing() {
        let raw = r#"{"path":"C:\\tmp\\output-file.md","status":200}"#;
        let path = extract_output_path_from_jsonish(raw).expect("path should parse");
        assert_eq!(path, r#"C:\tmp\output-file.md"#);
    }
}

#[cfg(windows)]
fn configure_runtime_loader_paths(runtime_dir: &Path) {
    use std::collections::HashSet;
    use std::iter;
    use std::os::windows::ffi::OsStrExt;

    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x00001000;
    const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x00000400;

    unsafe extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut core::ffi::c_void;
        fn SetDllDirectoryW(path_name: *const u16) -> i32;
    }

    fn wide(path: &Path) -> Vec<u16> {
        path.as_os_str()
            .encode_wide()
            .chain(iter::once(0))
            .collect()
    }

    let mut dirs = vec![runtime_dir.to_path_buf()];
    dirs.push(runtime_dir.join("vendor").join("ffmpeg").join("bin"));
    dirs.push(runtime_dir.join("vendor").join("ffmpeg"));
    dirs.push(runtime_dir.join("vendor").join("pdfium"));
    dirs.push(runtime_dir.join("vendor").join("cuda"));

    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for dir in dirs {
        let abs = dir.canonicalize().unwrap_or(dir);
        if !abs.exists() {
            continue;
        }
        let key = abs.to_string_lossy().to_string();
        if seen.insert(key) {
            deduped.push(abs);
        }
    }

    unsafe {
        let _ = SetDefaultDllDirectories(
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
        );
    }
    for dir in &deduped {
        let wide_dir = wide(dir);
        unsafe {
            let cookie = AddDllDirectory(wide_dir.as_ptr());
            if cookie.is_null() {
                let _ = SetDllDirectoryW(wide_dir.as_ptr());
            }
        }
    }

    if !deduped.is_empty() {
        let prefix = deduped
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(";");
        let current = std::env::var("PATH").unwrap_or_default();
        let next = if current.is_empty() {
            prefix
        } else {
            format!("{prefix};{current}")
        };
        std::env::set_var("PATH", next);
    }
}

#[cfg(not(windows))]
fn configure_runtime_loader_paths(_runtime_dir: &Path) {}
