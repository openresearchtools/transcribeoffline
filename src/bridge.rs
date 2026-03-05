use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;

#[repr(C)]
pub struct llama_server_bridge {
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

pub struct BridgeApi {
    runtime_dir: PathBuf,
    _lib: Library,
    default_params: FnDefaultParams,
    default_chat_request: FnDefaultChatRequest,
    default_audio_raw_request: FnDefaultAudioRawRequest,
    empty_vlm_result: FnEmptyVlmResult,
    empty_json_result: FnEmptyJsonResult,
    create: FnCreate,
    destroy: FnDestroy,
    chat_complete: FnChatComplete,
    audio_raw: FnAudioRaw,
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

impl Drop for BridgeHandle<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                (self.api.destroy)(self.ptr);
            }
        }
    }
}

impl BridgeApi {
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
                empty_vlm_result,
                empty_json_result,
                create,
                destroy,
                chat_complete,
                audio_raw,
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
        serde_json::from_str(&json_text).map_err(|e| anyhow!("invalid audio response json: {e}"))
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
