use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tar::Archive;
use zip::ZipArchive;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use crate::settings::AppPaths;

const DEFAULT_MANIFEST_URL: &str =
    "https://github.com/openresearchtools/engine/releases/latest/download/engine-manifest.json";
const APP_UA: &str = "TranscribeOffline/1.0";

#[derive(Debug, Clone, Deserialize, Serialize)]
struct EngineManifest {
    #[serde(default)]
    tag: String,
    #[serde(default)]
    assets: Vec<ManifestAsset>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ManifestAsset {
    #[serde(default)]
    id: String,
    #[serde(default)]
    platform: String,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    archive: String,
    #[serde(default)]
    file_name: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestSources {
    #[serde(default)]
    sources: Vec<String>,
}

pub fn install_or_repair_runtime_with_backend(
    runtime_dir: &Path,
    paths: &AppPaths,
    preferred_backend: Option<&str>,
    mut on_status: impl FnMut(String),
) -> Result<PathBuf> {
    if runtime_dir.as_os_str().is_empty() {
        bail!("runtime directory is empty");
    }

    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest = load_engine_manifest(&exe_dir, paths)?;
    let mut assets = filtered_assets_for_platform(&manifest);
    if assets.is_empty() {
        bail!(
            "engine manifest for tag '{}' contains no assets for platform '{}'",
            manifest.tag.trim(),
            current_platform_key()
        );
    }
    let preferred_backend = preferred_backend
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty());
    let asset = if let Some(preferred) = preferred_backend {
        if let Some(index) = assets
            .iter()
            .position(|asset| asset.backend.trim().eq_ignore_ascii_case(&preferred))
        {
            assets.remove(index)
        } else {
            let mut available = assets
                .iter()
                .map(|asset| asset.backend.trim().to_ascii_lowercase())
                .filter(|backend| !backend.is_empty())
                .collect::<Vec<_>>();
            available.sort();
            available.dedup();
            let available = if available.is_empty() {
                "<none>".to_string()
            } else {
                available.join(", ")
            };
            bail!(
                "requested backend '{}' is not available for platform '{}' in manifest tag '{}' (available: {})",
                preferred,
                current_platform_key(),
                manifest.tag.trim(),
                available
            );
        }
    } else {
        assets.remove(0)
    };
    on_status(format!(
        "Installing runtime asset: {}",
        describe_asset(&asset)
    ));
    let installed_dir = install_runtime_asset(runtime_dir, &asset, &mut on_status)?;
    on_status("Runtime install finished.".to_string());
    Ok(installed_dir)
}

pub fn available_runtime_backends(paths: &AppPaths) -> Result<Vec<String>> {
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let manifest = load_engine_manifest_local_or_cached(&exe_dir, paths)?;
    let mut out = Vec::<String>::new();
    for asset in filtered_assets_for_platform(&manifest) {
        let backend = asset.backend.trim().to_ascii_lowercase();
        if backend.is_empty() {
            continue;
        }
        if !out.iter().any(|v| v.eq_ignore_ascii_case(&backend)) {
            out.push(backend);
        }
    }
    if out.is_empty() {
        bail!(
            "engine manifest for tag '{}' has no backend entries for platform '{}'",
            manifest.tag.trim(),
            current_platform_key()
        );
    }
    Ok(out)
}

fn load_engine_manifest_local_or_cached(
    exe_dir: &Path,
    paths: &AppPaths,
) -> Result<EngineManifest> {
    let mut errors = Vec::<String>::new();

    for file in local_manifest_file_candidates(exe_dir) {
        if !file.exists() {
            continue;
        }
        match fs::read_to_string(&file)
            .with_context(|| format!("failed reading '{}'", file.display()))
            .and_then(|raw| parse_manifest(&raw))
        {
            Ok(manifest) => return Ok(manifest),
            Err(err) => errors.push(format!("{}: {err}", file.display())),
        }
    }

    let cache_file = cached_manifest_file(paths);
    if cache_file.exists() {
        match fs::read_to_string(&cache_file)
            .with_context(|| format!("failed reading '{}'", cache_file.display()))
            .and_then(|raw| parse_manifest(&raw))
        {
            Ok(manifest) => return Ok(manifest),
            Err(err) => errors.push(format!("{}: {err}", cache_file.display())),
        }
    }

    bail!(
        "failed to load local/cached engine manifest:\n{}",
        errors.join("\n")
    )
}

fn describe_asset(asset: &ManifestAsset) -> String {
    let id = if asset.id.trim().is_empty() {
        asset.file_name.trim()
    } else {
        asset.id.trim()
    };
    let backend = asset.backend.trim();
    let file_name = asset.file_name.trim();
    if backend.is_empty() {
        if file_name.is_empty() {
            id.to_string()
        } else {
            format!("{id} [{file_name}]")
        }
    } else if file_name.is_empty() {
        format!("{id} ({backend})")
    } else {
        format!("{id} ({backend}) [{file_name}]")
    }
}

fn current_platform_key() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows-x64"
    } else if cfg!(target_os = "macos") {
        "macos-arm64"
    } else {
        "ubuntu-x64"
    }
}

fn filtered_assets_for_platform(manifest: &EngineManifest) -> Vec<ManifestAsset> {
    filtered_assets_for_platform_key(manifest, current_platform_key())
}

fn filtered_assets_for_platform_key(
    manifest: &EngineManifest,
    platform: &str,
) -> Vec<ManifestAsset> {
    let mut assets = manifest
        .assets
        .iter()
        .filter(|asset| asset.platform.eq_ignore_ascii_case(platform))
        .cloned()
        .collect::<Vec<_>>();
    assets.sort_by_key(|asset| backend_priority_for_platform(platform, &asset.backend));
    assets
}

fn backend_priority_for_platform(platform: &str, backend: &str) -> usize {
    let p = platform.to_ascii_lowercase();
    let b = backend.to_ascii_lowercase();
    if p == "windows-x64" {
        return match b.as_str() {
            "vulkan" => 0,
            "cuda" => 1,
            _ => 9,
        };
    }
    if p == "macos-arm64" {
        return match b.as_str() {
            "metal" => 0,
            _ => 9,
        };
    }
    match b.as_str() {
        "vulkan" => 0,
        _ => 9,
    }
}

fn local_manifest_file_candidates(exe_dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![exe_dir
        .join("runtime-manifests")
        .join("engine-manifest.json")];

    if let Ok(cwd) = env::current_dir() {
        out.push(cwd.join("runtime-manifests").join("engine-manifest.json"));
    }

    out.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime-manifests")
            .join("engine-manifest.json"),
    );

    out
}

fn cached_manifest_file(paths: &AppPaths) -> PathBuf {
    paths
        .config_dir
        .join("runtime-manifests")
        .join("engine-manifest.json")
}

fn manifest_sources_candidates(exe_dir: &Path, paths: &AppPaths) -> Vec<PathBuf> {
    let mut out = vec![exe_dir
        .join("runtime-manifests")
        .join("engine-manifest-sources.json")];

    if let Ok(cwd) = env::current_dir() {
        out.push(
            cwd.join("runtime-manifests")
                .join("engine-manifest-sources.json"),
        );
    }

    out.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("runtime-manifests")
            .join("engine-manifest-sources.json"),
    );

    out.push(
        paths
            .config_dir
            .join("runtime-manifests")
            .join("engine-manifest-sources.json"),
    );

    out
}

fn load_manifest_sources(exe_dir: &Path, paths: &AppPaths) -> Vec<String> {
    let mut out = vec![DEFAULT_MANIFEST_URL.to_string()];
    for candidate in manifest_sources_candidates(exe_dir, paths) {
        if !candidate.exists() {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&candidate) else {
            continue;
        };
        let Ok(parsed) = serde_json::from_str::<ManifestSources>(&raw) else {
            continue;
        };
        for source in parsed.sources {
            let source = source.trim();
            if source.is_empty() {
                continue;
            }
            if !out.iter().any(|x| x.eq_ignore_ascii_case(source)) {
                out.push(source.to_string());
            }
        }
    }
    out
}

fn parse_manifest(raw: &str) -> Result<EngineManifest> {
    let parsed: EngineManifest =
        serde_json::from_str(raw).context("invalid engine manifest json")?;
    if parsed.assets.is_empty() {
        bail!("engine manifest has no assets");
    }
    Ok(parsed)
}

fn load_engine_manifest(exe_dir: &Path, paths: &AppPaths) -> Result<EngineManifest> {
    let mut errors = Vec::<String>::new();

    for file in local_manifest_file_candidates(exe_dir) {
        if !file.exists() {
            continue;
        }
        match fs::read_to_string(&file)
            .with_context(|| format!("failed reading '{}'", file.display()))
            .and_then(|raw| parse_manifest(&raw))
        {
            Ok(manifest) => return Ok(manifest),
            Err(err) => errors.push(format!("{}: {err}", file.display())),
        }
    }

    let client = Client::builder()
        .user_agent(APP_UA)
        .timeout(Duration::from_secs(25))
        .build()
        .context("failed to build HTTP client")?;

    for url in load_manifest_sources(exe_dir, paths) {
        if url.trim().is_empty() {
            continue;
        }
        let response = client.get(&url).send();
        match response {
            Ok(resp) => {
                let resp = match resp.error_for_status() {
                    Ok(ok) => ok,
                    Err(err) => {
                        errors.push(format!("{url}: {err}"));
                        continue;
                    }
                };
                let text = match resp.text() {
                    Ok(text) => text,
                    Err(err) => {
                        errors.push(format!("{url}: {err}"));
                        continue;
                    }
                };
                match parse_manifest(&text) {
                    Ok(manifest) => {
                        let cache_path = cached_manifest_file(paths);
                        if let Some(parent) = cache_path.parent() {
                            let _ = fs::create_dir_all(parent);
                        }
                        let _ = fs::write(
                            &cache_path,
                            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
                        );
                        return Ok(manifest);
                    }
                    Err(err) => {
                        errors.push(format!("{url}: {err}"));
                    }
                }
            }
            Err(err) => errors.push(format!("{url}: {err}")),
        }
    }

    let cache_file = cached_manifest_file(paths);
    if cache_file.exists() {
        match fs::read_to_string(&cache_file)
            .with_context(|| format!("failed reading '{}'", cache_file.display()))
            .and_then(|raw| parse_manifest(&raw))
        {
            Ok(manifest) => return Ok(manifest),
            Err(err) => errors.push(format!("{}: {err}", cache_file.display())),
        }
    }

    bail!(
        "failed to load engine manifest from local or remote sources:\n{}",
        errors.join("\n")
    )
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut value = bytes as f64;
    let mut idx = 0usize;
    while value >= 1024.0 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{bytes} {}", UNITS[idx])
    } else {
        format!("{value:.1} {}", UNITS[idx])
    }
}

fn download_file_with_progress(
    client: &Client,
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(u64, Option<u64>, f64),
) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating '{}'", parent.display()))?;
    }
    let mut response = client
        .get(url)
        .send()
        .with_context(|| format!("download request failed: {url}"))?
        .error_for_status()
        .with_context(|| format!("download request returned error status: {url}"))?;

    let total = response.content_length();
    let tmp = dest.with_extension("download");
    let mut file = File::create(&tmp)
        .with_context(|| format!("failed to create temporary file '{}'", tmp.display()))?;

    let mut buf = vec![0_u8; 64 * 1024];
    let mut downloaded = 0_u64;
    let started = Instant::now();
    let mut last_emit = Instant::now();

    loop {
        let n = response
            .read(&mut buf)
            .with_context(|| format!("failed reading response body: {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("failed writing '{}'", tmp.display()))?;
        downloaded += n as u64;

        if last_emit.elapsed() >= Duration::from_millis(300) {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            on_progress(downloaded, total, downloaded as f64 / elapsed);
            last_emit = Instant::now();
        }
    }

    file.flush()
        .with_context(|| format!("failed flushing '{}'", tmp.display()))?;
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    on_progress(downloaded, total, downloaded as f64 / elapsed);

    if dest.exists() {
        fs::remove_file(dest)
            .with_context(|| format!("failed replacing existing file '{}'", dest.display()))?;
    }
    fs::rename(&tmp, dest).with_context(|| {
        format!(
            "failed moving downloaded file '{}' to '{}'",
            tmp.display(),
            dest.display()
        )
    })?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed opening '{}'", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed reading '{}'", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_zip_file(zip_path: &Path, out_dir: &Path) -> Result<()> {
    let file =
        File::open(zip_path).with_context(|| format!("failed opening '{}'", zip_path.display()))?;
    let mut archive = ZipArchive::new(file)
        .with_context(|| format!("failed to parse '{}'", zip_path.display()))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .with_context(|| format!("failed reading zip entry #{i}"))?;
        let Some(enclosed) = entry.enclosed_name().map(|p| p.to_path_buf()) else {
            continue;
        };
        let out_path = out_dir.join(enclosed);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)
                .with_context(|| format!("failed creating '{}'", out_path.display()))?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed creating '{}'", parent.display()))?;
        }
        let mut out_file = File::create(&out_path)
            .with_context(|| format!("failed creating '{}'", out_path.display()))?;
        std::io::copy(&mut entry, &mut out_file)
            .with_context(|| format!("failed extracting '{}'", out_path.display()))?;
        #[cfg(unix)]
        if let Some(mode) = entry.unix_mode() {
            let _ = fs::set_permissions(&out_path, fs::Permissions::from_mode(mode));
        }
    }

    Ok(())
}

fn extract_tar_gz_file(tgz_path: &Path, out_dir: &Path) -> Result<()> {
    let file =
        File::open(tgz_path).with_context(|| format!("failed opening '{}'", tgz_path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive
        .unpack(out_dir)
        .with_context(|| format!("failed extracting '{}'", tgz_path.display()))?;
    Ok(())
}

fn flatten_single_nested_root(out_dir: &Path) -> Result<()> {
    let mut entries = fs::read_dir(out_dir)
        .with_context(|| format!("failed listing '{}'", out_dir.display()))?
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy() != ".DS_Store")
        .collect::<Vec<_>>();
    if entries.len() != 1 {
        return Ok(());
    }
    let root = entries.remove(0).path();
    if !root.is_dir() {
        return Ok(());
    }
    for entry in
        fs::read_dir(&root).with_context(|| format!("failed listing '{}'", root.display()))?
    {
        let entry = entry.with_context(|| format!("failed reading '{}'", root.display()))?;
        let from = entry.path();
        let to = out_dir.join(entry.file_name());
        fs::rename(&from, &to)
            .with_context(|| format!("failed moving '{}' to '{}'", from.display(), to.display()))?;
    }
    fs::remove_dir_all(&root).with_context(|| format!("failed removing '{}'", root.display()))?;
    Ok(())
}

fn install_runtime_asset(
    runtime_dir: &Path,
    asset: &ManifestAsset,
    on_status: &mut impl FnMut(String),
) -> Result<PathBuf> {
    if asset.url.trim().is_empty() {
        bail!("selected runtime asset has empty URL");
    }

    let client = Client::builder()
        .user_agent(APP_UA)
        .timeout(Duration::from_secs(1800))
        .build()
        .context("failed building HTTP client")?;

    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_root = env::temp_dir().join(format!("transcribe-offline-runtime-{now_ns}"));
    fs::create_dir_all(&temp_root)
        .with_context(|| format!("failed creating '{}'", temp_root.display()))?;

    let install_result = (|| -> Result<PathBuf> {
        let archive_name = if asset.file_name.trim().is_empty() {
            if asset.archive.eq_ignore_ascii_case("tar.gz") {
                "engine.tar.gz".to_string()
            } else {
                "engine.zip".to_string()
            }
        } else {
            asset.file_name.clone()
        };
        let archive_path = temp_root.join(&archive_name);
        on_status(format!("Downloading runtime: {}", asset.url.trim()));
        download_file_with_progress(&client, &asset.url, &archive_path, |done, total, speed| {
            let status = if let Some(total) = total {
                format!(
                    "Downloading runtime: {} / {} at {}/s",
                    human_bytes(done),
                    human_bytes(total),
                    human_bytes(speed as u64)
                )
            } else {
                format!(
                    "Downloading runtime: {} at {}/s",
                    human_bytes(done),
                    human_bytes(speed as u64)
                )
            };
            on_status(status);
        })?;

        if !asset.sha256.trim().is_empty() {
            let got = sha256_file(&archive_path)?;
            if !got.eq_ignore_ascii_case(asset.sha256.trim()) {
                bail!(
                    "runtime archive SHA256 mismatch for '{}': expected {}, got {}",
                    archive_name,
                    asset.sha256.trim(),
                    got
                );
            }
        }

        if runtime_dir.exists() {
            fs::remove_dir_all(runtime_dir).with_context(|| {
                format!(
                    "failed clearing existing runtime '{}'",
                    runtime_dir.display()
                )
            })?;
        }
        fs::create_dir_all(runtime_dir).with_context(|| {
            format!(
                "failed creating runtime directory '{}'",
                runtime_dir.display()
            )
        })?;

        let archive_kind = if asset.archive.eq_ignore_ascii_case("tar.gz")
            || archive_name.to_ascii_lowercase().ends_with(".tar.gz")
        {
            "tar.gz"
        } else {
            "zip"
        };
        on_status(format!("Extracting runtime archive ({archive_kind})..."));
        match archive_kind {
            "tar.gz" => extract_tar_gz_file(&archive_path, runtime_dir)?,
            _ => extract_zip_file(&archive_path, runtime_dir)?,
        }
        flatten_single_nested_root(runtime_dir)?;
        Ok(runtime_dir.to_path_buf())
    })();

    let _ = fs::remove_dir_all(&temp_root);
    install_result
}

#[cfg(test)]
mod tests {
    use super::{filtered_assets_for_platform_key, EngineManifest, ManifestAsset};

    fn asset(platform: &str, backend: &str, id: &str) -> ManifestAsset {
        ManifestAsset {
            id: id.to_string(),
            platform: platform.to_string(),
            backend: backend.to_string(),
            archive: "zip".to_string(),
            file_name: format!("{id}.zip"),
            url: format!("https://example.invalid/{id}.zip"),
            sha256: String::new(),
        }
    }

    #[test]
    fn selects_windows_asset_order_by_backend_priority() {
        let manifest = EngineManifest {
            tag: "test".to_string(),
            assets: vec![
                asset("windows-x64", "cuda", "win-cuda"),
                asset("windows-x64", "vulkan", "win-vulkan"),
            ],
        };

        let got = filtered_assets_for_platform_key(&manifest, "windows-x64");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].id, "win-vulkan");
        assert_eq!(got[1].id, "win-cuda");
    }

    #[test]
    fn selects_macos_arm64_assets() {
        let manifest = EngineManifest {
            tag: "test".to_string(),
            assets: vec![
                asset("macos-arm64", "metal", "mac-metal"),
                asset("windows-x64", "vulkan", "win-vulkan"),
            ],
        };

        let got = filtered_assets_for_platform_key(&manifest, "macos-arm64");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, "mac-metal");
    }

    #[test]
    fn selects_ubuntu_x64_assets() {
        let manifest = EngineManifest {
            tag: "test".to_string(),
            assets: vec![
                asset("ubuntu-x64", "vulkan", "linux-vulkan"),
                asset("ubuntu-x64", "cpu", "linux-cpu"),
            ],
        };

        let got = filtered_assets_for_platform_key(&manifest, "ubuntu-x64");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].id, "linux-vulkan");
        assert_eq!(got[1].id, "linux-cpu");
    }
}
