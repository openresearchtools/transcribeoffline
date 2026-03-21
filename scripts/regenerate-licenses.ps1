param(
  [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
  $RepoRoot = (Resolve-Path $RepoRoot).Path
}

function Write-Utf8File {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [Parameter(Mandatory = $true)][string]$Content
  )

  $parent = Split-Path -Parent $Path
  if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }

  Set-Content -LiteralPath $Path -Value $Content -Encoding utf8
}

function Escape-MdCell {
  param([AllowNull()][string]$Text)

  if ([string]::IsNullOrWhiteSpace($Text)) {
    return "-"
  }

  $escaped = $Text -replace "\|", "\\|" -replace "`r?`n", " "
  if ([string]::IsNullOrWhiteSpace($escaped)) {
    return "-"
  }
  return $escaped
}

function Read-TextFileSafe {
  param(
    [Parameter(Mandatory = $true)][string]$Path
  )

  try {
    return Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
  } catch {
    return "[unreadable text file: $Path]"
  }
}

function Fetch-TextFromUrl {
  param(
    [Parameter(Mandatory = $true)][string]$Url
  )

  $ProgressPreference = "SilentlyContinue"
  $resp = Invoke-WebRequest `
    -Uri $Url `
    -UseBasicParsing `
    -Headers @{ "User-Agent" = "transcribe-offline-license-generator" } `
    -ErrorAction Stop
  return [string]$resp.Content
}

function Append-LicenseTextSection {
  param(
    [Parameter(Mandatory = $true)][System.Text.StringBuilder]$Builder,
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$Body,
    [string]$HeadingPrefix = "###"
  )

  [void]$Builder.AppendLine("$HeadingPrefix $Title")
  [void]$Builder.AppendLine("")
  [void]$Builder.AppendLine("~~~~text")
  [void]$Builder.AppendLine($Body.TrimEnd())
  [void]$Builder.AppendLine("~~~~")
  [void]$Builder.AppendLine("")
}

function Extract-BetweenMarkers {
  param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][string]$BeginMarker,
    [Parameter(Mandatory = $true)][string]$EndMarker
  )

  $pattern = "(?s)$([regex]::Escape($BeginMarker)).*?$([regex]::Escape($EndMarker))"
  $match = [regex]::Match($Text, $pattern)
  if ($match.Success) {
    return $match.Value
  }
  return ""
}

function Append-EngineLicenseSection {
  param(
    [Parameter(Mandatory = $true)][System.Text.StringBuilder]$Builder,
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Entries
  )

  [void]$Builder.AppendLine("## $Title")
  [void]$Builder.AppendLine("")

  if (@($Entries).Count -eq 0) {
    [void]$Builder.AppendLine("_None found in runtime snapshot._")
    [void]$Builder.AppendLine("")
    return
  }

  $entriesByPath = @($Entries) |
    Sort-Object RelativePath |
    Group-Object RelativePath |
    ForEach-Object { $_.Group[0] }

  foreach ($entry in $entriesByPath) {
    Append-LicenseTextSection `
      -Builder $Builder `
      -Title $entry.RelativePath `
      -Body (Read-TextFileSafe -Path $entry.FullPath) `
      -HeadingPrefix "###"
  }
}

function Get-RelativePath {
  param(
    [Parameter(Mandatory = $true)][string]$BasePath,
    [Parameter(Mandatory = $true)][string]$TargetPath
  )

  $baseFull = [System.IO.Path]::GetFullPath($BasePath)
  if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
    $baseFull += [System.IO.Path]::DirectorySeparatorChar
  }

  $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
  $baseUri = New-Object System.Uri($baseFull)
  $targetUri = New-Object System.Uri($targetFull)
  $relativeUri = $baseUri.MakeRelativeUri($targetUri)
  $relativePath = [System.Uri]::UnescapeDataString($relativeUri.ToString())
  return ($relativePath -replace "/", "\")
}

function Get-LicenseCandidateRelativePaths {
  param(
    [Parameter(Mandatory = $true)][string]$PackageRoot,
    [AllowNull()][string]$DeclaredLicenseFile
  )

  $paths = New-Object System.Collections.Generic.List[string]
  $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

  if (-not [string]::IsNullOrWhiteSpace($DeclaredLicenseFile)) {
    $declaredPath = Join-Path $PackageRoot $DeclaredLicenseFile
    if (Test-Path -LiteralPath $declaredPath -PathType Leaf) {
      $declaredRel = Get-RelativePath -BasePath $PackageRoot -TargetPath (Resolve-Path -LiteralPath $declaredPath).Path
      if ($seen.Add($declaredRel)) {
        $paths.Add($declaredRel) | Out-Null
      }
    }
  }

  $nameRegex = '^(LICENSE(|[-_.].*)|LICENCE(|[-_.].*)|COPYING(|[-_.].*)|NOTICE(|[-_.].*)|UNLICENSE(|[-_.].*)|COPYRIGHT(|[-_.].*))$'
  $searchRoots = New-Object System.Collections.Generic.List[string]
  $searchRoots.Add($PackageRoot) | Out-Null

  foreach ($subdir in @("license", "licenses", "LICENCE", "LICENCES", "copying", "COPYING", "notice", "NOTICE")) {
    $candidate = Join-Path $PackageRoot $subdir
    if (Test-Path -LiteralPath $candidate -PathType Container) {
      $searchRoots.Add($candidate) | Out-Null
    }
  }

  foreach ($root in $searchRoots) {
    $files = Get-ChildItem -LiteralPath $root -File -Recurse -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -imatch $nameRegex }
    foreach ($file in $files) {
      $relativePath = Get-RelativePath -BasePath $PackageRoot -TargetPath $file.FullName
      if ($seen.Add($relativePath)) {
        $paths.Add($relativePath) | Out-Null
      }
    }
  }

  $paths.Sort([System.StringComparer]::OrdinalIgnoreCase)
  return ,$paths
}

Push-Location $RepoRoot
try {
  $metadataRaw = & cargo metadata --format-version 1 --locked --quiet
  if ($LASTEXITCODE -ne 0) {
    throw "cargo metadata failed with exit code $LASTEXITCODE"
  }
} finally {
  Pop-Location
}

$metadata = $metadataRaw | ConvertFrom-Json
if ($null -eq $metadata.resolve -or [string]::IsNullOrWhiteSpace([string]$metadata.resolve.root)) {
  throw "cargo metadata did not return a dependency resolve graph."
}

$rootId = [string]$metadata.resolve.root
$nodeById = @{}
foreach ($node in $metadata.resolve.nodes) {
  $nodeById[[string]$node.id] = $node
}

$packageById = @{}
foreach ($pkg in $metadata.packages) {
  $packageById[[string]$pkg.id] = $pkg
}

if (-not $packageById.ContainsKey($rootId)) {
  throw "Unable to locate workspace root package in metadata: $rootId"
}

$visited = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
$stack = New-Object System.Collections.Generic.Stack[string]
$stack.Push($rootId)
while ($stack.Count -gt 0) {
  $current = $stack.Pop()
  if (-not $visited.Add($current)) {
    continue
  }
  if ($nodeById.ContainsKey($current)) {
    foreach ($dep in $nodeById[$current].dependencies) {
      $stack.Push([string]$dep)
    }
  }
}

$registryPrefix = "registry+https://github.com/rust-lang/crates.io-index"
$dependencyPackages = @()
foreach ($id in $visited) {
  if ($id -eq $rootId) {
    continue
  }
  if (-not $packageById.ContainsKey($id)) {
    continue
  }
  $pkg = $packageById[$id]
  if ($null -eq $pkg.source) {
    continue
  }
  if (-not ([string]$pkg.source).StartsWith($registryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    continue
  }
  $dependencyPackages += $pkg
}

$dependencyPackages = $dependencyPackages | Sort-Object name, version

$licensesDir = Join-Path $RepoRoot "licenses"
if (Test-Path -LiteralPath $licensesDir) {
  Remove-Item -LiteralPath $licensesDir -Recurse -Force
}

$upstreamDir = Join-Path $licensesDir "upstream"
$rustCratesDir = Join-Path $upstreamDir "rust-crates"
New-Item -ItemType Directory -Force -Path $rustCratesDir | Out-Null
$engineUpstreamDir = Join-Path $upstreamDir "openresearchtools-engine"
New-Item -ItemType Directory -Force -Path $engineUpstreamDir | Out-Null

$officialEngineFiles = @(
  [pscustomobject]@{
    Name = "LICENSES.md"
    Url = "https://raw.githubusercontent.com/openresearchtools/engine/main/third_party/LICENSES.md"
  },
  [pscustomobject]@{
    Name = "README.md"
    Url = "https://raw.githubusercontent.com/openresearchtools/engine/main/third_party/README.md"
  }
)

$officialEngineFetchRows = @()
foreach ($spec in $officialEngineFiles) {
  $target = Join-Path $engineUpstreamDir $spec.Name
  $ok = $false
  $errorText = ""
  try {
    $content = Fetch-TextFromUrl -Url $spec.Url
    Write-Utf8File -Path $target -Content $content
    $ok = $true
  } catch {
    $errorText = $_.Exception.Message
  }
  $officialEngineFetchRows += [pscustomobject]@{
    Name = $spec.Name
    Url = $spec.Url
    Path = $target
    Ok = $ok
    Error = $errorText
  }
}

$rows = @()
$manualReviewCrates = New-Object System.Collections.Generic.List[string]

foreach ($pkg in $dependencyPackages) {
  $pkgName = [string]$pkg.name
  $pkgVersion = [string]$pkg.version
  $manifestPath = [string]$pkg.manifest_path
  $pkgRoot = Split-Path -Parent $manifestPath
  $declaredLicenseFile = if ($pkg.license_file) { [string]$pkg.license_file } else { "" }
  $relativeLicensePaths = Get-LicenseCandidateRelativePaths -PackageRoot $pkgRoot -DeclaredLicenseFile $declaredLicenseFile

  $crateOutDir = Join-Path $rustCratesDir ("{0}-{1}" -f $pkgName, $pkgVersion)
  New-Item -ItemType Directory -Force -Path $crateOutDir | Out-Null

  $copied = New-Object System.Collections.Generic.List[string]
  foreach ($relativePath in $relativeLicensePaths) {
    $sourceFile = Join-Path $pkgRoot $relativePath
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
      continue
    }

    $targetFile = Join-Path $crateOutDir ($relativePath -replace "/", "\")
    $targetParent = Split-Path -Parent $targetFile
    if ($targetParent -and -not (Test-Path -LiteralPath $targetParent)) {
      New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
    }
    Copy-Item -LiteralPath $sourceFile -Destination $targetFile -Force
    $copied.Add(($relativePath -replace "\\", "/")) | Out-Null
  }

  $declaredLicenseExpression = if ($pkg.license) { [string]$pkg.license } else { "UNKNOWN" }
  if ($copied.Count -eq 0) {
    $fallbackFileName = "LICENSE-DECLARED.txt"
    $fallbackText = @(
      "crate: $pkgName",
      "version: $pkgVersion",
      "declared_license_expression: $declaredLicenseExpression",
      "note: no standalone license file was found in the published crate package.",
      "crates_io: https://crates.io/crates/$pkgName/$pkgVersion"
    ) -join "`n"
    Write-Utf8File -Path (Join-Path $crateOutDir $fallbackFileName) -Content $fallbackText
    $copied.Add($fallbackFileName) | Out-Null
  }

  if ($declaredLicenseExpression -eq "UNKNOWN") {
    $manualReviewCrates.Add(("{0}-{1}" -f $pkgName, $pkgVersion)) | Out-Null
  }

  $sourceTxtLines = @(
    "crate: $pkgName",
    "version: $pkgVersion",
    "source: $($pkg.source)",
    "manifest_path: $manifestPath",
    "license_expression: $declaredLicenseExpression",
    "declared_license_file: $declaredLicenseFile",
    "copied_license_files: $($copied -join '; ')",
    "crates_io: https://crates.io/crates/$pkgName/$pkgVersion"
  )
  if ($pkg.repository) {
    $sourceTxtLines += "repository: $($pkg.repository)"
  }
  Write-Utf8File -Path (Join-Path $crateOutDir "SOURCE.txt") -Content ($sourceTxtLines -join "`n")

  $rows += [pscustomobject]@{
    Name = $pkgName
    Version = $pkgVersion
    LicenseExpression = $declaredLicenseExpression
    DeclaredLicenseFile = if ($declaredLicenseFile) { $declaredLicenseFile } else { "-" }
    CopiedLicenseFiles = if ($copied.Count -gt 0) { ($copied -join "; ") } else { "-" }
    Source = if ($pkg.source) { [string]$pkg.source } else { "-" }
    Repository = if ($pkg.repository) { [string]$pkg.repository } else { "-" }
    CrateDir = $crateOutDir
  }
}

$rows = $rows | Sort-Object Name, Version
$generatedAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")
$rootPackage = $packageById[$rootId]

$licenseStats = $rows |
  Group-Object LicenseExpression |
  Sort-Object -Property @{ Expression = "Count"; Descending = $true }, @{ Expression = "Name"; Descending = $false }
$licenseStatLines = @()
foreach ($stat in $licenseStats) {
  $licenseStatLines += ("- {0}: {1} crate(s)" -f $stat.Name, $stat.Count)
}
if ($licenseStatLines.Count -eq 0) {
  $licenseStatLines += "- none"
}

$directDependencyLines = @()
if ($nodeById.ContainsKey($rootId)) {
  $directRows = @()
  foreach ($depId in $nodeById[$rootId].dependencies) {
    $depKey = [string]$depId
    if (-not $packageById.ContainsKey($depKey)) {
      continue
    }
    $depPkg = $packageById[$depKey]
    if ($null -eq $depPkg.source) {
      continue
    }
    $directRows += [pscustomobject]@{
      Name = [string]$depPkg.name
      Version = [string]$depPkg.version
      License = if ($depPkg.license) { [string]$depPkg.license } else { "UNKNOWN" }
    }
  }
  $directRows = $directRows | Sort-Object Name, Version -Unique
  foreach ($row in $directRows) {
    $directDependencyLines += ("- {0} {1}: {2}" -f $row.Name, $row.Version, $row.License)
  }
}
if ($directDependencyLines.Count -eq 0) {
  $directDependencyLines += "- none"
}

$reviewLines = @()
foreach ($item in ($manualReviewCrates | Sort-Object)) {
  $reviewLines += ("- {0}" -f $item)
}
if ($reviewLines.Count -eq 0) {
  $reviewLines += "- none"
}

$licenseIndex = @(
  "# License Index (Transcribe Offline)",
  "",
  "## Bundled outputs",
  "",
  "1. THIRD_PARTY_NOTICES_ALL.md",
  "2. THIRD_PARTY_LICENSES_ALL.md",
  "3. ENGINE_THIRD_PARTY_LICENSES_FULL.md",
  "",
  "These are the markdown files bundled into the app binary.",
  "",
  "## Folder contents",
  "",
  "- Consolidated notice page (app + models + engine): THIRD_PARTY_NOTICES_ALL.md",
  "- Full app licenses per package (with full text): THIRD_PARTY_LICENSES_ALL.md",
  "- Full engine licenses per package/file (with full text): ENGINE_THIRD_PARTY_LICENSES_FULL.md",
  "",
  "## Temporary generation inputs",
  "",
  "- Full app licenses per package before bundling: APP_THIRD_PARTY_LICENSES_FULL.md",
  "- Rust dependency inventory: RUST_DEPENDENCIES_FULL.md",
  "- Official engine snapshot files: upstream/openresearchtools-engine/*",
  "- Rust crate license files copied from local Cargo registry: upstream/rust-crates/*",
  "- Model notices and citations: MODEL_NOTICES_AND_CITATIONS.md",
  "",
  "## Engine license source",
  "",
  "Engine legal documents are generated from the repository-tracked snapshot files under:",
  "- upstream/openresearchtools-engine/*",
  "",
  "## Legacy archive",
  "",
  "The previous GTK-era snapshot is preserved at ../legacy-licenses."
) -join "`n"
Write-Utf8File -Path (Join-Path $licensesDir "README.md") -Content $licenseIndex

$thirdPartyNotices = @(
  "Transcribe Offline is distributed under the MIT License.",
  "Full third-party license texts are available in the in-app Third-party licenses view.",
  "Model notices and citations are listed below."
) -join "`n"
Write-Utf8File -Path (Join-Path $licensesDir "THIRD_PARTY_NOTICES.md") -Content $thirdPartyNotices

$keyLicenses = @(
  "# Key Licenses (Transcribe Offline)",
  "",
  "## Wrapper app",
  "",
  ("- Package: {0}" -f $rootPackage.name),
  ("- Version: {0}" -f $rootPackage.version),
  ("- License: {0}" -f $rootPackage.license),
  "- License file: ../LICENSE",
  "",
  "## Direct Rust dependencies",
  ""
) + $directDependencyLines + @(
  "",
  "## Transitive Rust license-expression summary",
  ""
) + $licenseStatLines + @(
  "",
  "## Engine components",
  "",
  "Engine third-party licenses are captured in ENGINE_THIRD_PARTY_LICENSES_FULL.md from repository-tracked upstream snapshots."
)
Write-Utf8File -Path (Join-Path $licensesDir "KEY_LICENSES.md") -Content ($keyLicenses -join "`n")

$rustDepsDoc = @(
  "# Rust Dependencies (Full)",
  "",
  ("- Root package: {0} {1}" -f $rootPackage.name, $rootPackage.version),
  ("- Total crates from crates.io: {0}" -f $rows.Count),
  "",
  "## Crates requiring manual license review (unknown license expression)",
  ""
) + $reviewLines + @(
  "",
  "## Dependency table",
  "",
  "| Crate | Version | License expression | Declared license_file | Copied license files |",
  "| --- | --- | --- | --- | --- |"
)

foreach ($row in $rows) {
  $rustDepsDoc += ("| {0} | {1} | {2} | {3} | {4} |" -f `
    (Escape-MdCell $row.Name), `
    (Escape-MdCell $row.Version), `
    (Escape-MdCell $row.LicenseExpression), `
    (Escape-MdCell $row.DeclaredLicenseFile), `
    (Escape-MdCell $row.CopiedLicenseFiles))
}

Write-Utf8File -Path (Join-Path $licensesDir "RUST_DEPENDENCIES_FULL.md") -Content ($rustDepsDoc -join "`n")

$upstreamReadme = @(
  "# Upstream License Sources",
  "",
  "## Rust crates",
  "",
  "Per-crate folders under rust-crates/ contain:",
  "",
  "- SOURCE.txt with package provenance.",
  "- copied license-related files found from crate source and license_file metadata.",
  "",
  "Source packages come from crates.io registry entries resolved by Cargo lockfile.",
  "",
  "## Openresearchtools-Engine official snapshots",
  "",
  "Files fetched from official engine repository (main branch):"
)
foreach ($row in $officialEngineFetchRows) {
  if ($row.Ok) {
    $upstreamReadme += ("- {0} (from {1})" -f $row.Name, $row.Url)
  } else {
    $upstreamReadme += ("- {0} fetch failed: {1}" -f $row.Name, $row.Error)
  }
}
$upstreamReadme = ($upstreamReadme -join "`n")
Write-Utf8File -Path (Join-Path $upstreamDir "README.md") -Content $upstreamReadme

$modelNoticesOut = Join-Path $licensesDir "MODEL_NOTICES_AND_CITATIONS.md"
$modelNotices = @(
  "# Model Notices and Citations",
  "",
  "Citations below are upstream original model/research citations.",
  "Non-endorsement: inclusion, conversion, or runtime use in Transcribe Offline does not imply endorsement, sponsorship, affiliation, or approval by original authors/maintainers/institutions.",
  "Runtime/framework package licenses are listed in Engine licenses.",
  "Converted/interoperability artifacts in this app are not official upstream model releases.",
  "",
  "## Whisper models",
  "",
  "- Download sources used by app (converted artifacts):",
  "  - https://huggingface.co/openresearchtools/whisper-large-v3-turbo-GGML",
  "  - https://huggingface.co/openresearchtools/whisper-large-v3-GGML",
  "- Original upstream model card references:",
  "  - https://huggingface.co/openai/whisper-large-v3-turbo",
  "  - https://huggingface.co/openai/whisper-large-v3",
  "- Model card license references at time of writing:",
  "  - whisper-large-v3-turbo: mit",
  "  - whisper-large-v3: apache-2.0",
  "- Runtime artifact format: GGML.",
  "- Conversion notice: distributed artifacts are conversion/interoperability outputs and may differ from original upstream packaging.",
  "- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.",
  "- Users are responsible for complying with upstream model terms.",
  "",
  "## Voxtral realtime live transcription models",
  "",
  "- Download source used by app (converted artifacts): https://huggingface.co/openresearchtools/Voxtral-Mini-4B-Realtime-2602",
  "- Original upstream model card reference: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602",
  "- Model card license reference at time of writing: apache-2.0.",
  "- Runtime artifact format: GGUF.",
  "- Conversion notice: distributed artifacts are conversion/interoperability outputs.",
  "- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.",
  "",
  "## NVIDIA Sortformer live diarization model",
  "",
  "- Download source used by app (converted artifacts): https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf",
  "- Original upstream model card reference: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1",
  "- Model card license reference at time of writing: apache-2.0.",
  "- Runtime artifact format: GGUF.",
  "- Conversion notice: distributed artifacts are conversion/interoperability outputs.",
  "- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.",
  "",
  "## Local chat / anonymisation model",
  "",
  "- Suggested download source used by app (GGUF artifacts): https://huggingface.co/openresearchtools/Qwen3.5-9B-GGUF",
  "- Original upstream model family reference: https://huggingface.co/Qwen/Qwen3.5-9B",
  "- Model card license reference at time of writing: apache-2.0.",
  "- Runtime artifact format: GGUF.",
  "- Conversion notice: distributed artifacts are conversion/interoperability outputs.",
  "- Non-affiliation notice: listing or bundling does not imply affiliation with or endorsement by original authors.",
  "",
  "## Upstream original citations",
  "",
  "Whisper original reference:",
  "@misc{Radford2022Whisper, title={Robust Speech Recognition via Large-Scale Weak Supervision}, author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever}, year={2022}, eprint={2212.04356}, archivePrefix={arXiv}, primaryClass={eess.AS}}",
  "",
  "Voxtral original reference:",
  "@misc{MistralVoxtralRealtime2026, title={Voxtral Mini 4B Realtime}, author={Mistral AI}, year={2026}, howpublished={Hugging Face model card}}",
  "",
  "Sortformer original reference:",
  "@misc{NvidiaSortformer2026, title={diar_streaming_sortformer_4spk-v2.1}, author={NVIDIA}, year={2026}, howpublished={Hugging Face model card}}",
  "",
  "Qwen original reference:",
  "@misc{Qwen35_2025, title={Qwen3.5-9B}, author={Qwen Team}, year={2025}, howpublished={Hugging Face model card}}"
) -join "`n"
Write-Utf8File -Path $modelNoticesOut -Content $modelNotices

$officialEngineLicensePath = Join-Path $engineUpstreamDir "LICENSES.md"
$officialEngineLicenseText = if (Test-Path -LiteralPath $officialEngineLicensePath -PathType Leaf) {
  Read-TextFileSafe -Path $officialEngineLicensePath
} else {
  ""
}
$officialEngineReadmePath = Join-Path $engineUpstreamDir "README.md"
$officialEngineReadmeText = if (Test-Path -LiteralPath $officialEngineReadmePath -PathType Leaf) {
  Read-TextFileSafe -Path $officialEngineReadmePath
} else {
  ""
}
$engineCombinedLicenseForPage = if (-not [string]::IsNullOrWhiteSpace($officialEngineLicenseText)) {
  $markerMatch = [regex]::Match($officialEngineLicenseText, '(?s)BEGIN: Openresearchtools-Engine \[LICENSE\].*')
  if ($markerMatch.Success) {
    $markerMatch.Value.Trim()
  } else {
    $officialEngineLicenseText.Trim()
  }
} else {
  "Engine combined license bundle not found."
}
$engineNotices = if (-not [string]::IsNullOrWhiteSpace($officialEngineReadmeText)) {
  $officialEngineReadmeText.Trim()
} else {
  "Engine notices unavailable."
}
Write-Utf8File -Path (Join-Path $licensesDir "ENGINE_THIRD_PARTY_NOTICES.md") -Content $engineNotices

$engineLicenses = @(
  "# Engine Licenses",
  "",
  "See ENGINE_THIRD_PARTY_LICENSES_FULL.md for the current engine license bundle."
) -join "`n"
Write-Utf8File -Path (Join-Path $licensesDir "ENGINE_THIRD_PARTY_LICENSES.md") -Content $engineLicenses

$appFullBuilder = New-Object System.Text.StringBuilder
[void]$appFullBuilder.AppendLine("# Third-Party Licenses")
[void]$appFullBuilder.AppendLine("")

foreach ($row in $rows) {
  [void]$appFullBuilder.AppendLine(("## Crate: {0} {1}" -f $row.Name, $row.Version))
  [void]$appFullBuilder.AppendLine("")
  [void]$appFullBuilder.AppendLine(("- License expression: {0}" -f $row.LicenseExpression))
  [void]$appFullBuilder.AppendLine("")

  $crateDir = [string]$row.CrateDir
  $licenseFiles = @()
  if (-not [string]::IsNullOrWhiteSpace($crateDir) -and (Test-Path -LiteralPath $crateDir -PathType Container)) {
    $licenseFiles = @(Get-ChildItem -Recurse -File -LiteralPath $crateDir |
      Where-Object { $_.Name -ne "SOURCE.txt" } |
      Sort-Object FullName)
  }

  if (@($licenseFiles).Count -eq 0) {
    [void]$appFullBuilder.AppendLine("_No license files captured for this crate._")
    [void]$appFullBuilder.AppendLine("")
    continue
  }

  foreach ($file in $licenseFiles) {
    $rel = Get-RelativePath -BasePath $crateDir -TargetPath $file.FullName
    Append-LicenseTextSection -Builder $appFullBuilder -Title ($rel -replace "\\", "/") -Body (Read-TextFileSafe -Path $file.FullName)
  }
}
Write-Utf8File -Path (Join-Path $licensesDir "APP_THIRD_PARTY_LICENSES_FULL.md") -Content $appFullBuilder.ToString()

$engineFullBuilder = New-Object System.Text.StringBuilder
[void]$engineFullBuilder.AppendLine("# Engine Licenses")
[void]$engineFullBuilder.AppendLine("")
[void]$engineFullBuilder.AppendLine("~~~~text")
[void]$engineFullBuilder.AppendLine($engineCombinedLicenseForPage)
[void]$engineFullBuilder.AppendLine("~~~~")
Write-Utf8File -Path (Join-Path $licensesDir "ENGINE_THIRD_PARTY_LICENSES_FULL.md") -Content $engineFullBuilder.ToString()

$modelNoticesContent = if (Test-Path -LiteralPath $modelNoticesOut -PathType Leaf) {
  Get-Content -LiteralPath $modelNoticesOut -Raw
} else {
  "Model notices unavailable."
}

$combinedNotices = @(
  "# Third-Party Notices",
  "",
  "## App notices",
  "",
  $thirdPartyNotices.Trim(),
  "",
  "---",
  "",
  "## Model notices",
  "",
  $modelNoticesContent.Trim(),
  "",
  "---",
  "",
  "## Engine notices",
  "",
  $engineNotices.Trim()
) -join "`n"
Write-Utf8File -Path (Join-Path $licensesDir "THIRD_PARTY_NOTICES_ALL.md") -Content $combinedNotices

$combinedLicenses = (Read-TextFileSafe -Path (Join-Path $licensesDir "APP_THIRD_PARTY_LICENSES_FULL.md")).Trim()
Write-Utf8File -Path (Join-Path $licensesDir "THIRD_PARTY_LICENSES_ALL.md") -Content $combinedLicenses

# Keep only the files bundled into the app binary.
$bundledOutputs = @(
  "THIRD_PARTY_NOTICES_ALL.md",
  "THIRD_PARTY_LICENSES_ALL.md",
  "ENGINE_THIRD_PARTY_LICENSES_FULL.md"
)

Get-ChildItem -LiteralPath $licensesDir -File -ErrorAction SilentlyContinue |
  Where-Object { $bundledOutputs -notcontains $_.Name } |
  Remove-Item -Force -ErrorAction SilentlyContinue

Get-ChildItem -LiteralPath $licensesDir -Directory -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "License regeneration complete."
Write-Host "Root package: $($rootPackage.name) $($rootPackage.version)"
Write-Host "Crates.io dependencies documented: $($rows.Count)"
Write-Host "Crates requiring manual license review: $($manualReviewCrates.Count)"
