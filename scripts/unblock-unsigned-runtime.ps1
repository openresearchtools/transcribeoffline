param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimeDir
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RuntimeDir -PathType Container)) {
  throw "Runtime directory does not exist: $RuntimeDir"
}

# Clear Mark-of-the-Web for all runtime files so unsigned DLLs/EXEs can load.
Get-ChildItem -LiteralPath $RuntimeDir -Recurse -File -Force | ForEach-Object {
  try {
    Unblock-File -LiteralPath $_.FullName -ErrorAction Stop
  } catch {
    # Keep going; some files may not have MOTW stream.
  }
}

Write-Output "Unsigned runtime unblock complete for '$RuntimeDir'."
