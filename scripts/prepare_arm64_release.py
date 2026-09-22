#!/usr/bin/env python3
"""Assemble a new ARM64 release, preserving previous binary assets exactly."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--new-asset", required=True, type=Path)
    parser.add_argument("--retain", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", args.tag):
        raise SystemExit("Use an X.Y.Z release tag matching the Cargo package version")
    if args.tag == args.source:
        raise SystemExit("Source and destination releases must differ")
    if subprocess.run(["gh", "release", "view", args.tag, "--repo", args.repo],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise SystemExit("Refusing to overwrite an existing release")
    source = json.loads(run("gh", "api", f"repos/{args.repo}/releases/tags/{quote(args.source, safe='')}"))
    if source["draft"] or source["prerelease"]:
        raise SystemExit("Carry-forward source must be a published stable release")
    assets = {asset["name"]: asset for asset in source["assets"]}
    args.output.mkdir(parents=True, exist_ok=False)
    sums = []
    for name in args.retain:
        if Path(name).name != name or name == args.new_asset.name:
            raise SystemExit(f"Invalid retained filename: {name}")
        expected = assets[name].get("digest", "")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", expected):
            raise SystemExit(f"Source release lacks a SHA-256 digest for {name}")
        subprocess.run(["gh", "release", "download", args.source, "--repo", args.repo,
                        "--pattern", name, "--dir", str(args.output)], check=True)
        actual = digest(args.output / name)
        if expected != f"sha256:{actual}":
            raise SystemExit(f"Carried binary checksum mismatch: {name}")
        sums.append(f"{actual}  {name}\n")
        print(f"Preserved {name}: {actual}")
    import shutil
    shutil.copy2(args.new_asset, args.output / args.new_asset.name)
    sums.append(f"{digest(args.new_asset)}  {args.new_asset.name}\n")
    (args.output / "SHA256SUMS.txt").write_text("".join(sums))
    downloads = "\n".join(f"- [{name}](https://github.com/{args.repo}/releases/download/{args.tag}/{name})"
                          for name in [args.new_asset.name, *args.retain, "SHA256SUMS.txt"])
    notes = f"""Adds Linux ARM64 with automatic Vulkan Engine selection and no CUDA requirement.

The new ARM64 Debian package depends on `openresearchtools-engine (>= 1.17)`.
APT selects and downloads the ARM64 Vulkan engine automatically from the Open Research Tools repository.
For manual installation, download [engine-arm64.deb](https://github.com/openresearchtools/engine/releases/download/v1.17/engine-arm64.deb)
and install it together with the new app package using `sudo apt install ./engine-arm64.deb ./{args.new_asset.name}`.

Windows x64, macOS ARM64, and Linux AMD64 binaries are copied byte-for-byte from release {args.source}.
Their embedded versions and behavior remain unchanged. Only the Linux ARM64 application is newly built.

## Downloads

{downloads}
"""
    (args.output.parent / "release-notes.md").write_text(notes)


if __name__ == "__main__":
    main()
