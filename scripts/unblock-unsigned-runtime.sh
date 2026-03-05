#!/usr/bin/env sh
set -eu

runtime_dir="${1:-}"
if [ -z "$runtime_dir" ]; then
  echo "Runtime path argument is required." >&2
  exit 2
fi
if [ ! -d "$runtime_dir" ]; then
  echo "Runtime directory does not exist: $runtime_dir" >&2
  exit 2
fi

# macOS: clear quarantine bit recursively when present.
if command -v xattr >/dev/null 2>&1; then
  xattr -dr com.apple.quarantine "$runtime_dir" >/dev/null 2>&1 || true
fi

# Ensure runtime binaries/scripts are executable where relevant.
if command -v find >/dev/null 2>&1; then
  find "$runtime_dir" -type f \( \
    -name "ffmpeg" -o -name "ffprobe" -o -name "*.sh" -o \
    -name "*.dylib" -o -name "*.so" -o -name "*.so.*" \
  \) -exec chmod +x {} \; >/dev/null 2>&1 || true
fi

echo "Unsigned runtime unblock complete for '$runtime_dir'."
