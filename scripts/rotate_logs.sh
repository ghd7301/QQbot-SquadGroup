#!/bin/sh
# Rotate launchd logs for squad-qqbot-mvp.
# Usage: sh scripts/rotate_logs.sh
# Can be run manually or scheduled via cron (e.g., weekly).

set -eu

WORK_DIR="/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp/work"
KEEP=5
MAX_SIZE=$((5 * 1024 * 1024))  # 5MB

rotate_file() {
    filepath="$1"
    if [ ! -f "$filepath" ]; then
        return
    fi
    size=$(stat -f%z "$filepath" 2>/dev/null || echo 0)
    if [ "$size" -lt "$MAX_SIZE" ]; then
        return
    fi

    # Shift rotated files
    i=$KEEP
    while [ "$i" -gt 0 ]; do
        src="${filepath}.${i}"
        if [ "$i" -ge "$KEEP" ]; then
            [ -f "$src" ] && rm -f "$src"
        else
            dst="${filepath}.$((i + 1))"
            [ -f "$src" ] && mv "$src" "$dst"
        fi
        i=$((i - 1))
    done

    # Rotate current file
    mv "$filepath" "${filepath}.1"
    touch "$filepath"
    echo "Rotated: $filepath"
}

mkdir -p "$WORK_DIR"
rotate_file "$WORK_DIR/launchd.out.log"
rotate_file "$WORK_DIR/launchd.err.log"
echo "Log rotation complete."
