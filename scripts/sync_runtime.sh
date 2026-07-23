#!/bin/sh
set -eu

SOURCE_DIR="/Users/yuce/Documents/Codex/2026-07-20/wo/outputs/squad-qqbot-mvp"
RUNTIME_DIR="/Users/yuce/.codex/qqbot-runtime/squad-qqbot-mvp"

mkdir -p "$RUNTIME_DIR/work"
rsync -a \
  --exclude "work/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "$SOURCE_DIR/" "$RUNTIME_DIR/"
