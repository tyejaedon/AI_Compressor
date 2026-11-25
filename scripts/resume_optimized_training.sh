#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
LOG_DIR="$ROOT_DIR/models/optimized_run/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === 2/3 Training Image Lossy Autoencoder (compact, 8x bottleneck, entropy+QAT, fp16 TFLite) ==="
"$PYTHON_BIN" src/training/train_autoencoder_image_lossy_local.py \
  --preset m1-air-quality \
  --data-dir data/ImageData/archive \
  --real-only \
  --min-real-images 50 \
  --min-split-images 5 \
  --block-size 128 \
  --downsample-stages 3 \
  --latent-dim 32 \
  --rate-loss-mode entropy \
  --rate-lambda 0.0005 \
  --latent-bit-depth 6 \
  --quant-noise-anneal \
  --enable-entropy-coding \
  --entropy-coding-samples 12 \
  --compact-arch \
  --tflite-fp16 \
  --output-root models/optimized_run/image_lossy

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === 3/3 Training Video Autoencoder ((2+1)D, residual temporal, 6-bit QAT+entropy, fp16 TFLite) ==="
"$PYTHON_BIN" src/training/train_autoencoder_video_local.py \
  --preset m1-air-quality \
  --data-dir "data/VIDEO DATA" \
  --cache-dir models/cache/video \
  --decode-workers 4 \
  --frames 8 \
  --height 64 \
  --width 64 \
  --fps 12 \
  --latent-channels 16 \
  --latent-bit-depth 6 \
  --quant-noise-anneal \
  --rate-lambda 0.0003 \
  --temporal-mode residual \
  --loss-profile balanced \
  --loss-temporal-weight 0.15 \
  --run-mp4-benchmark \
  --benchmark-codecs h264,vp9 \
  --tflite-fp16 \
  --output-root models/optimized_run/video

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === Models 2 & 3 finished training! ==="
