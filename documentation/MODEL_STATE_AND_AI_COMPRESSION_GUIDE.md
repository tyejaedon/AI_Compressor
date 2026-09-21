# AI Compressor: Model Status and AI Compression Guide

## Purpose

This document summarizes:

- How AI-based compression works in this project
- The current state of each model pipeline (image, lossy image, audio, video)
- How Keras is used throughout the codebase
- Practical usage notes and current constraints

---

## What "AI Compression" Means Here

Traditional codecs (JPEG, MP3, H.264) use hand-designed transforms and rules.

AI compression in this repo uses **autoencoders**:

1. **Encoder path** maps input data to a compact latent representation
2. **Decoder path** reconstructs the signal from latent space
3. Training minimizes distortion (MSE/L1/SSIM-based losses) and sometimes bitrate proxy terms

The core trade-off is always:

- Smaller/more constrained latent -> smaller size potential, lower fidelity
- Larger latent/softer constraints -> higher fidelity, weaker compression

Project metrics focus primarily on:

- **PSNR** (reconstruction error quality)
- **SSIM** (structural similarity for image/video)
- **SNR/PSNR for audio**
- Derived file-size proxy comparisons (mostly JPEG/codec baseline comparisons in lossy flows)

---

## What Keras Is and How It Is Used

[Keras](https://keras.io/) is a high-level deep learning API (running on TensorFlow here) that simplifies model definition, training, callbacks, and export.

In this repo, Keras is used for:

- Building models via `tf.keras.layers` and `tf.keras.Model`
- Compiling with optimizer + loss + metrics
- Training via `model.fit(...)`
- Validation/testing via `model.evaluate(...)`
- Model checkpointing and early stopping callbacks
- Export to `.keras`, `.weights.h5`, and optional `.tflite`

Common callback patterns used:

- `ModelCheckpoint` (save best validation run)
- `EarlyStopping` (stop when validation stops improving)
- `ReduceLROnPlateau` (decay learning rate on plateau)
- Custom PSNR target callback(s)

---

## Current Model State (By Modality)

## 1) Image Upscaler / Reconstruction Model

Primary trainer:

- `train_autoencoder_image_local.py`

Current capabilities:

- Real-only mode and CIFAR-backed mode
- Explicit split mode:
  - `--train-dir`, `--val-dir`, `--test-dir`
- Optional external holdout evaluation:
  - `--holdout-dir`
- Super-resolution-style degradation pipeline:
  - Input is degraded/down-up sampled patch, target is clean patch
- Configurable composite loss:
  - `--loss-mse-weight`, `--loss-l1-weight`, `--loss-ssim-weight`
- Optional disabling of target-PSNR early stop:
  - `--disable-target-psnr-stop`
- Rich reporting:
  - `evaluation_report.md`
  - training curves + preview montage
  - size metrics and holdout metrics

Important current behavior:

- Uses random crops from source images (`--train-patches-per-image` controls expansion)
- Explicit split mode is preferred to avoid train/val/test leakage when dataset already has canonical splits

---

## 2) Image Lossy Compression Model

Primary trainer:

- `train_autoencoder_image_lossy_local.py`

Current capabilities:

- Real-only training support and dataset guards
- Explicit split mode:
  - `--train-dir`, `--val-dir`, `--test-dir`
- Less aggressive compression tuning compared to earlier settings:
  - Lower `rate_lambda` range
  - Higher default latent dimensions
- JPEG/WebP benchmark hooks for baseline comparison
- Matched-PSNR-style codec comparison logic with broader quality sweeps
- Size-focused reporting:
  - reconstructed JPEG size vs original baseline
  - model-vs-codec matched baseline size ratios

Known practical note:

- WebP baseline availability depends on TensorFlow build exposing WebP encode/decode APIs.
- For canonical datasets that already provide train/val/test folders (for example `data/cifar10 2`), explicit split mode avoids accidental split reshuffling.

---

## 3) Audio Autoencoder

Primary trainer:

- `train_autoencoder_audio_local.py`

Related prep utility:

- `prepare_audio_dataset.py`

Current capabilities:

- Real WAV ingestion from filesystem datasets
- Audio preview plot + preview WAV artifacts
- Optional MP3 benchmark path
- Model export (`.keras`, weights, optional `.tflite`)

Canonical dataset prep supports:

- Mixed codec ingestion (`.mp3/.flac/.ogg/...`) -> canonical format
- Normalization options
- Per-file metadata sidecars
- Preparation report generation

---

## 4) Video Autoencoder

Primary trainer:

- `train_autoencoder_video_local.py`

Current capabilities:

- FFmpeg-based real video decoding
- Supported containers include:
  - `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`
- 5D clip input shape:
  - `(batch, frames, height, width, channels)`
- Optional MP4 benchmark comparison report
- Standard training artifacts + optional TFLite export

Notes:

- Extension support != guaranteed codec decode support (depends on installed FFmpeg codec build)

---

## Production Bundle State

Bundle builder script:

- `build_best_model_bundle.py`

Current bundle flow:

- Select best report per modality by test PSNR where available
- Copy key artifacts into `models/production_bundle/...`
- Additional manual bundle assembly has also been used to combine:
  - latest image runs
  - previous audio/video artifacts

---

## Runtime / Hardware Notes (Current Environment)

From recent local checks in this workspace session:

- TensorFlow version: `2.21.0`
- `tf.config.list_physical_devices("GPU")` returned empty
- Current training executes on CPU in this environment

On M1/M2 Macs, Metal acceleration usually requires a compatible TensorFlow + tensorflow-metal pairing. If versions mismatch, plugin load failures can occur.

---

## How to Think About "Optimal" in This Repo

There is no single optimal model for all goals. Use-case-specific optimization:

- **Highest fidelity reconstruction**: larger latent, weaker compression pressure, PSNR/SSIM weighted losses
- **Smaller output files**: stronger rate pressure, lower latent, benchmark against JPEG/WebP/MP3/H.264 baselines
- **Fast local iteration**: fewer files, fewer patches/image, smaller blocks, fewer epochs, higher LR warm runs
- **Reliable evaluation**: explicit split directories + independent holdout set

---

## Recommended Evaluation Checklist

For each final candidate run:

1. Check train/val/test gap (overfit risk)
2. Check holdout metrics (true generalization)
3. Compare input->target vs output->target PSNR/SSIM deltas
4. Compare size ratios at matched quality baselines
5. Review preview artifacts for visual plausibility (not metrics alone)

---

## File Index (Key Entry Points)

- `train_autoencoder_image_local.py`
- `train_autoencoder_image_lossy_local.py`
- `train_autoencoder_audio_local.py`
- `train_autoencoder_video_local.py`
- `prepare_audio_dataset.py`
- `build_best_model_bundle.py`
- `README.md`

---

## Final Summary

At current state, the project has:

- A flexible image upscaler/reconstructor with explicit split and holdout evaluation support
- A tuned lossy image path with stronger baseline/size analysis controls
- Working audio/video trainers with real-data modes and deployment artifact outputs
- Production bundling support for best-run packaging

The biggest quality lever is not one flag but the combination of:

- clean split protocol,
- realistic degradation pipeline,
- objective weighting,
- and fair baseline comparisons.

