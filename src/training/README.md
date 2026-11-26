# Training & Autoencoder Encoders (`src/training/`)

This directory contains all **5 neural compression / restoration trainers** plus the **orchestration, hyperparameter search, and override helpers**.

## At a Glance

| File | Role | Architecture | Key Features |
|---|---|---|---|
| `train_autoencoder_image_local.py` | Image Remaster / Restoration | Pooling-free residual CNN (`SeparableConv2D` + `ResidualClipAdd`) | Edge/SSIM/L1/MSE loss, `--degrade-noise-std`, JSON+TFRecord cache, subprocess FP16 TFLite |
| `train_autoencoder_image_lossy_local.py` | Lossy Image Compression | Strided `SeparableConv2D` + `UpSampling2D` (`8x` spatial default) | `StraightThroughQuantize` (4/6/8/10-bit), QAT noise annealing, per-channel `EntropyRatePenalty`, range coder + JPEG/WebP benchmark |
| `train_upscaler_image_local.py` | Learned Super-Resolution | Residual CNN + sub-pixel / upsample SR head (`2x`/`3x`/`4x`) | Trained on synthetic degraded low-res -> high-res pairs; consumed by `upscale_reconstructed_images.py` |
| `train_autoencoder_audio_local.py` | Neural Audio Codec | SEANet-style 1D Conv + dilated resblocks + `SnakeActivation` | `64x` temporal downsample, `tanh` bottleneck, FSQ/STE quantization, Multi-Res STFT + Mel + A-weighted loss |
| `train_autoencoder_video_local.py` | Video Spatio-Temporal AE | Compact `(2+1)D` Conv3D (`2x` temporal, `8x` spatial) | Video-level non-leaking splits, `uint8` clip cache, `--temporal-mode residual`, 6/8-bit QAT, H.264/VP9 benchmark |
| `trainhighend.py` | Multi-Modality Orchestrator | CLI wrapper (`./trainhighend`) | Runs all 5 trainers sequentially with tuned configs; supports `--dry-run`, `--only`, `--epochs-scale` |
| `train_all_optimal_m1.py` | Retry-Until-Target Runner | Sequential M1 loop | Retries each modality until target PSNR/SNR or wall-clock budget is met |
| `random_search_hyperparams.py` | Hyperparameter Tuner | Budgeted random search | Samples spaces from `documentation/random_search_profile.json` and ranks trials by PSNR |
| `param_overrides.py` | JSON Config Loader | `load_overrides` / `apply_overrides` | Applies `--params-file` section overrides (`image`, `image_lossy`, `audio`, `video`) onto `argparse` |

---

## 1. Image Remaster Autoencoder (`train_autoencoder_image_local.py`)

**Purpose:** Restores and sharpens degraded/downscaled images at arbitrary resolution. Note that this model has **no spatial downsampling** (`strides=1`) and uses skip connections (`inputs + residual`), so it acts as a restoration/remaster stage rather than a bottleneck compressor.

- **Architecture (`build_autoencoder`)**:
  - Shallow `Conv2D` stem -> `SeparableConv2D` encoder/decoder stages (`--compact-arch` enabled by default; disable with `--no-compact-arch`) -> `1x1` `bottleneck` -> multi-stage `Add()` skips -> `tanh` `residual_head` -> registered Keras 3 `ResidualClipAdd` layer.
- **Data Pipeline**:
  - Supports `--use-cifar10`, auto-detected `train/val/test` folders, explicit `--train-dir`/`--val-dir`/`--test-dir`, or flat `--data-dir` with `--real-only`.
  - Applies configurable downscale degradation (`--upscale-factor`, `--degrade-interp`, `--degrade-noise-std 0.015`).
  - Logs unreadable/undersized images to `skipped_images.jsonl` and supports `--cache-dir` (JSON filter manifest + per-split TFRecord).
- **Example**:
  ```zsh
  python src/training/train_autoencoder_image_local.py \
    --preset m1-air-quality --data-dir data/ImageData/archive --real-only \
    --cache-dir models/cache/image_remaster --upscale-factor 2 --loss-profile perceptual --tflite-fp16
  ```

---

## 2. Image Lossy Rate-Distortion Autoencoder (`train_autoencoder_image_lossy_local.py`)

**Purpose:** Compresses images into a compact quantized latent tensor for true lossy bitstream storage and compares against matched-PSNR JPEG and WebP.

- **Architecture (`build_autoencoder`)**:
  - Encoder: `--downsample-stages` (default `3` = `8x` per side / `64x` area reduction) of stride-2 `Conv2D`/`SeparableConv2D` + residual blocks (`LeakyReLU`).
  - Bottleneck: `1x1` `sigmoid` -> `StraightThroughQuantize` (`--latent-bit-depth` in `{4,6,8,10}`, optional `--quant-noise-anneal` blending soft noise to hard rounding) -> `RatePenalty` or per-channel `EntropyRatePenalty` (`--rate-loss-mode entropy`).
  - Decoder: `UpSampling2D(bilinear) + SeparableConv2D` + residual blocks (eliminates `Conv2DTranspose` checkerboard artifacts).
- **Compression Accounting & Benchmarks**:
  - Reports empirical Shannon entropy (`latent_bits_per_pixel`, `estimated_bitstream_bytes_per_image`) and optional real range coding (`--enable-entropy-coding` with `_StaticRangeEncoder` / `_StaticRangeDecoder` round-trip verification).
  - Benchmarks against JPEG (`tf.io.encode_jpeg`) and WebP (`tf.io.encode_webp` with Pillow fallback) at matched PSNR.
- **Example**:
  ```zsh
  python src/training/train_autoencoder_image_lossy_local.py \
    --preset m1-air-balanced --use-cifar10 --downsample-stages 3 --latent-dim 32 \
    --rate-loss-mode entropy --rate-lambda 0.0005 --latent-bit-depth 6 --quant-noise-anneal --enable-entropy-coding
  ```

---

## 3. Learned Image Upscaler (`train_upscaler_image_local.py`)

**Purpose:** Single-image super-resolution (`2x`, `3x`, `4x`) trained on degraded low-res -> high-res patch pairs, used by `src/reporting/upscale_reconstructed_images.py --upscaler-mode learned`.

- **Architecture**: Residual CNN body (`--num-residual-blocks`, `--model-base-filters`) with learned upsampling head.
- **Example**:
  ```zsh
  python src/training/train_upscaler_image_local.py --preset m1-air-balanced --use-cifar10 --upscale-factor 2
  ```

---

## 4. Audio Codec Autoencoder (`train_autoencoder_audio_local.py`)

**Purpose:** Compresses 16 kHz mono/stereo waveforms into a low-rate quantized latent representation and reconstructs audio with minimal phase/spectral distortion.

- **Architecture (`build_audio_autoencoder`)**:
  - SEANet-style 1D convolutional encoder/decoder with `SnakeActivation` (`x + sin^2(alpha * x) / alpha`) and dilated residual blocks (dilations `1, 3`) — replaces the legacy `Flatten -> Dense` bottleneck for a `>50x` parameter reduction and arbitrary-length streaming compatibility.
  - `64x` temporal downsampling (`strides=(2,2,4,4)` when `clip_len % 64 == 0`) -> `tanh` `bottleneck` -> `StraightThroughQuantize` (`--latent-bit-depth`, `--quant-noise-anneal`).
- **Loss & Metrics (`make_audio_loss`, `compute_audio_perceptual_metrics`)**:
  - Combines weighted time-domain MSE/L1 with **Multi-Resolution STFT loss** (FFT sizes `256, 512, 1024`: spectral convergence + log-magnitude L1), optional log-mel envelope loss (`--loss-mel-weight`), and psychoacoustic A-weighted STFT loss (`--loss-a-weight`).
  - `--eval-perceptual-metrics` computes Log-Spectral Distance (LSD), A-weighted LSD, and a STOI subband envelope correlation proxy; `--run-mp3-benchmark` compares against `ffmpeg` MP3 CBR bitrates.
- **Example**:
  ```zsh
  python src/training/train_autoencoder_audio_local.py \
    --preset m1-air-balanced --latent-bit-depth 8 --quant-noise-anneal \
    --loss-profile perceptual --eval-perceptual-metrics
  ```

---

## 5. Video Autoencoder (`train_autoencoder_video_local.py`)

**Purpose:** Compresses fixed-length RGB video clips (`T x H x W x 3`) into a quantized 3D convolutional latent tensor and benchmarks against H.264 and VP9.

- **Architecture (`build_video_autoencoder`)**:
  - Factorized `(2+1)D` spatial+temporal residual blocks (`_conv2p1d_block`) with `2x` temporal and `8x` spatial downsampling (`128x` volume reduction before channels) and **no bottleneck-bypassing skip connections**.
  - Optional `--temporal-mode residual` (`TemporalDeltaInput`) encodes frame `0` plus inter-frame deltas `frame[t] - frame[t-1]`.
  - Bottleneck: `VideoStraightThroughQuantize` (`--latent-bit-depth` in `{0,4,6,8,10}`) + per-channel `VideoEntropyRatePenalty` (`--rate-lambda`).
- **Data Pipeline & Benchmarks**:
  - **Zero-leakage splitting**: Splits by video file first (or contiguous non-overlapping temporal segments if `< 3` videos), stores clips in `uint8` RAM, supports parallel `ffmpeg` decoding (`--decode-workers`) and `--cache-dir` `.npz` caching.
  - `--run-mp4-benchmark --benchmark-codecs h264,vp9` compares model PSNR and latent byte size against `libx264` and `libvpx-vp9`.
- **Example**:
  ```zsh
  python src/training/train_autoencoder_video_local.py \
    --preset m1-air-balanced --cache-dir models/cache/video --decode-workers 4 \
    --latent-channels 16 --latent-bit-depth 6 --quant-noise-anneal \
    --temporal-mode residual --loss-temporal-weight 0.15 --run-mp4-benchmark --benchmark-codecs h264,vp9
  ```

---

## 6. Orchestrators & Utilities

- **`trainhighend.py`**: Invoked via `./trainhighend` from repo root. Runs curated high-end training configs across `image_standard`, `image_lossy`, `image_upscaler`, `audio`, and `video`.
- **`train_all_optimal_m1.py`**: Sequential retry loop across modalities that increases epochs/capacity if target PSNR/SNR is not met within a wall-clock budget.
- **`random_search_hyperparams.py`**: Runs budgeted trials over `documentation/random_search_profile.json` and writes `best_params.json`.
- **`param_overrides.py`**: Shared JSON parameter-override helper (`--params-file`).
