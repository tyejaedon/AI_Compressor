# Model Params Input

Use `--params-file` to load model/training parameters from JSON instead of passing many CLI flags.

Example file:

- `documentation/model_params_example.json`

The file can be:

1. Sectioned by modality/trainer (`image`, `image_lossy`, `audio`, `video`), or
2. A flat JSON object (for a single trainer).

CLI flags still win over file values if both are provided.

## Examples

```zsh
python src/training/train_autoencoder_image_local.py --params-file documentation/model_params_example.json --data-dir data/ImageData/archive --real-only --no-export-tflite
python src/training/train_autoencoder_image_lossy_local.py --params-file documentation/model_params_example.json --data-dir data/ImageData/archive --real-only --no-export-tflite
python src/training/train_autoencoder_audio_local.py --params-file documentation/model_params_example.json --data-dir data/AudioData/ESC-50-master/audio --no-export-tflite
python src/training/train_autoencoder_video_local.py --params-file documentation/model_params_example.json --data-dir "data/VIDEO DATA" --no-export-tflite
python src/training/train_upscaler_image_local.py --params-file documentation/model_params_example.json --data-dir data/ImageData/archive --real-only --no-export-tflite
```

## Image trainer data cache (Milestone 4)

`train_autoencoder_image_local.py` now supports an optional on-disk decoded-tensor cache for real-image runs:

| Flag | Default | Purpose |
|---|---|---|
| `--cache-dir <path>` | off | Writes/reads TFRecord-backed image tensors for each real-data split (`train`, `validation`, `test`) so repeated runs can skip re-decoding unchanged source images. |

Notes:

- The cache is **opt-in** and leaves prior behavior unchanged when omitted.
- Cache entries are keyed by split membership plus the requested `block_size`, and the key hashes each source file's path, size, and `mtime`, so changing the underlying image set automatically invalidates the old cache.
- Run reports surface the cache status under `split_info.cache.*` (`built`, `hit`, or `disabled`) along with the TFRecord/manifest paths.

## Image lossy trainer: real vs proxy compression metrics (Milestone 1)

`evaluation_report.md` from `train_autoencoder_image_lossy_local.py` distinguishes two families of size metrics:

- **JPEG/WebP baseline proxy** (`lossy_size_stats`, `matched_psnr_baseline_benchmark`): re-encodes the
  model's *reconstructed pixels* as JPEG/WebP and compares byte sizes. This has always existed and is a
  proxy for "how compressible does the output look", not the model's own bitstream.
- **Real latent bitrate** (`latent_rate_stats`, new): the empirical Shannon entropy of the model's own
  quantized latent tensor, reported as `latent_bits_per_pixel` and `estimated_bitstream_bytes_per_image`.
  This is what the model would actually cost to transmit if paired with an entropy coder.
- **Prototype entropy coder** (`entropy_coding`, opt-in via `--enable-entropy-coding`): actually range-codes
  a small sample of test-set latents and reports the real compressed byte count, for comparison against the
  entropy estimate above (they should be very close).

New `image_lossy` flags (all default to preserving prior behavior):

| Flag | Default | Purpose |
|---|---|---|
| `--rate-loss-mode {magnitude,entropy}` | `magnitude` | Rate loss formulation used during training; `entropy` uses a differentiable soft-histogram entropy estimate (in bits, typical scale ~0-5+) instead of mean \|latent\| (typical scale ~0-1). **When switching to `entropy`, reduce `--rate-lambda` by roughly 10-100x** vs. what you'd use for `magnitude` (e.g. keep the default `0.002`, or go lower), or the rate term can overwhelm reconstruction loss and stall PSNR entirely. |
| `--enable-entropy-coding` | off | Run the prototype range coder over a handful of test-set latents after training. |
| `--entropy-coding-samples` | `8` | How many test images to actually range-encode when the above is enabled. |
| `--latent-bit-depth {4,6,8,10}` | `8` | Quantization bit-depth for the latent bottleneck (Milestone 2). |
| `--quant-noise-anneal` | off | Fade additive quantization noise from full strength to zero over training (QAT-style annealing, Milestone 2). |

## Audio trainer: real vs theoretical compression accounting (Milestone A1)

`audio_evaluation_report.md` from `train_autoencoder_audio_local.py` now reports theoretical and measured compression metrics side by side:

- **Theoretical compression ratio** (`estimated_input_to_latent_ratio`, `theoretical.latent_bits_per_clip`): compares raw PCM bit size (`clip_len * 16 bits`) against an assumed latent bit allocation (`latent_dim * latent_bits`). This metric preserves backward compatibility with baseline reports.
- **Actual compressed bytes per clip** (`actual_latent_bytes_per_clip`, `actual_compression_ratio`): measured by encoding empirical test latents through a static range coder and calculating Shannon entropy (`actual.entropy_bits_per_dim`). When latent quantization is disabled, reports the true float32 latent cost (`latent_dim * 4 bytes/clip`).
- **Quantization-Aware Training (QAT)**: `--latent-bit-depth` (e.g. `8`) inserts a `StraightThroughQuantize` layer after the linear bottleneck, allowing gradients to flow back via straight-through estimation (STE) during forward/backward passes.

New `audio` flags (all default to preserving prior behavior):

| Flag | Default | Purpose |
|---|---|---|
| `--latent-bit-depth <int>` | `0` (disabled/float32) | Discrete quantization bit-depth for the latent vector (e.g. `8` for 256 levels). When `0`, leaves the latent as standard unquantized float32. |
| `--quant-noise-anneal` | off | Linearly fades additive quantization noise from full strength to zero across training epochs so the network smoothly adapts to discrete levels. |

## Audio trainer: perceptual quality & psychoacoustic loss (Milestone A3)

`train_autoencoder_audio_local.py` supports configurable loss profiles and perceptual frequency-domain losses:

- **Loss profiles (`--loss-profile`)**:
  - `balanced` (default): reproduces baseline weights `(mse=0.55, l1=0.10, stft=0.35, mel=0.0, a_weight=0.0)`.
  - `snr`: heavily focuses on time-domain waveform alignment `(mse=0.70, l1=0.15, stft=0.15, mel=0.0, a_weight=0.0)`.
  - `perceptual`: emphasizes psychoacoustic frequency sensitivity `(mse=0.20, l1=0.05, stft=0.15, mel=0.35, a_weight=0.25)`.
- **Log-Mel spectral loss (`--loss-mel-weight`)**: computes L1 distance over log-mel spectrograms generated via `tf.signal.linear_to_mel_weight_matrix`.
- **A-weighted STFT loss (`--loss-a-weight`)**: applies the IEC 61672:2003 A-weighting frequency sensitivity curve to the STFT magnitude error, penalizing artifacts most heavily where human hearing is most sensitive (~2.5–3 kHz).
- **Perceptual evaluation metrics (`--eval-perceptual-metrics`)**: evaluates zero-dependency perceptual audio quality on the test set, computing Log-Spectral Distance (LSD in dB), A-weighted LSD, and sub-band envelope correlation (STOI proxy).

| Flag | Default | Purpose |
|---|---|---|
| `--loss-profile {balanced,snr,perceptual}` | `balanced` | Loss profile preset. `balanced` reproduces default baseline behavior. |
| `--loss-mel-weight <float>` | `0.0` (or per profile) | Weight for log-mel spectral envelope loss term. |
| `--loss-a-weight <float>` | `0.0` (or per profile) | Weight for psychoacoustic A-weighted STFT magnitude loss. |
| `--mel-bins <int>` | `64` | Number of mel filterbank bins used when mel loss is active. |
| `--eval-perceptual-metrics` | off | Compute zero-dependency perceptual metrics (LSD in dB, A-LSD in dB, STOI envelope proxy) in evaluation reports. |

## Audio trainer: data pipeline robustness (Milestone A4)

`train_autoencoder_audio_local.py` and `prepare_audio_dataset.py` support multi-channel input, bandlimited polyphase resampling, additive noise augmentation, and corrupted audio recovery:

- **Channel modes (`--channels {mono,stereo}`)**:
  - `mono` (default): 1-channel autoencoder architecture. Multi-channel audio files are downmixed according to `--stereo-downmix`.
  - `stereo`: 2-channel autoencoder architecture with independent convolutional processing across stereo channels.
- **Stereo downmix strategy (`--stereo-downmix {average,left,right,error}`)**:
  - `average` (default): averages channels across the channel axis with clean warning logs.
  - `left` / `right`: selects a discrete channel.
  - `error`: strictly raises an exception if multi-channel input is detected.
- **Resampling algorithm (`--resample-method {linear,polyphase}`)**:
  - `linear` (default): fast linear interpolation (preserves prior benchmark behavior).
  - `polyphase`: bandlimited Kaiser-windowed sinc interpolation with anti-aliasing FIR filtering in pure NumPy.
- **Additive noise augmentation (`--augment-noise-std <float>`)**:
  - Injects additive Gaussian noise into the training input (`--augment-noise-std`, default 0.0 / off) to train the model as a robust denoising autoencoder.
- **Corrupted input recovery**:
  - WAV ingest gracefully recovers from corrupted headers, invalid bytes, and clips NaN/Inf samples to valid dynamic ranges without pipeline crashes.

| Flag | Default | Purpose |
|---|---|---|
| `--channels {mono,stereo}` | `mono` | Channel architecture mode (1 channel for mono, 2 channels for stereo). |
| `--stereo-downmix {average,left,right,error}` | `average` | Downmixing strategy for multi-channel source files when mono mode is active. |
| `--resample-method {linear,polyphase}` | `linear` | Sample rate conversion algorithm: `linear` or `polyphase` (Kaiser sinc). |
| `--augment-noise-std <float>` | `0.0` (off) | Standard deviation of additive Gaussian noise during training. |

## Audio trainer: full CLI flag reference (Milestone A5)

Below is the complete reference of all command-line arguments supported by `src/training/train_autoencoder_audio_local.py`:

### Model architecture & signal representation
| Flag | Default | Description |
|---|---|---|
| `--preset {custom,m1-air-fast,m1-air-balanced,m1-air-quality}` | `m1-air-fast` | High-level hardware/quality configuration preset. |
| `--sample-rate <int>` | `22050` | Audio sampling frequency in Hz. |
| `--clip-seconds <float>` | `0.5` | Clip duration in seconds per training window. |
| `--latent-dim <int>` | `1024` | Bottleneck dimension size. |
| `--model-base-filters <int>` | `64` | Conv1D base filter channel depth. |
| `--model-kernel-size <int>` | `5` | Conv1D convolutional kernel size. |
| `--channels {mono,stereo}` | `mono` | Model channel architecture: `mono` (1 channel) or `stereo` (2 channels). |
| `--stereo-downmix {average,left,right,error}` | `average` | Multi-channel downmixing rule when loading audio in mono mode. |
| `--resample-method {linear,polyphase}` | `linear` | Sample-rate interpolation algorithm (`linear` or `polyphase` sinc). |

### Quantization & bitrate accounting
| Flag | Default | Description |
|---|---|---|
| `--latent-bit-depth <int>` | `0` (float32) | Discrete quantization bit-depth for bottleneck latents (0 = disabled). |
| `--quant-noise-anneal` | off | Linearly fade additive quantization noise to zero during training epochs. |
| `--latent-bits <int>` | `8` | Assumed bit allocation for backward-compatible theoretical ratio accounting. |

### Loss functions & perceptual objectives
| Flag | Default | Description |
|---|---|---|
| `--loss-profile {balanced,snr,perceptual}` | `balanced` | Loss profile preset configuring default weights across objectives. |
| `--loss-mse-weight <float>` | `0.55` | Time-domain mean squared error loss weight. |
| `--loss-l1-weight <float>` | `0.10` | Time-domain mean absolute error loss weight. |
| `--loss-stft-weight <float>` | `0.35` | STFT magnitude spectral loss weight. |
| `--loss-mel-weight <float>` | `0.0` | Psychoacoustic log-mel spectral envelope loss weight. |
| `--loss-a-weight <float>` | `0.0` | Psychoacoustic A-weighted STFT spectral loss weight. |
| `--mel-bins <int>` | `64` | Number of mel filterbank bins used when mel loss is active. |
| `--loss-hard-weight <float>` | `1.5` | Upweight multiplier for worst reconstruction errors. |
| `--loss-stft-power <float>` | `1.2` | Exponent applied to STFT magnitude error term. |

### Training, optimization & regularization
| Flag | Default | Description |
|---|---|---|
| `--batch-size <int>` | `16` | Batch size per training step. |
| `--epochs <int>` | `20` | Maximum number of training epochs. |
| `--lr <float>` | `0.0001` | Adam learning rate. |
| `--seed <int>` | `42` | Random seed for data shuffling and weight initialization. |
| `--latent-l1 <float>` | `0.0` | L1 sparsity regularization penalty on latent vectors. |
| `--latent-l2 <float>` | `0.0` | L2 weight decay penalty on latent vectors. |
| `--target-snr-db <float>` | `90.0` | Validation SNR (dB) threshold for early stopping. |
| `--disable-target-snr-stop` | off | Run all epochs without early stopping on target SNR. |
| `--augment-noise-std <float>` | `0.0` | Standard deviation of additive Gaussian noise during training. |
| `--shuffle-buffer <int>` | `256` | Buffer size for training dataset shuffling. |

### Dataset, evaluation & export
| Flag | Default | Description |
|---|---|---|
| `--data-dir <path>` | required | Root directory containing audio files. |
| `--real-val-ratio <float>` | `0.15` | Fraction of real files for validation. |
| `--real-test-ratio <float>` | `0.15` | Fraction of real files for test evaluation. |
| `--real-file-limit <int>` | `0` (all) | Maximum number of audio files to process. |
| `--eval-perceptual-metrics` | off | Compute perceptual metrics (LSD in dB, A-LSD, STOI envelope proxy). |
| `--run-mp3-benchmark` | off | Benchmark reconstruction size against MP3 CBR bitrates. |
| `--output-root <path>` | `models/audio_local_run` | Base output directory for training runs and evaluation reports. |
| `--params-file <path>` | none | JSON configuration file specifying hyperparameter overrides. |
| `--export-tflite` / `--no-export-tflite` | `--export-tflite` | Enable or disable exporting TFLite flatbuffers. |
| `--tflite-fp16` | off | Enable float16 quantization during TFLite export. |

## Learned image upscaler (Milestone 3)

`train_upscaler_image_local.py` trains a small residual CNN + sub-pixel (depth-to-space) upsampler on the
same bicubic-downscale-plus-noise degrade recipe used by `train_autoencoder_image_local.py`, but produces a
genuinely lower-resolution input (rather than a same-size denoising target). Its own
`upscaler_evaluation_report.md` reports learned-vs-bicubic PSNR/SSIM directly.

`upscale_reconstructed_images.py --upscaler-mode {bicubic,learned}` (default `bicubic`, unchanged) can then
use a trained upscaler model in place of plain BICUBIC resizing via `--learned-upscaler-model <path>`,
with `--upscaler-tile-size`/`--upscaler-tile-overlap` bounding memory use on large images via overlapping,
blended tiles. If no learned model path is given (or it fails to load), it falls back to BICUBIC with a
warning.
