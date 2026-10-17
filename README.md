# AI Compressor - Neural Compression for Image, Audio, and Video

Local-first neural compression project with autoencoders for image, audio, and video.

## What is Neural compression

Traditional codecs (JPEG/MP3/H.264) are hand-engineered. Neural compression learns compact latent representations with an encoder/decoder and optimizes reconstruction quality directly for your target data and constraints.

## Current benchmark snapshot (October 2026)

| Modality / Task | Test PSNR | Test MSE | Test Loss | Test SSIM/SNR | Key Focus / Baseline |
|:----------------|----------:|---------:|----------:|--------------:|:---------------------|
| Image (Standard Super-Res) | 72.518 dB | 8.225e-08 | 2.640e-05 | SSIM 0.999999 | 2x Bicubic restoration |
| Image (Lossy Rate-Distortion) | 25.676 dB | 0.00321 | 0.01361 | SSIM 0.9123 | 18.47 bpp, 0.829x JPEG ratio |
| Image (Learned Upscaler) | 26.338 dB | 0.00267 | 0.01947 | SSIM 0.8855 | Outperforms Bicubic (0.8705) |
| Audio (Waveform Autoencoder) | 32.223 dB | 0.00100 | 0.02400 | SNR 21.981 dB | Multi-scale spectral recon |
| Video (Spatio-Temporal AE) | 30.423 dB | 0.00100 | 0.01400 | SSIM 0.8980 | 3D conv latent compression |

Average across best models:

- Test PSNR: **37.436 dB**
- Lossy & Generative average PSNR (Audio, Video, Lossy Image, Upscaler): **28.665 dB**

## Model Hyperparameters
Best model hyperparameters for each modality and architecture (from high-end benchmark configurations):

| Modality / Task | Latent Dim / Channels | Filters / ResBlocks | Learning Rate | Batch Size | Epochs | Optimization / Notes |
|:----------------|----------------------:|--------------------:|--------------:|-----------:|-------:|:---------------------|
| Image (Standard Super-Res) | 192 | 64 base filters | 0.00025 | 32 | 12 | 2x upscale, SSIM + Charbonnier loss |
| Image (Lossy Rate-Distortion) | 48 | 32 base filters | 0.00025 | 32 | 6 | Rate-lambda 0.0006, 8-bit QAT, entropy coding |
| Image (Learned Upscaler) | N/A | 40 filters, 5 resblocks | 0.00025 | 32 | 7 | 2x pixel shuffle, residual blocks |
| Audio (Waveform AE) | 96 (8-bit QAT) | 32 base filters | 0.00012 | 16 | 16 | Continuous phase, 8-bit QAT, 1204.7x measured ratio |
| Video (Spatio-Temporal AE) | 192 | 24 base filters | 0.00010 | 8 | 15 | 3D conv spatio-temporal encoder |

## Training Metrics  (x-axis : Epochs)
### Image Model:
![training_metrics.png](models/production_bundle/best_20260702_235656/image/training_metrics.png)
image model training metrics (PSNR, MSE, Loss, SSIM) over epochs
### Audio Model:
![training_metrics.png](models/production_bundle/best_20260702_235656/audio/audio_training_metrics.png)
audio model training metrics (PSNR, MSE, Loss, SNR) over epochs
### Video Model:
![training_metrics.png](models/production_bundle/best_20260702_235656/video/video_training_view_metrics.png)
video model training metrics (PSNR, MSE, Loss, SSIM) over epochs

## Preview reconstruction quality of best models
preview of reconstruction quality for best models on test data, showing original and reconstructed samples.
### Image:
![image_reconstruction.png](models/production_bundle/best_20260702_235656/image/preview_reconstructions.png)

### Audio:
![audio_reconstruction.png](models/production_bundle/best_20260702_235656/audio/audio_preview_waveform.png)
Playback audio: [audio_preview_reconstruction.wav](models/production_bundle/best_20260702_235656/audio/audio_preview_waveform.wav)
### Video:
![video_reconstruction.png](models/production_bundle/best_20260702_235656/video/video_preview_reconstruction.png)

Source reports:

- `models/highend_run/20261013_231532/image_standard/20261013_231537/evaluation_report.md`
- `models/highend_run/20261013_232606/image_lossy/20261013_235837/evaluation_report.md`
- `models/highend_run/20261014_013253/image_upscaler/20261014_013303/upscaler_evaluation_report.md`
- `models/highend_run/20261017_135058/audio/20261017_135103/audio_evaluation_report.md`
- `models/random_search_goal30/video/trial_001/20260702_183130/video_evaluation_report.md`

## Quick start
