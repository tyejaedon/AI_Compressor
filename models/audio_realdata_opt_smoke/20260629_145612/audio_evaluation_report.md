# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-06-29T11:58:02.923517Z",
  "config": {
    "output_root": "models/audio_realdata_opt_smoke",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "latent_dim": 224,
    "batch_size": 8,
    "epochs": 1,
    "lr": 0.00012,
    "seed": 42,
    "data_dir": "data/AudioData/ESC-50-master/audio",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_file_limit": 80,
    "synthetic_only": false,
    "train_samples": 7000,
    "val_samples": 2000,
    "test_samples": 2000,
    "target_psnr": 99.0,
    "shuffle_buffer": 1024,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp3_benchmark": false,
    "synthetic_profile": "expansive"
  },
  "history": {
    "loss": [
      0.08427789062261581
    ],
    "mse": [
      0.015657206997275352
    ],
    "psnr_metric": [
      19.20149803161621
    ],
    "snr_db_metric": [
      -0.10396277904510498
    ],
    "val_loss": [
      0.026314787566661835
    ],
    "val_mse": [
      0.0023533469066023827
    ],
    "val_psnr_metric": [
      26.814273834228516
    ],
    "val_snr_db_metric": [
      0.07866333425045013
    ],
    "learning_rate": [
      0.00011999999696854502
    ]
  },
  "test_metrics": {
    "loss": 0.07036590576171875,
    "mse": 0.008565266616642475,
    "psnr_metric": 20.94500732421875,
    "snr_db_metric": 0.07499679923057556
  },
  "preview_plot": "models/audio_realdata_opt_smoke/20260629_145612/audio_preview_waveform.png",
  "preview_wav": "models/audio_realdata_opt_smoke/20260629_145612/audio_preview_reconstruction.wav",
  "benchmark": null
}
```
