# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T15:31:11.508946Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_goal30/audio/trial_001",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 0.5,
    "latent_dim": 160,
    "batch_size": 8,
    "epochs": 18,
    "lr": 7.42e-05,
    "seed": 42,
    "data_dir": "data/AudioData/ESC-50-master/audio",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_file_limit": 0,
    "synthetic_only": true,
    "train_samples": 5000,
    "val_samples": 1000,
    "test_samples": 1000,
    "target_psnr": 30.0,
    "shuffle_buffer": 1024,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp3_benchmark": false,
    "synthetic_profile": "legacy"
  },
  "history": {
    "loss": [
      0.046369053423404694
    ],
    "mse": [
      0.006367736030369997
    ],
    "psnr_metric": [
      27.73446273803711
    ],
    "snr_db_metric": [
      17.403806686401367
    ],
    "val_loss": [
      0.023490702733397484
    ],
    "val_mse": [
      0.0005797746707685292
    ],
    "val_psnr_metric": [
      32.531795501708984
    ],
    "val_snr_db_metric": [
      22.067720413208008
    ],
    "learning_rate": [
      7.419999747071415e-05
    ]
  },
  "test_metrics": {
    "loss": 0.02415713481605053,
    "mse": 0.0006248309509828687,
    "psnr_metric": 32.222923278808594,
    "snr_db_metric": 21.981340408325195
  },
  "preview_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_goal30/audio/trial_001/20260702_182301/audio_preview_waveform.png",
  "preview_wav": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_goal30/audio/trial_001/20260702_182301/audio_preview_reconstruction.wav",
  "metrics_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_goal30/audio/trial_001/20260702_182301/audio_training_metrics.png",
  "benchmark": null
}
```
