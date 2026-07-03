# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T13:53:53.344528Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_002",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "latent_dim": 160,
    "batch_size": 16,
    "epochs": 4,
    "lr": 0.0001783,
    "seed": 42,
    "data_dir": "data/AudioData/ESC-50-master/audio",
    "real_val_ratio": 0.15,
    "real_test_ratio": 0.15,
    "real_file_limit": 0,
    "synthetic_only": true,
    "train_samples": 600,
    "val_samples": 150,
    "test_samples": 150,
    "target_psnr": 24.0,
    "shuffle_buffer": 1024,
    "export_tflite": false,
    "tflite_fp16": false,
    "run_mp3_benchmark": false,
    "synthetic_profile": "legacy"
  },
  "history": {
    "loss": [
      0.13419972360134125,
      0.05567559599876404
    ],
    "mse": [
      0.03994511440396309,
      0.005329846870154142
    ],
    "psnr_metric": [
      15.66964054107666,
      23.052518844604492
    ],
    "snr_db_metric": [
      5.3866119384765625,
      12.790081977844238
    ],
    "val_loss": [
      0.06914333999156952,
      0.04482623189687729
    ],
    "val_mse": [
      0.00868525356054306,
      0.0027761301025748253
    ],
    "val_psnr_metric": [
      20.667156219482422,
      25.659404754638672
    ],
    "val_snr_db_metric": [
      10.38209342956543,
      15.432205200195312
    ],
    "learning_rate": [
      0.00017830000433605164,
      0.00017830000433605164
    ]
  },
  "test_metrics": {
    "loss": 0.046463143080472946,
    "mse": 0.0033675197046250105,
    "psnr_metric": 24.852813720703125,
    "snr_db_metric": 14.467424392700195
  },
  "preview_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_002/20260702_164942/audio_preview_waveform.png",
  "preview_wav": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_002/20260702_164942/audio_preview_reconstruction.wav",
  "metrics_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_002/20260702_164942/audio_training_metrics.png",
  "benchmark": null
}
```
