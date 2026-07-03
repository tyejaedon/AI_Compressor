# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T13:59:36.830937Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_001",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 0.5,
    "latent_dim": 320,
    "batch_size": 8,
    "epochs": 4,
    "lr": 0.0001979,
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
      0.09605338424444199
    ],
    "mse": [
      0.02307307720184326
    ],
    "psnr_metric": [
      19.517866134643555
    ],
    "snr_db_metric": [
      9.161006927490234
    ],
    "val_loss": [
      0.0426434725522995
    ],
    "val_mse": [
      0.0023570836056023836
    ],
    "val_psnr_metric": [
      26.545330047607422
    ],
    "val_snr_db_metric": [
      15.86004638671875
    ],
    "learning_rate": [
      0.00019789999350905418
    ]
  },
  "test_metrics": {
    "loss": 0.04598380625247955,
    "mse": 0.0034201075322926044,
    "psnr_metric": 25.413921356201172,
    "snr_db_metric": 15.196470260620117
  },
  "preview_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_001/20260702_165754/audio_preview_waveform.png",
  "preview_wav": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_001/20260702_165754/audio_preview_reconstruction.wav",
  "metrics_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_001/20260702_165754/audio_training_metrics.png",
  "benchmark": null
}
```
