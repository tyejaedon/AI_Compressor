# Audio Evaluation Report

- Source: `audio_evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T13:57:28.838259Z",
  "config": {
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_003",
    "preset": "custom",
    "sample_rate": 16000,
    "clip_seconds": 0.5,
    "latent_dim": 256,
    "batch_size": 12,
    "epochs": 4,
    "lr": 6.29e-05,
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
      0.17556072771549225,
      0.07187782227993011,
      0.05249181017279625
    ],
    "mse": [
      0.062253545969724655,
      0.00929292757064104,
      0.004727967549115419
    ],
    "psnr_metric": [
      12.717889785766602,
      20.601783752441406,
      23.599349975585938
    ],
    "snr_db_metric": [
      2.3851327896118164,
      10.255975723266602,
      13.206979751586914
    ],
    "val_loss": [
      0.08086775988340378,
      0.056955788284540176,
      0.04505237564444542
    ],
    "val_mse": [
      0.012035854160785675,
      0.005344064440578222,
      0.003569834865629673
    ],
    "val_psnr_metric": [
      19.262393951416016,
      23.015676498413086,
      24.692853927612305
    ],
    "val_snr_db_metric": [
      8.632009506225586,
      12.486429214477539,
      14.05184555053711
    ],
    "learning_rate": [
      6.289999873843044e-05,
      6.289999873843044e-05,
      6.289999873843044e-05
    ]
  },
  "test_metrics": {
    "loss": 0.0463777631521225,
    "mse": 0.003959535621106625,
    "psnr_metric": 24.427879333496094,
    "snr_db_metric": 14.218048095703125
  },
  "preview_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_003/20260702_165401/audio_preview_waveform.png",
  "preview_wav": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_003/20260702_165401/audio_preview_reconstruction.wav",
  "metrics_plot": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/audio/trial_003/20260702_165401/audio_training_metrics.png",
  "benchmark": null
}
```
