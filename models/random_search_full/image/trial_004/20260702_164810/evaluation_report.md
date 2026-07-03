# Evaluation Report

- Source: `evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T13:48:50.499272Z",
  "config": {
    "preset": "custom",
    "use_cifar10": false,
    "data_dir": "/Users/tyejaedon/PycharmProjects/AI_Compressor/data/_random_search_missing",
    "valid_dir": "data/cifar10_valid",
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/image/trial_004",
    "block_size": 96,
    "latent_dim": 48,
    "epochs": 4,
    "batch_size": 8,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lr": 0.0002533,
    "allow_synthetic": true,
    "dummy_samples": 180,
    "extensive_dummy_samples": 240,
    "auto_generate_dummy": true,
    "cifar_train_limit": 0,
    "cifar_test_limit": 0,
    "target_psnr": 22.0,
    "export_tflite": false,
    "tflite_fp16": false
  },
  "split_info": {
    "source": "synthetic",
    "train": 126,
    "validation": 27,
    "test": 27
  },
  "history": {
    "loss": [
      0.09790802747011185,
      0.07240189611911774,
      0.050893254578113556,
      0.03998827189207077
    ],
    "mse": [
      0.041464194655418396,
      0.0328538715839386,
      0.023487387225031853,
      0.018096692860126495
    ],
    "psnr_metric": [
      13.831624031066895,
      14.865596771240234,
      16.311172485351562,
      17.45437240600586
    ],
    "ssim_metric": [
      0.08227087557315826,
      0.4036405682563782,
      0.6213402152061462,
      0.7168576121330261
    ],
    "val_loss": [
      0.08726464956998825,
      0.057729121297597885,
      0.0457661934196949,
      0.034857407212257385
    ],
    "val_mse": [
      0.03781416267156601,
      0.026696287095546722,
      0.021470725536346436,
      0.015587987378239632
    ],
    "val_psnr_metric": [
      14.224003791809082,
      15.73921012878418,
      16.684492111206055,
      18.069000244140625
    ],
    "val_ssim_metric": [
      0.21337498724460602,
      0.5554720163345337,
      0.6786342859268188,
      0.7613195776939392
    ],
    "learning_rate": [
      0.0002533000078983605,
      0.0002533000078983605,
      0.0002533000078983605,
      0.0002533000078983605
    ]
  },
  "test_metrics": {
    "loss": 0.03480413928627968,
    "mse": 0.015573682263493538,
    "psnr_metric": 18.060325622558594,
    "ssim_metric": 0.7620533108711243
  },
  "plot_path": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/image/trial_004/20260702_164810/training_metrics.png"
}
```
