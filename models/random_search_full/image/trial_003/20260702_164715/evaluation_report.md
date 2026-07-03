# Evaluation Report

- Source: `evaluation_report.md`

```json
{
  "timestamp": "2026-07-02T13:48:06.575544Z",
  "config": {
    "preset": "custom",
    "use_cifar10": false,
    "data_dir": "/Users/tyejaedon/PycharmProjects/AI_Compressor/data/_random_search_missing",
    "valid_dir": "data/cifar10_valid",
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/image/trial_003",
    "block_size": 96,
    "latent_dim": 128,
    "epochs": 4,
    "batch_size": 4,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lr": 9.3e-05,
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
      0.0986151322722435,
      0.06932009011507034,
      0.04332044720649719,
      0.028629491105675697
    ],
    "mse": [
      0.041688207536935806,
      0.030624644830822945,
      0.01950896345078945,
      0.01243528537452221
    ],
    "psnr_metric": [
      13.807866096496582,
      15.2105073928833,
      17.13640785217285,
      19.120685577392578
    ],
    "ssim_metric": [
      0.07267638295888901,
      0.41679924726486206,
      0.681900680065155,
      0.8088126182556152
    ],
    "val_loss": [
      0.09060860425233841,
      0.0503954254090786,
      0.03591790422797203,
      0.023685602471232414
    ],
    "val_mse": [
      0.038596514612436295,
      0.02315363846719265,
      0.015823138877749443,
      0.010542099364101887
    ],
    "val_psnr_metric": [
      14.135039329528809,
      16.35636329650879,
      18.0078125,
      19.770221710205078
    ],
    "val_ssim_metric": [
      0.16456377506256104,
      0.6231555938720703,
      0.7420174479484558,
      0.8627024292945862
    ],
    "learning_rate": [
      9.300000237999484e-05,
      9.300000237999484e-05,
      9.300000237999484e-05,
      9.300000237999484e-05
    ]
  },
  "test_metrics": {
    "loss": 0.02341282367706299,
    "mse": 0.010347043164074421,
    "psnr_metric": 19.8521785736084,
    "ssim_metric": 0.8633803725242615
  },
  "plot_path": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/random_search_full/image/trial_003/20260702_164715/training_metrics.png"
}
```
