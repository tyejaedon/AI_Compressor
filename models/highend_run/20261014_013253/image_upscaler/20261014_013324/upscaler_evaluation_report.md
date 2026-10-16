# Learned Upscaler Evaluation Report

- Source: `upscaler_evaluation_report.md`

```json
{
  "timestamp": "2026-10-14T00:13:35.608279Z",
  "config": {
    "preset": "custom",
    "use_cifar10": true,
    "cifar_train_limit": 20000,
    "cifar_test_limit": 5000,
    "data_dir": "",
    "train_dir": "",
    "val_dir": "",
    "test_dir": "",
    "split_file_limit": 0,
    "real_only": false,
    "real_file_limit": 0,
    "min_real_images": 12,
    "min_split_images": 3,
    "output_root": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/highend_run/20261014_013253/image_upscaler",
    "params_file": "",
    "block_size": 32,
    "upscale_factor": 2,
    "degrade_interp": "bicubic",
    "degrade_noise_std": 0.015,
    "model_base_filters": 40,
    "model_kernel_size": 3,
    "num_residual_blocks": 5,
    "epochs": 7,
    "batch_size": 32,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lr": 0.00025,
    "train_patches_per_image": 4,
    "target_psnr": 35.0,
    "sample_count": 8,
    "export_tflite": false,
    "tflite_fp16": false
  },
  "split_info": {
    "source": "cifar10",
    "note": "CIFAR-10 is natively 32x32; high-res targets are bicubic-upsampled to --block-size before degrading.",
    "train": 14000,
    "validation": 3000,
    "test": 5000,
    "block_size": 32,
    "upscale_factor": 2
  },
  "history": {
    "loss": [
      0.03717426955699921,
      0.02242824248969555,
      0.021111495792865753,
      0.020416058599948883,
      0.019990911707282066,
      0.019680079072713852,
      0.019465168938040733
    ],
    "mse": [
      0.010099554434418678,
      0.003708668751642108,
      0.003243764629587531,
      0.003001933451741934,
      0.002861885353922844,
      0.0027614710852503777,
      0.0027118006255477667
    ],
    "psnr_metric": [
      21.784101486206055,
      24.82777214050293,
      25.423410415649414,
      25.777694702148438,
      25.99709701538086,
      26.16423225402832,
      26.24665641784668
    ],
    "ssim_metric": [
      0.8094427585601807,
      0.8715134859085083,
      0.8776318430900574,
      0.880916178226471,
      0.8829501867294312,
      0.8844383358955383,
      0.8855760097503662
    ],
    "val_loss": [
      0.023450015112757683,
      0.021727891638875008,
      0.020813612267374992,
      0.020066868513822556,
      0.019732272252440453,
      0.01957879215478897,
      0.019349033012986183
    ],
    "val_mse": [
      0.004063768312335014,
      0.0034992562141269445,
      0.003154595149680972,
      0.0028331272769719362,
      0.0027757021598517895,
      0.00267720315605402,
      0.0026414452586323023
    ],
    "val_psnr_metric": [
      24.419424057006836,
      25.03135871887207,
      25.536081314086914,
      26.060842514038086,
      26.13475799560547,
      26.32257652282715,
      26.375137329101562
    ],
    "val_ssim_metric": [
      0.866707980632782,
      0.8749881386756897,
      0.8791344165802002,
      0.8822907209396362,
      0.8841952681541443,
      0.8846620321273804,
      0.8859926462173462
    ],
    "learning_rate": [
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628,
      0.0002500000118743628
    ]
  },
  "test_metrics": {
    "loss": 0.01946624554693699,
    "mse": 0.0026747777592390776,
    "psnr_metric": 26.3378849029541,
    "ssim_metric": 0.8855071067810059
  },
  "bicubic_baseline": {
    "bicubic_psnr": 26.36417158090385,
    "bicubic_ssim": 0.870505374328346
  },
  "plot_path": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/highend_run/20261014_013253/image_upscaler/20261014_013324/training_metrics.png",
  "comparison_panel_path": "/Users/tyejaedon/PycharmProjects/AI_Compressor/models/highend_run/20261014_013253/image_upscaler/20261014_013324/upscaler_comparison_panel.png"
}
```
