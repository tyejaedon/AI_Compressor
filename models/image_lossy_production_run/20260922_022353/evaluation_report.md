# Image Lossy Compression Evaluation Report

- Source: `evaluation_report.md`

```json
{
  "timestamp": "2026-09-21T23:28:23.257957Z",
  "config": {
    "preset": "custom",
    "use_cifar10": true,
    "data_dir": "data",
    "train_dir": "",
    "val_dir": "",
    "test_dir": "",
    "split_file_limit": 0,
    "output_root": "models/image_lossy_production_run",
    "params_file": "",
    "block_size": 32,
    "latent_dim": 24,
    "model_base_filters": 32,
    "model_kernel_size": 3,
    "epochs": 10,
    "batch_size": 16,
    "seed": 42,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "lr": 0.0002,
    "real_only": false,
    "real_file_limit": 0,
    "min_real_images": 300,
    "min_split_images": 20,
    "target_psnr": 25.0,
    "rate_lambda": 0.0008,
    "jpeg_quality": 82,
    "sample_count": 64,
    "run_baseline_benchmark": true,
    "target_compression_ratio": null,
    "export_tflite": false,
    "tflite_fp16": false,
    "rate_loss_mode": "entropy",
    "enable_entropy_coding": true,
    "entropy_coding_samples": 16,
    "latent_bit_depth": 6,
    "quant_noise_anneal": true
  },
  "split_info": {
    "source": "cifar10",
    "train": 35000,
    "validation": 7500,
    "test": 10000
  },
  "history": {
    "loss": [
      0.0456804484128952,
      0.021515440195798874,
      0.017748167738318443
    ],
    "mse": [
      0.015262660570442677,
      0.005083069670945406,
      0.003917192108929157
    ],
    "psnr_metric": [
      19.742538452148438,
      23.620922088623047,
      24.757286071777344
    ],
    "ssim_metric": [
      0.7148693203926086,
      0.8626521229743958,
      0.8876779675483704
    ],
    "val_loss": [
      0.02569701336324215,
      0.018924666568636894,
      0.01658467762172222
    ],
    "val_mse": [
      0.0067101726308465,
      0.004229805897921324,
      0.0035391165874898434
    ],
    "val_psnr_metric": [
      22.495805740356445,
      24.414913177490234,
      25.247621536254883
    ],
    "val_ssim_metric": [
      0.837775468826294,
      0.8794745802879333,
      0.8952738046646118
    ],
    "learning_rate": [
      0.00019999999494757503,
      0.00019999999494757503,
      0.00019999999494757503
    ]
  },
  "test_metrics": {
    "loss": 0.016610050573945045,
    "mse": 0.0035480090882629156,
    "psnr_metric": 25.227075576782227,
    "ssim_metric": 0.8950949907302856
  },
  "lossy_size_stats": {
    "sample_dir": "models/image_lossy_production_run/20260922_022353/lossy_samples",
    "sample_count": 16,
    "jpeg_quality": 82,
    "avg_psnr": 24.635616302490234,
    "mean_original_jpeg_bytes": 685.0,
    "mean_reconstructed_jpeg_bytes": 618.375,
    "mean_reconstructed_png_bytes": 2244.75,
    "recon_jpeg_vs_orig_jpeg_ratio": 0.9027372262773723,
    "recon_png_vs_recon_jpeg_ratio": 3.6300788356579745
  },
  "matched_psnr_baseline_benchmark": {
    "available": true,
    "sample_count": 16,
    "model_target_psnr": 24.635616302490234,
    "model_reconstructed_jpeg_bytes": 618.375,
    "model_export_jpeg_quality": 82,
    "jpeg_baseline": {
      "codec": "jpeg",
      "available": true,
      "target_psnr": 24.635616302490234,
      "matched": {
        "quality": 20,
        "mean_psnr": 25.073241114616394,
        "mean_bytes": 419.1875
      },
      "candidates": [
        {
          "quality": 5,
          "mean_psnr": 20.348292112350464,
          "mean_bytes": 330.5625
        },
        {
          "quality": 10,
          "mean_psnr": 22.766945481300354,
          "mean_bytes": 367.625
        },
        {
          "quality": 15,
          "mean_psnr": 24.170923948287964,
          "mean_bytes": 397.4375
        },
        {
          "quality": 20,
          "mean_psnr": 25.073241114616394,
          "mean_bytes": 419.1875
        },
        {
          "quality": 25,
          "mean_psnr": 25.78477454185486,
          "mean_bytes": 440.75
        },
        {
          "quality": 30,
          "mean_psnr": 26.28005599975586,
          "mean_bytes": 459.0
        },
        {
          "quality": 35,
          "mean_psnr": 26.715341448783875,
          "mean_bytes": 475.375
        },
        {
          "quality": 40,
          "mean_psnr": 27.11963188648224,
          "mean_bytes": 488.9375
        },
        {
          "quality": 45,
          "mean_psnr": 27.475231766700745,
          "mean_bytes": 504.6875
        },
        {
          "quality": 50,
          "mean_psnr": 27.78060519695282,
          "mean_bytes": 518.8125
        },
        {
          "quality": 55,
          "mean_psnr": 28.11734700202942,
          "mean_bytes": 533.25
        },
        {
          "quality": 60,
          "mean_psnr": 28.431426286697388,
          "mean_bytes": 550.125
        },
        {
          "quality": 65,
          "mean_psnr": 28.885807871818542,
          "mean_bytes": 570.5625
        },
        {
          "quality": 70,
          "mean_psnr": 29.407655835151672,
          "mean_bytes": 595.625
        },
        {
          "quality": 75,
          "mean_psnr": 29.95944094657898,
          "mean_bytes": 622.8125
        },
        {
          "quality": 80,
          "mean_psnr": 30.767322421073914,
          "mean_bytes": 664.5
        },
        {
          "quality": 85,
          "mean_psnr": 31.684731364250183,
          "mean_bytes": 719.75
        },
        {
          "quality": 90,
          "mean_psnr": 33.09095060825348,
          "mean_bytes": 811.375
        },
        {
          "quality": 95,
          "mean_psnr": 34.408981800079346,
          "mean_bytes": 991.25
        }
      ]
    },
    "webp_baseline": {
      "codec": "webp",
      "available": false,
      "reason": "TensorFlow build does not expose tf.io.encode_webp/decode_webp"
    },
    "model_vs_jpeg_matched_size_ratio": 1.4751751900998957
  },
  "latent_rate_stats": {
    "latent_entropy_bits_per_symbol": 4.606657417579034,
    "latent_symbols_per_image": 1536,
    "latent_bits_per_pixel": 6.909986126368551,
    "estimated_bitstream_bytes_per_image": 884.4782241751745
  },
  "entropy_coding": {
    "available": true,
    "sample_count": 16,
    "symbols_per_image": 1536,
    "total_symbols_encoded": 24576,
    "actual_compressed_bytes": 14164,
    "actual_compressed_bytes_per_image": 885.25,
    "entropy_estimate_bytes_per_image": 884.4782241751745,
    "note": "Static per-sample frequency table is not itself transmitted/counted; a deployed coder would need an agreed-upon or separately-coded entropy model."
  },
  "plot_path": "models/image_lossy_production_run/20260922_022353/training_metrics.png"
}
```
