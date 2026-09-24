#!/usr/bin/env python3
"""Run image autoencoder reconstruction and upscale outputs to original size or 720p."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from report_markdown import read_markdown_json_report, write_markdown_json_report


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct and upscale real images with a trained autoencoder.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/production_bundle/best_20260629_142034/image/best_model.keras",
        help="Path to trained Keras image model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ImageData/archive/data",
        help="Directory with input images (recursive)",
    )
    parser.add_argument("--count", type=int, default=8, help="Number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--upscale-mode",
        type=str,
        default="original",
        choices=["original", "720p"],
        help="Target output size mode",
    )
    parser.add_argument(
        "--upscaler-mode",
        type=str,
        default="bicubic",
        choices=["bicubic", "learned"],
        help="How to upscale the model's reconstruction to the target size: classic BICUBIC (default, "
        "unchanged prior behavior) or a learned super-resolution model (see train_upscaler_image_local.py)",
    )
    parser.add_argument(
        "--learned-upscaler-model",
        type=str,
        default="",
        help="Path to a .keras model trained by train_upscaler_image_local.py; required for --upscaler-mode learned "
        "(falls back to BICUBIC with a warning if omitted or the path can't be loaded)",
    )
    parser.add_argument(
        "--upscaler-tile-size",
        type=int,
        default=128,
        help="Tile size (input-resolution pixels) used when running the learned upscaler, to bound memory on large images",
    )
    parser.add_argument(
        "--upscaler-tile-overlap",
        type=int,
        default=16,
        help="Overlap (input-resolution pixels) between tiles for the learned upscaler, blended with a linear ramp",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/upscaled_reconstructions",
        help="Output root directory",
    )
    return parser.parse_args()


def _read_latent_dim_from_report(report_path):
    payload = read_markdown_json_report(report_path)
    if not payload:
        return None
    latent = payload.get("config", {}).get("latent_dim")
    if isinstance(latent, int) and latent > 0:
        return latent
    return None


def load_model_with_fallback(model_path):
    model_path = Path(model_path)
    try:
        # Local trusted artifact; required for Lambda layers in some training runs.
        return tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    except Exception as err:
        weights_path = model_path.with_name("production_model.weights.h5")
        report_path = model_path.with_name("evaluation_report.md")
        if not (weights_path.exists() and report_path.exists()):
            raise err

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
        from train_autoencoder_image_local import build_autoencoder

        latent_dim = _read_latent_dim_from_report(report_path) or 256
        model = build_autoencoder(latent_dim)
        # Build variables before loading weights.
        _ = model(tf.zeros((1, 96, 96, 3), dtype=tf.float32), training=False)
        model.load_weights(str(weights_path))
        return model


def load_learned_upscaler(model_path):
    """Load a model trained by train_upscaler_image_local.py. Returns (model, upscale_factor) or (None, None)."""
    path = Path(model_path)
    if not model_path or not path.exists():
        print(f"[!] --learned-upscaler-model not found ('{model_path}'); falling back to BICUBIC.")
        return None, None

    # Importing the trainer module runs its @register_keras_serializable decorators
    # (e.g. PixelShuffle) so tf.keras.models.load_model can resolve the custom layer
    # by registered name, even though this script never otherwise uses the module.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    try:
        import train_upscaler_image_local  # noqa: F401
    except Exception as exc:
        print(f"[!] Could not import train_upscaler_image_local for custom layer registration ({exc}).")

    try:
        model = tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
    except Exception as exc:
        print(f"[!] Failed to load learned upscaler model ({exc}); falling back to BICUBIC.")
        return None, None

    probe_size = 16
    probe = tf.zeros((1, probe_size, probe_size, 3), dtype=tf.float32)
    try:
        out = model(probe, training=False)
    except Exception as exc:
        print(f"[!] Learned upscaler model failed a probe forward pass ({exc}); falling back to BICUBIC.")
        return None, None
    factor = max(1, round(int(out.shape[1]) / probe_size))
    return model, factor


def _blend_ramp_1d(size, overlap):
    """1D blend weights: linear ramp up/down over `overlap` samples at each edge, 1.0 in the middle."""
    weights = np.ones(size, dtype=np.float32)
    if overlap > 0 and size > 2 * overlap:
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        weights[:overlap] = ramp
        weights[-overlap:] = ramp[::-1]
    return weights


def apply_learned_upscaler_tiled(model, image_np, upscale_factor, tile_size=128, tile_overlap=16):
    """Run a learned upscaler over a (possibly large) image using overlapping tiles.

    Tiles are blended with a linear ramp in the overlap region to avoid visible
    seams, and the last row/column of tiles is snapped to the image edge so the
    whole image is covered even when it isn't an exact multiple of the stride.
    Bounds memory use so this can run on full-resolution real photos on an M1.
    """
    h, w = int(image_np.shape[0]), int(image_np.shape[1])
    tile = max(16, int(tile_size))
    overlap = max(0, min(int(tile_overlap), tile // 2 - 1))
    stride = max(1, tile - overlap)

    if h <= tile and w <= tile:
        pred = model.predict(image_np[np.newaxis, ...], verbose=0)[0]
        return np.clip(pred, 0.0, 1.0)

    ys = sorted(set(list(range(0, max(1, h - tile + 1), stride)) + [max(0, h - tile)]))
    xs = sorted(set(list(range(0, max(1, w - tile + 1), stride)) + [max(0, w - tile)]))

    out_h, out_w = h * upscale_factor, w * upscale_factor
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_sum = np.zeros((out_h, out_w, 1), dtype=np.float32)

    ramp = _blend_ramp_1d(tile, overlap)
    tile_weight = (ramp[:, None] * ramp[None, :])[:, :, None]
    tile_weight_up = np.repeat(np.repeat(tile_weight, upscale_factor, axis=0), upscale_factor, axis=1)

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            patch = image_np[y0:y1, x0:x1, :]
            ph, pw = patch.shape[0], patch.shape[1]
            if ph < tile or pw < tile:
                patch = np.pad(patch, ((0, tile - ph), (0, tile - pw), (0, 0)), mode="reflect")

            pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
            pred = pred[: ph * upscale_factor, : pw * upscale_factor, :]
            w_mask = tile_weight_up[: ph * upscale_factor, : pw * upscale_factor, :]

            oy0, ox0 = y0 * upscale_factor, x0 * upscale_factor
            oy1, ox1 = oy0 + pred.shape[0], ox0 + pred.shape[1]
            canvas[oy0:oy1, ox0:ox1, :] += pred * w_mask
            weight_sum[oy0:oy1, ox0:ox1, :] += w_mask

    weight_sum = np.maximum(weight_sum, 1e-6)
    return np.clip(canvas / weight_sum, 0.0, 1.0)


def apply_learned_upscale_to_target(model, image_np, upscale_factor, target_w, target_h, tile_size, tile_overlap, max_passes=3):
    """Repeatedly apply the learned upscaler's native factor until >= target size, then
    bicubic-resize to the exact target dimensions (the learned factor rarely divides the
    target size evenly, e.g. arbitrary --upscale-mode 720p targets)."""
    current = image_np
    passes = 0
    while (current.shape[0] < target_h or current.shape[1] < target_w) and passes < max_passes:
        current = apply_learned_upscaler_tiled(model, current, upscale_factor, tile_size, tile_overlap)
        passes += 1
    if current.shape[0] != target_h or current.shape[1] != target_w:
        current = tf.image.resize(current, [target_h, target_w], method=tf.image.ResizeMethod.BICUBIC).numpy()
        current = np.clip(current, 0.0, 1.0)
    return current, passes


def collect_images(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def decode_image(path):
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def target_size_keep_aspect(orig_w, orig_h, mode):
    if mode == "original":
        return int(orig_w), int(orig_h)

    if mode == "720p":
        if orig_w >= orig_h:
            target_h = 720
            target_w = int(round(target_h * (orig_w / max(orig_h, 1))))
        else:
            target_w = 720
            target_h = int(round(target_w * (orig_h / max(orig_w, 1))))
        return int(target_w), int(target_h)

    raise ValueError("Unsupported upscale mode")


def save_comparison(original, recon_small, recon_upscaled, title, out_path, bicubic_upscaled=None):
    n_panels = 4 if bicubic_upscaled is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    axes[0].imshow(np.clip(original, 0.0, 1.0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(recon_small, 0.0, 1.0))
    axes[1].set_title("Reconstructed (model size)")
    axes[1].axis("off")

    if bicubic_upscaled is not None:
        axes[2].imshow(np.clip(bicubic_upscaled, 0.0, 1.0))
        axes[2].set_title("Upscaled (bicubic)")
        axes[2].axis("off")
        axes[3].imshow(np.clip(recon_upscaled, 0.0, 1.0))
        axes[3].set_title("Upscaled (learned)")
        axes[3].axis("off")
    else:
        axes[2].imshow(np.clip(recon_upscaled, 0.0, 1.0))
        axes[2].set_title("Upscaled")
        axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=165)
    plt.close(fig)


def main():
    args = parse_args()

    model = load_model_with_fallback(args.model)
    _, in_h, in_w, in_c = model.input_shape
    if in_c != 3:
        raise ValueError(f"Expected RGB model with 3 channels, got {model.input_shape}")

    dynamic_spatial = in_h is None or in_w is None

    effective_upscaler_mode = args.upscaler_mode
    learned_model, learned_factor = None, None
    if args.upscaler_mode == "learned":
        learned_model, learned_factor = load_learned_upscaler(args.learned_upscaler_model)
        if learned_model is None:
            effective_upscaler_mode = "bicubic"

    paths = collect_images(args.data_dir)
    if len(paths) == 0:
        raise ValueError(f"No images found in '{args.data_dir}'")

    rng = np.random.default_rng(args.seed)
    k = min(args.count, len(paths))
    picks = [paths[int(i)] for i in rng.choice(len(paths), size=k, replace=False)]

    run_dir = os.path.join(args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    out_recon = os.path.join(run_dir, "upscaled")
    out_comp = os.path.join(run_dir, "comparisons")
    os.makedirs(out_recon, exist_ok=True)
    os.makedirs(out_comp, exist_ok=True)

    samples = []
    for idx, path in enumerate(picks, start=1):
        img = decode_image(path)
        orig_h = int(img.shape[0])
        orig_w = int(img.shape[1])

        if dynamic_spatial:
            # Feed full-resolution image directly when the model supports dynamic spatial dims.
            model_in = img
        else:
            model_in = tf.image.resize(img, [in_h, in_w], method=tf.image.ResizeMethod.BILINEAR)
        model_in_h = int(model_in.shape[0])
        model_in_w = int(model_in.shape[1])
        pred = model.predict(model_in[tf.newaxis, ...], verbose=0)[0]

        target_w, target_h = target_size_keep_aspect(orig_w, orig_h, args.upscale_mode)
        bicubic_up = tf.image.resize(pred, [target_h, target_w], method=tf.image.ResizeMethod.BICUBIC).numpy()
        learned_passes = None
        if effective_upscaler_mode == "learned":
            pred_up, learned_passes = apply_learned_upscale_to_target(
                learned_model,
                pred.astype(np.float32),
                learned_factor,
                target_w,
                target_h,
                args.upscaler_tile_size,
                args.upscaler_tile_overlap,
            )
        else:
            pred_up = bicubic_up

        # Create comparison using original resized to target for visual parity and PSNR
        orig_target = tf.image.resize(img, [target_h, target_w], method=tf.image.ResizeMethod.BILINEAR).numpy()
        psnr_up = float(tf.image.psnr(orig_target, pred_up, max_val=1.0).numpy())
        psnr_bicubic = float(tf.image.psnr(orig_target, bicubic_up, max_val=1.0).numpy())

        stem = Path(path).stem
        up_path = os.path.join(out_recon, f"{idx:02d}_{stem}_upscaled.png")
        comp_path = os.path.join(out_comp, f"{idx:02d}_{stem}_comparison.png")

        tf.keras.utils.save_img(up_path, np.clip(pred_up, 0.0, 1.0))
        save_comparison(
            img.numpy(),
            pred,
            pred_up,
            f"{Path(path).name} | PSNR(upscaled)={psnr_up:.2f} dB",
            comp_path,
            bicubic_upscaled=bicubic_up if effective_upscaler_mode == "learned" else None,
        )

        samples.append(
            {
                "source": path,
                "name": Path(path).name,
                "original_size": [orig_w, orig_h],
                "model_input_size": [model_in_w, model_in_h],
                "upscaled_size": [target_w, target_h],
                "upscaler_mode": effective_upscaler_mode,
                "psnr_upscaled": psnr_up,
                "psnr_bicubic_baseline": psnr_bicubic,
                "learned_upscaler_passes": learned_passes,
                "upscaled_png": up_path,
                "comparison_png": comp_path,
            }
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "upscale_mode": args.upscale_mode,
        "upscaler_mode_requested": args.upscaler_mode,
        "upscaler_mode_effective": effective_upscaler_mode,
        "learned_upscaler_model": args.learned_upscaler_model or None,
        "output_dir": run_dir,
        "samples": samples,
    }
    report_path = os.path.join(run_dir, "upscale_report.md")
    write_markdown_json_report(report, report_path, title="Upscale Report")

    print(run_dir)
    print(report_path)


if __name__ == "__main__":
    main()

