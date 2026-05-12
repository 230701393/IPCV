"""
Smoke Density Estimation & Air Quality Monitoring
Pipeline: DCP Haze Removal + CLAHE Clarity Enhancement
Dataset: RESIDE Indoor Haze (1399 images)
"""

import cv2
import numpy as np
from pathlib import Path
import json
import time


# ─────────────────────────────────────────────
# 1. DARK CHANNEL PRIOR (DCP) HAZE REMOVAL
# ─────────────────────────────────────────────

def dark_channel(image, patch_size=15):
    """Compute the dark channel of a hazy image."""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(image, dark_ch, top_percent=0.001):
    """Estimate global atmospheric light A from brightest dark channel pixels."""
    h, w = dark_ch.shape
    num_pixels = h * w
    num_brightest = max(int(num_pixels * top_percent), 1)

    flat_dark = dark_ch.flatten()
    flat_image = image.reshape(-1, 3)

    indices = np.argsort(flat_dark)[-num_brightest:]
    brightest_pixels = flat_image[indices]
    atmospheric_light = np.max(brightest_pixels, axis=0)
    return atmospheric_light.astype(np.float64)


def estimate_transmission(image, atmospheric_light, omega=0.95, patch_size=15):
    """Estimate transmission map t(x) = 1 - omega * dark_channel(I/A)."""
    img_norm = image.astype(np.float64) / (atmospheric_light + 1e-6)
    img_norm = np.clip(img_norm, 0, 1)
    dark_norm = dark_channel(img_norm, patch_size)
    transmission = 1.0 - omega * dark_norm
    return transmission


def guided_filter(guide, src, radius=60, eps=1e-3):
    """Fast guided filter for transmission map refinement."""
    guide = guide.astype(np.float64) / 255.0
    src = src.astype(np.float64)

    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_gs = mean_gs - mean_guide * mean_src

    mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_g = mean_gg - mean_guide * mean_guide

    a = cov_gs / (var_g + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output = mean_a * guide + mean_b
    return output


def recover_scene_radiance(image, transmission, atmospheric_light, t0=0.1):
    """Recover haze-free scene radiance J(x)."""
    image = image.astype(np.float64)
    t = np.maximum(transmission, t0)
    t3 = np.stack([t, t, t], axis=2)

    J = (image - atmospheric_light) / t3 + atmospheric_light
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def dcp_dehaze(image, patch_size=15, omega=0.95, guided=True):
    """Full DCP dehazing pipeline."""
    dark_ch = dark_channel(image, patch_size)
    atm_light = estimate_atmospheric_light(image, dark_ch)
    transmission = estimate_transmission(image, atm_light, omega, patch_size)

    if guided:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transmission = guided_filter(gray, transmission)

    dehazed = recover_scene_radiance(image, transmission, atm_light)
    return dehazed, transmission, atm_light


# ─────────────────────────────────────────────
# 2. CLAHE CLARITY ENHANCER
# ─────────────────────────────────────────────

def clahe_enhance(image, clip_limit=3.0, tile_grid=(8, 8)):
    """Apply CLAHE on L channel of LAB colorspace for perceptual enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced


def adaptive_sharpen(image, strength=0.5):
    """Unsharp masking for edge sharpening post-enhancement."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened


# ─────────────────────────────────────────────
# 3. SMOKE DENSITY METRICS
# ─────────────────────────────────────────────

def compute_smoke_density(image, transmission):
    """
    Estimate smoke/haze density from transmission map.
    Density = 1 - mean(transmission) → higher = more haze.
    """
    density = 1.0 - np.mean(transmission)
    return float(np.clip(density, 0, 1))


def compute_visibility_score(image):
    """
    Visibility metric based on image gradient energy (Laplacian variance).
    Higher = sharper/clearer image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = np.var(lap)
    return float(score)


def compute_contrast_ratio(image):
    """Michelson contrast ratio of luminance channel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    lmax, lmin = gray.max(), gray.min()
    if lmax + lmin == 0:
        return 0.0
    return float((lmax - lmin) / (lmax + lmin))


def compute_fog_density_index(image):
    """
    FDI: Ratio of near-white pixels (typical haze signature).
    Normalized 0–1; higher = more fog/smoke.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > 200)
    fdi = white_pixels / gray.size
    return float(fdi)


def estimate_aqi_from_density(smoke_density):
    """
    Heuristic mapping: smoke density → AQI band.
    Based on PM2.5 correlations from RESIDE benchmark literature.
    """
    d = smoke_density
    if d < 0.15:
        return {"aqi": int(d * 333), "category": "Good",         "color": "#00e400"}
    elif d < 0.30:
        return {"aqi": int(51 + (d - 0.15) * 660), "category": "Moderate",      "color": "#ffff00"}
    elif d < 0.50:
        return {"aqi": int(101 + (d - 0.30) * 500),"category": "Unhealthy (Sensitive)", "color": "#ff7e00"}
    elif d < 0.70:
        return {"aqi": int(151 + (d - 0.50) * 500),"category": "Unhealthy",     "color": "#ff0000"}
    elif d < 0.85:
        return {"aqi": int(201 + (d - 0.70) * 667),"category": "Very Unhealthy","color": "#8f3f97"}
    else:
        return {"aqi": int(301 + (d - 0.85) * 1327),"category": "Hazardous",    "color": "#7e0023"}


# ─────────────────────────────────────────────
# 4. FULL PROCESSING PIPELINE
# ─────────────────────────────────────────────

def process_image(image_path, output_dir=None):
    """
    Full pipeline:
      1. Load hazy image
      2. DCP dehazing
      3. CLAHE + sharpen enhancement
      4. Compute all metrics
      5. Save outputs if output_dir given
    Returns metrics dict.
    """
    img_path = Path(image_path)
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    t0 = time.time()

    # ── Step 1: DCP Haze Removal
    dehazed, transmission, atm_light = dcp_dehaze(image)

    # ── Step 2: CLAHE Enhancement
    enhanced = clahe_enhance(dehazed, clip_limit=3.0)
    enhanced = adaptive_sharpen(enhanced, strength=0.4)

    elapsed = time.time() - t0

    # ── Step 3: Metrics
    smoke_density    = compute_smoke_density(image, transmission)
    visibility_orig  = compute_visibility_score(image)
    visibility_enh   = compute_visibility_score(enhanced)
    contrast_orig    = compute_contrast_ratio(image)
    contrast_enh     = compute_contrast_ratio(enhanced)
    fdi              = compute_fog_density_index(image)
    aqi_info         = estimate_aqi_from_density(smoke_density)

    visibility_gain  = ((visibility_enh - visibility_orig) / (visibility_orig + 1e-6)) * 100

    metrics = {
        "filename":        img_path.name,
        "smoke_density":   round(smoke_density, 4),
        "fog_density_idx": round(fdi, 4),
        "visibility_orig": round(visibility_orig, 2),
        "visibility_enh":  round(visibility_enh, 2),
        "visibility_gain_pct": round(visibility_gain, 1),
        "contrast_orig":   round(contrast_orig, 4),
        "contrast_enh":    round(contrast_enh, 4),
        "atm_light":       atm_light.tolist(),
        "aqi":             aqi_info["aqi"],
        "aqi_category":    aqi_info["category"],
        "aqi_color":       aqi_info["color"],
        "processing_time_s": round(elapsed, 3),
    }

    # ── Step 4: Save outputs
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = img_path.stem

        cv2.imwrite(str(out / f"{stem}_dehazed.png"),  dehazed)
        cv2.imwrite(str(out / f"{stem}_enhanced.png"), enhanced)

        # Transmission heatmap
        t_vis = (transmission * 255).astype(np.uint8)
        t_color = cv2.applyColorMap(t_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(out / f"{stem}_transmission.png"), t_color)

        # Side-by-side comparison
        h = min(image.shape[0], 400)
        scale = h / image.shape[0]
        w = int(image.shape[1] * scale)
        orig_r  = cv2.resize(image,    (w, h))
        enh_r   = cv2.resize(enhanced, (w, h))
        divider = np.ones((h, 4, 3), dtype=np.uint8) * 80
        comparison = np.hstack([orig_r, divider, enh_r])
        cv2.imwrite(str(out / f"{stem}_comparison.png"), comparison)

        with open(str(out / f"{stem}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics, image, dehazed, enhanced, transmission


# ─────────────────────────────────────────────
# 5. BATCH PROCESSING (RESIDE Dataset)
# ─────────────────────────────────────────────

def batch_process_reside(dataset_dir, output_dir="output_results", max_images=None):
    """Process RESIDE Indoor dataset images in batch."""
    dataset_path = Path(dataset_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_files = [
        p for p in dataset_path.rglob("*")
        if p.suffix.lower() in extensions
    ]

    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images in {dataset_dir}")

    all_metrics = []
    for i, img_path in enumerate(image_files, 1):
        try:
            metrics, *_ = process_image(img_path, output_dir=output_dir)
            all_metrics.append(metrics)
            print(
                f"[{i:4d}/{len(image_files)}] {metrics['filename']:<30s} "
                f"Density={metrics['smoke_density']:.3f}  "
                f"AQI={metrics['aqi']:3d} ({metrics['aqi_category']})  "
                f"Vis+{metrics['visibility_gain_pct']:+.1f}%"
            )
        except Exception as e:
            print(f"  ✗ Error on {img_path.name}: {e}")

    # Summary statistics
    if all_metrics:
        densities = [m["smoke_density"]   for m in all_metrics]
        gains     = [m["visibility_gain_pct"] for m in all_metrics]
        aqis      = [m["aqi"]             for m in all_metrics]

        summary = {
            "total_images":          len(all_metrics),
            "avg_smoke_density":     round(np.mean(densities), 4),
            "avg_visibility_gain":   round(np.mean(gains), 2),
            "avg_aqi":               round(np.mean(aqis), 1),
            "max_smoke_density":     round(np.max(densities), 4),
            "min_smoke_density":     round(np.min(densities), 4),
            "aqi_distribution": {
                "Good":                 sum(1 for m in all_metrics if m["aqi_category"] == "Good"),
                "Moderate":             sum(1 for m in all_metrics if m["aqi_category"] == "Moderate"),
                "Unhealthy (Sensitive)":sum(1 for m in all_metrics if "Sensitive" in m["aqi_category"]),
                "Unhealthy":            sum(1 for m in all_metrics if m["aqi_category"] == "Unhealthy"),
                "Very Unhealthy":       sum(1 for m in all_metrics if m["aqi_category"] == "Very Unhealthy"),
                "Hazardous":            sum(1 for m in all_metrics if m["aqi_category"] == "Hazardous"),
            }
        }

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(str(out_path / "batch_summary.json"), "w") as f:
            json.dump({"summary": summary, "images": all_metrics}, f, indent=2)

        print("\n── Batch Summary ──────────────────────────────────")
        print(f"  Images processed  : {summary['total_images']}")
        print(f"  Avg smoke density : {summary['avg_smoke_density']}")
        print(f"  Avg visibility gain: +{summary['avg_visibility_gain']}%")
        print(f"  Avg AQI           : {summary['avg_aqi']}")
        print(f"  AQI distribution  : {summary['aqi_distribution']}")
        print(f"  Results saved to  : {output_dir}/")

    return all_metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # Single image
        path = sys.argv[1]
        print(f"Processing: {path}")
        metrics, orig, dehazed, enhanced, trans = process_image(
            path, output_dir="single_output"
        )
        print(json.dumps(metrics, indent=2))

    elif len(sys.argv) >= 3 and sys.argv[1] == "--batch":
        # Batch: python haze_pipeline.py --batch /path/to/reside [max_n]
        dataset_dir = sys.argv[2]
        max_n = int(sys.argv[3]) if len(sys.argv) > 3 else None
        batch_process_reside(dataset_dir, max_images=max_n)

    else:
        print("Usage:")
        print("  Single image : python haze_pipeline.py image.jpg")
        print("  Batch (RESIDE): python haze_pipeline.py --batch /data/reside [max_images]")
