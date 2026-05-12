"""
Evaluation Metrics & Visualization Utilities
for Smoke Density / Haze Removal Quality Assessment
"""

import cv2
import numpy as np
import json
from pathlib import Path


# ─────────────────────────────────────────────
# IMAGE QUALITY METRICS
# ─────────────────────────────────────────────

def psnr(original, enhanced):
    """Peak Signal-to-Noise Ratio (dB). Higher = better."""
    mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def ssim(img1, img2):
    """
    Structural Similarity Index (simplified single-scale).
    Returns value in [-1, 1]; 1 = identical.
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return float(np.mean(ssim_map))


def entropy(image):
    """Shannon entropy of grayscale histogram. Higher = more detail/information."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def mean_gradient(image):
    """Mean gradient magnitude — higher = more edges/sharpness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag))


def color_cast_index(image):
    """
    Measure color channel imbalance (haze adds a grayish/white cast).
    Lower = more neutral/balanced colors.
    """
    b, g, r = cv2.split(image.astype(float))
    means = np.array([b.mean(), g.mean(), r.mean()])
    return float(np.std(means))


def evaluate_pair(hazy, enhanced, gt=None):
    """
    Full metric suite comparing hazy vs enhanced images.
    If ground truth provided, compute reference-based metrics.
    """
    metrics = {
        "psnr_hazy_vs_enhanced": round(psnr(hazy, enhanced), 2),
        "ssim_hazy_vs_enhanced": round(ssim(
            cv2.cvtColor(hazy, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        ), 4),
        "entropy_hazy":     round(entropy(hazy), 4),
        "entropy_enhanced": round(entropy(enhanced), 4),
        "entropy_gain":     round(entropy(enhanced) - entropy(hazy), 4),
        "gradient_hazy":    round(mean_gradient(hazy), 4),
        "gradient_enhanced":round(mean_gradient(enhanced), 4),
        "gradient_gain_pct":round(
            (mean_gradient(enhanced) - mean_gradient(hazy))
            / (mean_gradient(hazy) + 1e-6) * 100, 1
        ),
        "color_cast_hazy":     round(color_cast_index(hazy), 4),
        "color_cast_enhanced": round(color_cast_index(enhanced), 4),
    }

    if gt is not None:
        metrics["psnr_vs_gt"]  = round(psnr(gt, enhanced), 2)
        metrics["ssim_vs_gt"]  = round(ssim(
            cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        ), 4)

    return metrics


# ─────────────────────────────────────────────
# VISUALIZATION UTILITIES
# ─────────────────────────────────────────────

def make_analysis_grid(image, dehazed, enhanced, transmission, save_path=None):
    """
    Create a 2×3 analysis visualization grid:
    [Original Hazy | DCP Dehazed | CLAHE Enhanced]
    [Dark Channel   | Transmission Map | Difference Map]
    """
    target_h = 240
    def resize(img):
        h, w = img.shape[:2]
        s = target_h / h
        return cv2.resize(img, (int(w * s), target_h))

    from haze_pipeline import dark_channel

    # Row 1
    orig_r = resize(image)
    deh_r  = resize(dehazed)
    enh_r  = resize(enhanced)

    # Row 2
    dark = dark_channel(image)
    dark_colored = cv2.applyColorMap(dark, cv2.COLORMAP_INFERNO)
    dark_r = resize(dark_colored)

    t_vis = (np.clip(transmission, 0, 1) * 255).astype(np.uint8)
    t_color = cv2.applyColorMap(t_vis, cv2.COLORMAP_JET)
    t_r = resize(t_color)

    diff = cv2.absdiff(image, enhanced)
    diff_amp = cv2.convertScaleAbs(diff, alpha=3)
    diff_color = cv2.applyColorMap(
        cv2.cvtColor(diff_amp, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT
    )
    diff_r = resize(diff_color)

    # Add labels
    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    row1 = np.hstack([
        label(orig_r, "Original Hazy"),
        label(deh_r,  "DCP Dehazed"),
        label(enh_r,  "CLAHE Enhanced"),
    ])
    row2 = np.hstack([
        label(dark_r, "Dark Channel"),
        label(t_r,    "Transmission Map"),
        label(diff_r, "Enhancement Diff"),
    ])

    sep = np.zeros((4, row1.shape[1], 3), dtype=np.uint8)
    grid = np.vstack([row1, sep, row2])

    if save_path:
        cv2.imwrite(save_path, grid)

    return grid


def plot_histogram_comparison(hazy, enhanced, save_path=None):
    """
    Compare BGR channel histograms before and after.
    Returns a matplotlib figure image as numpy array.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        fig.patch.set_facecolor("#0d1117")

        channels = ["Blue", "Green", "Red"]
        colors_h = ["#4488ff", "#44ff88", "#ff4455"]
        colors_e = ["#0022cc", "#007744", "#aa0011"]

        for i, (ax, ch, c_h, c_e) in enumerate(
            zip(axes, channels, colors_h, colors_e)
        ):
            hist_h = cv2.calcHist([hazy],     [i], None, [256], [0, 256]).flatten()
            hist_e = cv2.calcHist([enhanced], [i], None, [256], [0, 256]).flatten()
            hist_h /= hist_h.sum()
            hist_e /= hist_e.sum()

            ax.plot(hist_h, color=c_h, alpha=0.7, linewidth=1.5, label="Hazy")
            ax.plot(hist_e, color=c_e, alpha=0.9, linewidth=1.5, label="Enhanced")
            ax.fill_between(range(256), hist_h, alpha=0.15, color=c_h)
            ax.fill_between(range(256), hist_e, alpha=0.15, color=c_e)
            ax.set_title(ch, color="white", fontsize=11)
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="gray")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.legend(fontsize=8, labelcolor="white",
                      framealpha=0.2, facecolor="#0d1117")

        plt.suptitle("Channel Histogram: Hazy vs Enhanced",
                     color="white", fontsize=13, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
        plt.close()

    except ImportError:
        print("matplotlib not installed — skipping histogram plot")


def generate_report(metrics_list, output_path="report.json"):
    """Aggregate metrics across a batch and save summary report."""
    if not metrics_list:
        return {}

    def mean(key): return float(np.mean([m[key] for m in metrics_list if key in m]))
    def std(key):  return float(np.std( [m[key] for m in metrics_list if key in m]))

    report = {
        "n_images": len(metrics_list),
        "smoke_density":    {"mean": round(mean("smoke_density"), 4),
                             "std":  round(std("smoke_density"), 4)},
        "visibility_gain":  {"mean": round(mean("visibility_gain_pct"), 2),
                             "std":  round(std("visibility_gain_pct"), 2)},
        "aqi":              {"mean": round(mean("aqi"), 1),
                             "std":  round(std("aqi"), 1)},
        "processing_time_s":{"mean": round(mean("processing_time_s"), 3)},
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    import sys
    from haze_pipeline import process_image

    if len(sys.argv) < 2:
        print("Usage: python evaluation.py image.jpg [gt_clean.jpg]")
        sys.exit(1)

    hazy_path = sys.argv[1]
    gt_path   = sys.argv[2] if len(sys.argv) > 2 else None

    metrics_pipe, orig, dehazed, enhanced, trans = process_image(hazy_path)
    gt = cv2.imread(gt_path) if gt_path else None

    eval_metrics = evaluate_pair(orig, enhanced, gt)
    print(json.dumps({**metrics_pipe, **eval_metrics}, indent=2))

    make_analysis_grid(orig, dehazed, enhanced, trans, save_path="analysis_grid.png")
    plot_histogram_comparison(orig, enhanced, save_path="histogram_comparison.png")
    print("Saved: analysis_grid.png, histogram_comparison.png")
