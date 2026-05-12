"""
Streamlit Web UI — Smoke Density Estimation & Air Quality Monitor
Run: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import time
from pathlib import Path

from haze_pipeline import (
    dcp_dehaze, clahe_enhance, adaptive_sharpen,
    compute_smoke_density, compute_visibility_score,
    compute_contrast_ratio, compute_fog_density_index,
    estimate_aqi_from_density
)

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Smoke & AQI Monitor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #2a2a3e;
    }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 600; }
    .aqi-good        { color: #00e400; }
    .aqi-moderate    { color: #ffff00; }
    .aqi-sensitive   { color: #ff7e00; }
    .aqi-unhealthy   { color: #ff4444; }
    .aqi-very        { color: #cc44cc; }
    .aqi-hazardous   { color: #cc0033; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def transmission_to_heatmap(trans):
    t_uint8 = (np.clip(trans, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(t_uint8, cv2.COLORMAP_JET)
    return bgr_to_pil(heatmap)

def dark_channel_to_pil(dark):
    d_uint8 = (np.clip(dark / (dark.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)
    return bgr_to_pil(colored)

def parse_beta_from_filename(filename):
    """Extract ground truth beta (haze density) from RESIDE filename like 1324_1_0.82218.png"""
    try:
        parts = Path(filename).stem.split("_")
        return float(parts[-1])
    except:
        return None

def aqi_color_class(category):
    mapping = {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy (Sensitive)": "aqi-sensitive",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very",
        "Hazardous": "aqi-hazardous",
    }
    return mapping.get(category, "aqi-good")

def aqi_alert_type(category):
    if category == "Good":             return "success"
    if category == "Moderate":         return "warning"
    return "error"

def run_pipeline(image_bgr, clip_limit, sharpen_strength, patch_size, omega):
    t0 = time.time()
    dehazed, transmission, atm_light = dcp_dehaze(
        image_bgr, patch_size=patch_size, omega=omega, guided=True
    )
    enhanced = clahe_enhance(dehazed, clip_limit=clip_limit)
    enhanced = adaptive_sharpen(enhanced, strength=sharpen_strength)
    elapsed = time.time() - t0

    smoke_density   = compute_smoke_density(image_bgr, transmission)
    vis_orig        = compute_visibility_score(image_bgr)
    vis_enh         = compute_visibility_score(enhanced)
    contrast_orig   = compute_contrast_ratio(image_bgr)
    contrast_enh    = compute_contrast_ratio(enhanced)
    fdi             = compute_fog_density_index(image_bgr)
    aqi_info        = estimate_aqi_from_density(smoke_density)
    vis_gain        = (vis_enh - vis_orig) / (vis_orig + 1e-6) * 100

    from haze_pipeline import dark_channel as dc
    dark_ch = dc(image_bgr, patch_size)

    return {
        "dehazed": dehazed,
        "enhanced": enhanced,
        "transmission": transmission,
        "dark_channel": dark_ch,
        "atm_light": atm_light,
        "smoke_density": smoke_density,
        "fdi": fdi,
        "vis_orig": vis_orig,
        "vis_enh": vis_enh,
        "vis_gain": vis_gain,
        "contrast_orig": contrast_orig,
        "contrast_enh": contrast_enh,
        "aqi": aqi_info["aqi"],
        "aqi_category": aqi_info["category"],
        "aqi_color": aqi_info["color"],
        "elapsed": elapsed,
    }


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.title("🌫️ Smoke & AQI Monitor")
    st.caption("DCP Haze Removal + CLAHE | RESIDE Indoor Dataset")
    st.divider()

    st.subheader("⚙️ Pipeline Settings")
    patch_size = st.slider("DCP patch size", 5, 30, 15, step=2,
                           help="Larger = smoother dark channel, slower")
    omega = st.slider("Haze removal strength (ω)", 0.5, 1.0, 0.95, step=0.05,
                      help="0.95 is standard; lower retains more haze for sky regions")
    clip_limit = st.slider("CLAHE clip limit", 1.0, 8.0, 3.0, step=0.5,
                           help="Higher = more contrast enhancement, may introduce noise")
    sharpen = st.slider("Sharpening strength", 0.0, 1.0, 0.4, step=0.1)

    st.divider()
    st.subheader("📁 Mode")
    mode = st.radio("", ["Upload image", "Batch folder"], label_visibility="collapsed")

    st.divider()
    st.caption("Dataset: RESIDE Indoor Haze (ITS) — 1399 images")
    st.caption("Pipeline: DCP → Guided Filter → CLAHE → Unsharp Mask")


# ── Main ──────────────────────────────────────
st.title("Smoke Density Estimation & Air Quality Monitoring")

# ════════════════════════════════════════════
# MODE 1: Single Image Upload
# ════════════════════════════════════════════
if mode == "Upload image":

    uploaded = st.file_uploader(
        "Upload a hazy image (RESIDE .png or any hazy photo)",
        type=["png", "jpg", "jpeg", "bmp"],
        help="RESIDE filenames like 1324_1_0.82218.png will auto-extract ground truth beta"
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not decode image. Try a different file.")
            st.stop()

        # Check for RESIDE ground truth beta in filename
        gt_beta = parse_beta_from_filename(uploaded.name)
        if gt_beta is not None:
            st.info(f"📌 RESIDE image detected — ground truth β (haze density) = **{gt_beta:.5f}**")

        with st.spinner("Running DCP + CLAHE pipeline..."):
            r = run_pipeline(image_bgr, clip_limit, sharpen, patch_size, omega)

        # ── AQI Alert Banner
        cat = r["aqi_category"]
        alert_msg = (
            f"**AQI {r['aqi']} — {cat}** | "
            f"Smoke Density: {r['smoke_density']:.3f} | "
            f"Processing time: {r['elapsed']:.2f}s"
        )
        if cat == "Good":
            st.success(alert_msg)
        elif cat == "Moderate":
            st.warning(alert_msg)
        else:
            st.error(alert_msg)

        # ── Metrics Row
        cols = st.columns(5)
        metrics_data = [
            ("Smoke Density",    f"{r['smoke_density']:.3f}", cat),
            ("AQI",              str(r["aqi"]),               cat),
            ("Visibility Gain",  f"+{r['vis_gain']:.1f}%",   "Good"),
            ("Contrast (orig)",  f"{r['contrast_orig']:.3f}", "Good"),
            ("Fog Density Idx",  f"{r['fdi']:.3f}",           cat),
        ]
        for col, (label, value, category) in zip(cols, metrics_data):
            css_cls = aqi_color_class(category)
            col.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value {css_cls}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Ground truth comparison
        if gt_beta is not None:
            error = abs(r["smoke_density"] - gt_beta)
            st.metric(
                label="DCP Estimation Error vs Ground Truth β",
                value=f"{error:.4f}",
                delta=f"Predicted {r['smoke_density']:.4f} vs GT {gt_beta:.4f}",
                delta_color="inverse"
            )

        st.divider()

        # ── Image Grid
        st.subheader("Visual Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(bgr_to_pil(image_bgr), caption="Original Hazy", use_container_width=True)
        with c2:
            st.image(bgr_to_pil(r["dehazed"]), caption="DCP Dehazed", use_container_width=True)
        with c3:
            st.image(bgr_to_pil(r["enhanced"]), caption="CLAHE Enhanced", use_container_width=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            st.image(dark_channel_to_pil(r["dark_channel"]),
                     caption="Dark Channel (Inferno)", use_container_width=True)
        with c5:
            st.image(transmission_to_heatmap(r["transmission"]),
                     caption="Transmission Map (Jet)", use_container_width=True)

        # Difference map
        diff = cv2.absdiff(image_bgr, r["enhanced"])
        diff_amp = cv2.convertScaleAbs(diff, alpha=3)
        diff_color = cv2.applyColorMap(
            cv2.cvtColor(diff_amp, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT
        )
        with c6:
            st.image(bgr_to_pil(diff_color), caption="Enhancement Diff (Hot)", use_container_width=True)

        st.divider()

        # ── Histogram
        st.subheader("Channel Histograms — Original vs Enhanced")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        fig.patch.set_facecolor("#0d1117")
        ch_names = ["Blue", "Green", "Red"]
        colors_h = ["#5599ff", "#55ff99", "#ff5566"]
        colors_e = ["#0033bb", "#007744", "#bb0022"]

        for i, (ax, name, c_h, c_e) in enumerate(zip(axes, ch_names, colors_h, colors_e)):
            h1 = cv2.calcHist([image_bgr],     [i], None, [256], [0, 256]).flatten()
            h2 = cv2.calcHist([r["enhanced"]], [i], None, [256], [0, 256]).flatten()
            h1 /= h1.sum(); h2 /= h2.sum()
            ax.plot(h1, color=c_h, alpha=0.7, lw=1.5, label="Hazy")
            ax.plot(h2, color=c_e, alpha=0.9, lw=1.5, label="Enhanced")
            ax.fill_between(range(256), h1, alpha=0.15, color=c_h)
            ax.fill_between(range(256), h2, alpha=0.15, color=c_e)
            ax.set_title(name, color="white"); ax.set_facecolor("#161b22")
            ax.tick_params(colors="gray")
            for s in ax.spines.values(): s.set_edgecolor("#30363d")
            ax.legend(fontsize=8, labelcolor="white", framealpha=0.2, facecolor="#0d1117")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ── Download
        st.subheader("Download Results")
        dc1, dc2, dc3 = st.columns(3)

        def img_to_bytes(bgr_img):
            _, buf = cv2.imencode(".png", bgr_img)
            return buf.tobytes()

        dc1.download_button("⬇ Download Dehazed",  img_to_bytes(r["dehazed"]),
                            file_name="dehazed.png",  mime="image/png")
        dc2.download_button("⬇ Download Enhanced", img_to_bytes(r["enhanced"]),
                            file_name="enhanced.png", mime="image/png")

        metrics_json = {
            "filename":         uploaded.name,
            "smoke_density":    round(r["smoke_density"], 4),
            "gt_beta":          gt_beta,
            "fdi":              round(r["fdi"], 4),
            "visibility_gain":  round(r["vis_gain"], 2),
            "contrast_orig":    round(r["contrast_orig"], 4),
            "contrast_enh":     round(r["contrast_enh"], 4),
            "aqi":              r["aqi"],
            "aqi_category":     r["aqi_category"],
            "atm_light":        r["atm_light"].tolist(),
            "processing_time_s":round(r["elapsed"], 3),
        }
        dc3.download_button("⬇ Download Metrics JSON",
                            json.dumps(metrics_json, indent=2),
                            file_name="metrics.json", mime="application/json")

    else:
        st.info("👆 Upload a hazy image to begin. Use images from `input_images/hazy/` folder.")


# ════════════════════════════════════════════
# MODE 2: Batch Folder
# ════════════════════════════════════════════
else:
    st.subheader("Batch Processing — RESIDE Indoor Dataset")

    folder_path = st.text_input(
        "Hazy images folder path",
        value=r"input_images\hazy",
        placeholder=r"C:\Users\HP\Downloads\smoke_aqi_project\input_images\hazy"
    )
    max_imgs = st.slider("Max images to process", 5, 200, 20)

    if st.button("▶ Run Batch", type="primary"):
        folder = Path(folder_path)
        if not folder.exists():
            st.error(f"Folder not found: {folder_path}")
            st.stop()

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [p for p in folder.iterdir() if p.suffix.lower() in exts][:max_imgs]

        if not files:
            st.error("No images found in that folder.")
            st.stop()

        st.info(f"Processing {len(files)} images...")
        progress = st.progress(0)
        status   = st.empty()

        all_metrics = []
        for i, fp in enumerate(files):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            r = run_pipeline(img, clip_limit, sharpen, patch_size, omega)
            gt_beta = parse_beta_from_filename(fp.name)
            all_metrics.append({
                "filename":      fp.name,
                "smoke_density": round(r["smoke_density"], 4),
                "gt_beta":       gt_beta,
                "vis_gain":      round(r["vis_gain"], 1),
                "aqi":           r["aqi"],
                "aqi_category":  r["aqi_category"],
                "elapsed":       round(r["elapsed"], 3),
            })
            progress.progress((i + 1) / len(files))
            status.text(f"[{i+1}/{len(files)}] {fp.name} — AQI {r['aqi']} ({r['aqi_category']})")

        st.success(f"Done! Processed {len(all_metrics)} images.")

        # Summary stats
        densities = [m["smoke_density"] for m in all_metrics]
        gains     = [m["vis_gain"]      for m in all_metrics]
        aqis      = [m["aqi"]           for m in all_metrics]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Smoke Density", f"{np.mean(densities):.3f}")
        s2.metric("Avg AQI",           f"{np.mean(aqis):.0f}")
        s3.metric("Avg Visibility Gain",f"+{np.mean(gains):.1f}%")
        s4.metric("Images Processed",  len(all_metrics))

        # AQI distribution chart
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cats = ["Good", "Moderate", "Unhealthy (Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]
        colors_bar = ["#00e400", "#ffff00", "#ff7e00", "#ff4444", "#cc44cc", "#cc0033"]
        counts = [sum(1 for m in all_metrics if m["aqi_category"] == c) for c in cats]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
        bars = ax.bar(cats, counts, color=colors_bar, edgecolor="none", width=0.6)
        ax.bar_label(bars, fontsize=10, color="white", padding=3)
        ax.set_title("AQI Category Distribution", color="white", fontsize=12)
        ax.tick_params(colors="gray", labelsize=9)
        ax.set_ylim(0, max(counts) * 1.25 if counts else 10)
        for s in ax.spines.values(): s.set_edgecolor("#30363d")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Smoke density vs GT beta scatter (if RESIDE)
        gt_pairs = [(m["gt_beta"], m["smoke_density"]) for m in all_metrics if m["gt_beta"] is not None]
        if gt_pairs:
            betas, preds = zip(*gt_pairs)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor("#0d1117"); ax2.set_facecolor("#161b22")
            ax2.scatter(betas, preds, c="#5599ff", alpha=0.7, s=30, edgecolors="none")
            mn, mx = min(min(betas), min(preds)), max(max(betas), max(preds))
            ax2.plot([mn, mx], [mn, mx], "r--", lw=1, alpha=0.5, label="Perfect prediction")
            ax2.set_xlabel("Ground truth β", color="gray")
            ax2.set_ylabel("DCP predicted density", color="gray")
            ax2.set_title("DCP Estimation vs Ground Truth", color="white")
            ax2.tick_params(colors="gray")
            for s in ax2.spines.values(): s.set_edgecolor("#30363d")
            ax2.legend(labelcolor="white", facecolor="#0d1117", framealpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Results table
        st.subheader("Results Table")
        st.dataframe(
            all_metrics,
            use_container_width=True,
            column_config={
                "aqi": st.column_config.NumberColumn("AQI", format="%d"),
                "smoke_density": st.column_config.NumberColumn("Density", format="%.4f"),
                "vis_gain": st.column_config.NumberColumn("Vis Gain %", format="+%.1f"),
            }
        )

        # Download
        st.download_button(
            "⬇ Download Full Results JSON",
            json.dumps(all_metrics, indent=2),
            file_name="batch_results.json",
            mime="application/json"
        )
