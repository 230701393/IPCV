"""
Live Webcam Smoke Detection
Real-time DCP + CLAHE on webcam feed with AQI overlay
Run: python webcam.py
Controls: Q = quit | S = save snapshot | D = toggle dehaze | H = toggle HUD
"""

import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from haze_pipeline import (
    dark_channel, estimate_atmospheric_light,
    estimate_transmission, recover_scene_radiance,
    clahe_enhance, adaptive_sharpen,
    compute_smoke_density, compute_fog_density_index,
    estimate_aqi_from_density
)


# ── Config ────────────────────────────────────
PATCH_SIZE    = 9       # smaller for real-time speed
OMEGA         = 0.90
GUIDED_RADIUS = 20
CLAHE_CLIP    = 2.5
ALPHA_SMOOTH  = 0.15    # EMA smoothing for density readout
SNAPSHOT_DIR  = Path("webcam_snapshots")


# ── Fast lightweight dehazer for real-time ────
def fast_dehaze(frame, patch=9, omega=0.90):
    """Optimised DCP for real-time: smaller patch, no guided filter."""
    dark_ch  = dark_channel(frame, patch)
    atm      = estimate_atmospheric_light(frame, dark_ch)
    trans    = estimate_transmission(frame, atm, omega, patch)
    trans    = cv2.GaussianBlur(trans.astype(np.float32), (11, 11), 0)  # fast smooth
    dehazed  = recover_scene_radiance(frame, trans, atm, t0=0.1)
    return dehazed, trans


def enhance_frame(dehazed, clip=2.5):
    enhanced = clahe_enhance(dehazed, clip_limit=clip)
    return adaptive_sharpen(enhanced, strength=0.3)


# ── HUD drawing helpers ───────────────────────
AQI_COLORS = {
    "Good":                   (0,   220, 0),
    "Moderate":               (0,   220, 220),
    "Unhealthy (Sensitive)":  (0,   130, 255),
    "Unhealthy":              (0,   80,  255),
    "Very Unhealthy":         (180, 60,  200),
    "Hazardous":              (30,  20,  180),
}

def draw_hud(frame, density, aqi_info, fps, mode_label, smoothed_density):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cat = aqi_info["category"]
    color = AQI_COLORS.get(cat, (200, 200, 200))

    # Top-left panel background
    cv2.rectangle(overlay, (0, 0), (320, 180), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "SMOKE / AQI MONITOR", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # AQI value (large)
    cv2.putText(frame, f"AQI  {aqi_info['aqi']}", (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2, cv2.LINE_AA)

    # Category
    cv2.putText(frame, cat, (12, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Density bar
    bar_x, bar_y, bar_w, bar_h = 12, 112, 290, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
    fill = int(bar_w * min(smoothed_density, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 120), 1)
    cv2.putText(frame, f"Smoke density  {smoothed_density:.3f}", (12, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    # FDI row
    cv2.putText(frame, f"FPS {fps:.1f}   Mode: {mode_label}", (12, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    # Timestamp bottom-right
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (w - 200, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    # Controls help (bottom-left)
    cv2.putText(frame, "Q=quit  S=snapshot  D=dehaze  H=HUD", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    return frame


def draw_transmission_mini(frame, trans):
    """Draw a small transmission heatmap thumbnail in top-right corner."""
    h, w = frame.shape[:2]
    th, tw = 100, 140
    t_uint8  = (np.clip(trans, 0, 1) * 255).astype(np.uint8)
    t_color  = cv2.applyColorMap(t_uint8, cv2.COLORMAP_JET)
    t_small  = cv2.resize(t_color, (tw, th))
    frame[10: 10+th, w-tw-10: w-10] = t_small
    cv2.rectangle(frame, (w-tw-10, 10), (w-10, 10+th), (80, 80, 80), 1)
    cv2.putText(frame, "Transmission", (w-tw-4, 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    return frame


# ── Main loop ────────────────────────────────
def run_webcam(camera_index=0):
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}.")
        print("Try: python webcam.py 1   (for external webcam)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Webcam started. Controls: Q=quit | S=snapshot | D=toggle dehaze | H=toggle HUD")

    smoothed_density = 0.3
    show_hud   = True
    do_dehaze  = True
    prev_time  = time.time()
    fps        = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed. Exiting.")
            break

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-6))
        prev_time = now

        display = frame.copy()

        if do_dehaze:
            dehazed, transmission = fast_dehaze(frame, PATCH_SIZE, OMEGA)
            enhanced = enhance_frame(dehazed, CLAHE_CLIP)

            density = compute_smoke_density(frame, transmission)
            smoothed_density = (1 - ALPHA_SMOOTH) * smoothed_density + ALPHA_SMOOTH * density

            aqi_info = estimate_aqi_from_density(smoothed_density)
            display  = enhanced
            mode_lbl = "DCP+CLAHE"

            # Mini transmission thumbnail
            display = draw_transmission_mini(display, transmission)
        else:
            # Passthrough — still estimate density quickly
            from haze_pipeline import dark_channel as dc
            dark_ch = dc(frame, PATCH_SIZE)
            atm     = estimate_atmospheric_light(frame, dark_ch)
            trans   = estimate_transmission(frame, atm, OMEGA, PATCH_SIZE)
            density = compute_smoke_density(frame, trans)
            smoothed_density = (1 - ALPHA_SMOOTH) * smoothed_density + ALPHA_SMOOTH * density
            aqi_info = estimate_aqi_from_density(smoothed_density)
            mode_lbl = "Original"

        if show_hud:
            display = draw_hud(display, smoothed_density, aqi_info, fps, mode_lbl, smoothed_density)

        cv2.imshow("Smoke & AQI Monitor — Press Q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            # Save snapshot
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SNAPSHOT_DIR / f"snapshot_{ts}.png"
            cv2.imwrite(str(path), display)
            print(f"Saved: {path}")
        elif key == ord('d'):
            do_dehaze = not do_dehaze
            print(f"Dehazing: {'ON' if do_dehaze else 'OFF'}")
        elif key == ord('h'):
            show_hud = not show_hud

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam(cam_idx)
