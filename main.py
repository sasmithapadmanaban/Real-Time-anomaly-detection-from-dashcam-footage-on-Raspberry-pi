#!/usr/bin/env python3
"""
main.py
Edge AI Dashcam Anomaly Detection — Entry Point
Platform : Raspberry Pi 32-bit OS, Python 3.13
Runtime  : OpenCV DNN (no TFLite required)
"""

import cv2
import time
from camera_module import CameraModule
from detector import AnomalyDetector
from anomaly_logger import AnomalyLogger


def main():
    print("=" * 58)
    print("  EDGE AI DASHCAM ANOMALY DETECTION")
    print("  Model   : MobileNet SSD via OpenCV DNN")
    print("  Camera  : USB Webcam (index 0)")
    print("  Python  : 3.13 compatible")
    print("=" * 58)

    # ── Initialise modules ────────────────────────────────────
    camera = CameraModule(
        camera_index=0, width=640, height=480, fps=15)

    detector = AnomalyDetector(
        prototxt_path="models/mobilenet_ssd.prototxt",
        caffemodel_path="models/mobilenet_ssd.caffemodel",
        confidence_threshold=0.70)

    logger = AnomalyLogger(
        log_dir="logs",
        footage_dir="footage",
        snapshot_dir="snapshots")

    camera.start()
    print("\n[INFO] System running. Controls:")
    print("         Q — quit")
    print("         S — manual snapshot\n")

    fps_time   = time.time()
    frame_tick = 0
    fps        = 0.0

    while True:
        frame = camera.read()
        if frame is None:
            continue

        frame_tick += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps        = frame_tick / (now - fps_time)
            frame_tick = 0
            fps_time   = now

        # ── Detection ─────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Buffer frame for pre-event video clip ─────────────
        logger.buffer_frame(frame)

        # ── Log each detection ────────────────────────────────
        for det in detections:
            logger.log_event(det, frame)
            print(f"[ALERT] {det['label']:12s}  "
                  f"{det['confidence']*100:.1f}%  "
                  f"frame={logger.frame_count}")

        # ── Draw HUD ──────────────────────────────────────────
        display = detector.draw_hud(frame.copy(), detections, fps)

        cv2.imshow(
            "Edge AI Dashcam Anomaly Detection | Q=Quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] Quitting...")
            break
        elif key == ord('s'):
            p = logger.save_manual_snapshot(frame)
            print(f"[INFO] Manual snapshot: {p}")

    camera.stop()
    cv2.destroyAllWindows()
    logger.print_summary()


if __name__ == "__main__":
    main()
