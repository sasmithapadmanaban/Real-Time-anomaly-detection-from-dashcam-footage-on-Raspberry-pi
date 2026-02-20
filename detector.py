#!/usr/bin/env python3
"""
detector.py
Road anomaly detector using OpenCV DNN module.
Model  : MobileNet SSD (Caffe format)
Runtime: OpenCV DNN — works on Python 3.13, no TFLite needed
"""

import cv2
import numpy as np

ROAD_CLASSES = {
    "person", "bicycle", "car", "motorbike",
    "bus", "truck", "cat", "dog", "horse", "sheep", "cow"
}

ALL_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CLASS_COLORS = {
    "person":    (0,   0,   255),
    "bicycle":   (0,   165, 255),
    "car":       (0,   200, 0),
    "motorbike": (0,   165, 255),
    "bus":       (0,   0,   200),
    "truck":     (0,   0,   200),
    "cat":       (255, 0,   200),
    "dog":       (255, 0,   200),
    "horse":     (255, 0,   200),
    "sheep":     (255, 0,   200),
    "cow":       (255, 0,   200),
    "pothole":   (0,   0,   180),
    "obstacle":  (0,   140, 255),
}

COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0)
COLOR_RED    = (0,   0,   220)
COLOR_GREEN  = (0,   200, 0)
COLOR_YELLOW = (0,   210, 255)


def put_label(frame, text, pos, color=COLOR_WHITE,
              scale=0.55, thickness=1, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(frame, (x-3, y-h-4),
                      (x+w+3, y+baseline), COLOR_BLACK, -1)
    cv2.putText(frame, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


class AnomalyDetector:
    def _init_(self, prototxt_path, caffemodel_path,
                 confidence_threshold=0.70):
        self.threshold       = confidence_threshold
        self.prototxt_path   = prototxt_path
        self.caffemodel_path = caffemodel_path
        self.net             = None
        self.dnn_ok          = False
        self._load_model()

    def _load_model(self):
        try:
            self.net    = cv2.dnn.readNetFromCaffe(
                self.prototxt_path, self.caffemodel_path)
            self.dnn_ok = True
            print("[DETECTOR] MobileNet SSD loaded via OpenCV DNN")
            print(f"[DETECTOR] Confidence threshold: "
                  f"{self.threshold*100:.0f}%")
        except Exception as e:
            print(f"[DETECTOR] WARNING: DNN load failed: {e}")
            print("[DETECTOR] Running pothole-detection only.")

    def _enhance(self, frame):
        """CLAHE — robust detection in any lighting condition."""
        lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l       = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _run_dnn(self, frame):
        if not self.dnn_ok:
            return []
        h, w     = frame.shape[:2]
        enhanced = self._enhance(frame)
        blob     = cv2.dnn.blobFromImage(
            cv2.resize(enhanced, (300, 300)),
            0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        raw = self.net.forward()

        results = []
        for i in range(raw.shape[2]):
            conf = float(raw[0, 0, i, 2])
            if conf < self.threshold:
                continue
            idx = int(raw[0, 0, i, 1])
            if idx >= len(ALL_CLASSES):
                continue
            label = ALL_CLASSES[idx]
            if label not in ROAD_CLASSES:
                continue
            x1 = max(0, int(raw[0, 0, i, 3] * w))
            y1 = max(0, int(raw[0, 0, i, 4] * h))
            x2 = min(w, int(raw[0, 0, i, 5] * w))
            y2 = min(h, int(raw[0, 0, i, 6] * h))
            results.append({
                "label": label, "confidence": conf,
                "bbox": (x1, y1, x2, y2), "type": "obstacle"
            })
        return results

    def _detect_potholes(self, frame):
        h, w  = frame.shape[:2]
        roi   = frame[int(h * 0.60):h, :]
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (7, 7), 0)
        _, th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kern)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (2000 <= area <= 40000):
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < 0.20:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            yf   = y + int(h * 0.60)
            conf = min(0.50 + circ * 0.50, 0.95)
            if conf >= self.threshold * 0.85:
                results.append({
                    "label": "pothole", "confidence": conf,
                    "bbox": (x, yf, x+bw, yf+bh), "type": "pothole"
                })
        return results

    def detect(self, frame):
        return self._run_dnn(frame) + self._detect_potholes(frame)

    def draw_hud(self, frame, detections, fps):
        h, w  = frame.shape[:2]
        alert = len(detections) > 0
        col   = COLOR_RED if alert else COLOR_GREEN
        txt   = (f"  ANOMALY: {len(detections)} DETECTED"
                 if alert else "  ROAD CLEAR")

        cv2.rectangle(frame, (0, 0), (w, 36), COLOR_BLACK, -1)
        cv2.rectangle(frame, (0, 0), (w, 36), col, 2)
        put_label(frame, txt, (8, 25), col, 0.65, 2, False)
        put_label(frame, f"FPS:{fps:.1f}", (w-90, 25),
                  COLOR_YELLOW, 0.50, 1, False)

        road_y = int(h * 0.60)
        cv2.line(frame, (0, road_y), (w, road_y), (70, 70, 70), 1)
        put_label(frame, "ROAD ZONE", (5, road_y-5),
                  (100, 100, 100), 0.38, 1, False)

        for det in detections:
            lbl = det["label"]
            cf  = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]
            c   = CLASS_COLORS.get(lbl, COLOR_YELLOW)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            tag = f"{lbl.upper()} {cf*100:.0f}%"
            (tw, th2), _ = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame,
                          (x1, y1-th2-10), (x1+tw+6, y1), c, -1)
            cv2.putText(frame, tag, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR_WHITE, 1, cv2.LINE_AA)
            cl = 14
            cv2.line(frame,(x1,y1),(x1+cl,y1),COLOR_WHITE,2)
            cv2.line(frame,(x1,y1),(x1,y1+cl),COLOR_WHITE,2)
            cv2.line(frame,(x2,y2),(x2-cl,y2),COLOR_WHITE,2)
            cv2.line(frame,(x2,y2),(x2,y2-cl),COLOR_WHITE,2)

        if alert:
            cv2.rectangle(frame, (0,0), (w-1,h-1), COLOR_RED, 4)

        put_label(frame,
                  "OpenCV DNN | MobileNet SSD | 70% Threshold | CLAHE",
                  (5, h-8), (130,130,130), 0.38, 1, False)
        return frame
