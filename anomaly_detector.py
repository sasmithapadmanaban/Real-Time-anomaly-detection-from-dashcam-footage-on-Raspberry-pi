import cv2
import numpy as np
from collections import deque
from datetime import datetime
import os

class AnomalyDetector:
    """
    Multi-algorithm anomaly detector for 32-bit Raspberry Pi:
    1. Optical Flow     - Motion burst / sudden braking detection
    2. Frame Difference - Abrupt scene change detection
    3. Background Sub   - Foreign object / intrusion detection
    4. Motion History   - Sudden acceleration/deceleration pattern
    """

    def init(self, sensitivity=0.55, history_frames=30):
        self.sensitivity = sensitivity
        self.history = deque(maxlen=history_frames)
        self.motion_history = deque(maxlen=60)
        self.anomaly_log = []

        # Background subtractor — MOG2 is best for dashcam use
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50,
            detectShadows=False
        )

        # Optical flow parameters — tuned for 32-bit performance
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.feature_params = dict(
            maxCorners=80,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.prev_gray = None
        self.prev_points = None
        self.frame_count = 0
        self.anomaly_count = 0

        # Detection thresholds — tune these based on your environment
        self.thresholds = {
            'motion_burst': 20.0,
            'frame_diff': 35.0,
            'fg_density': 0.30,
            'flow_variance': 600.0
        }

        # Limit OpenCV threads for 32-bit RAM management
        cv2.setNumThreads(2)

        print("[DETECTOR] Anomaly detector initialized successfully.")

    def preprocess(self, frame):
        """Resize and convert to grayscale."""
        resized = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return resized, gray, blurred

    def detect_motion_burst(self, gray):
        """Detect sudden acceleration or braking via optical flow."""
        score = 0.0
        flow_vectors = []

        if self.prev_gray is None:
            self.prev_gray = gray
            return score, flow_vectors

        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, **self.feature_params
            )

        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )

            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]

            if len(good_new) > 0:
                flow = good_new - good_old
                magnitudes = np.sqrt(flow[:, 0]*2 + flow[:, 1]*2)
                score = float(np.mean(magnitudes))
                variance = float(np.var(magnitudes))
                flow_vectors = flow.tolist()

                # Chaotic flow = higher anomaly risk
                if variance > self.thresholds['flow_variance']:
                    score *= 1.5

        self.prev_gray = gray.copy()
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, mask=None, **self.feature_params
        )
        return score, flow_vectors

    def detect_frame_difference(self, gray):
        """Detect abrupt scene changes like collisions."""
        self.history.append(gray)
        if len(self.history) < 3:
            return 0.0
        diff = cv2.absdiff(gray, self.history[-3])
        score = float(np.mean(diff))
        return score

    def detect_foreground_intrusion(self, frame):
        """Detect unusual objects entering the dashcam scene."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        fg_pixels = np.count_nonzero(fg_mask)
        density = fg_pixels / total_pixels

        return density, fg_mask

    def analyze(self, frame):
        """
        Main analysis pipeline.
        Returns a result dictionary with all scores and flags.
        """
        self.frame_count += 1
        small_frame, gray, blurred = self.preprocess(frame)

        # Run all detectors
        motion_score, flow_vectors = self.detect_motion_burst(blurred)
        diff_score = self.detect_frame_difference(blurred)
        fg_density, fg_mask = self.detect_foreground_intrusion(small_frame)

        # Track motion over time
        self.motion_history.append(motion_score)

        # Weighted anomaly scoring
        anomaly_score = 0.0
        flags = []

        if motion_score > self.thresholds['motion_burst']:
            anomaly_score += 0.4
            flags.append(f"MOTION_BURST ({motion_score:.1f})")

        if diff_score > self.thresholds['frame_diff']:
            anomaly_score += 0.35
            flags.append(f"SCENE_CHANGE ({diff_score:.1f})")

        if fg_density > self.thresholds['fg_density']:
            anomaly_score += 0.25
            flags.append(f"INTRUSION ({fg_density:.2f})")

        # Detect sudden braking or acceleration pattern
        if len(self.motion_history) > 10:
            recent_avg = np.mean(list(self.motion_history)[-5:])
            older_avg = np.mean(list(self.motion_history)[-15:-5])
            if older_avg > 0:
                change_ratio = abs(recent_avg - older_avg) / older_avg
                if change_ratio > 0.7:
                    anomaly_score += 0.3
                    flags.append("SUDDEN_BRAKE_OR_ACCEL")

        is_anomaly = anomaly_score >= self.sensitivity

        result = {
            'frame_id': self.frame_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 3),
            'motion_score': round(motion_score, 2),
            'diff_score': round(diff_score, 2),
            'fg_density': round(fg_density, 3),
            'flags': flags,
            'fg_mask': fg_mask
        }

        if is_anomaly:
            self.anomaly_count += 1
            self.anomaly_log.append(result)
            print(f"[⚠️  ANOMALY] Frame {self.frame_count} | "
                  f"Score: {anomaly_score:.2f} | Flags: {flags}")

        return result
