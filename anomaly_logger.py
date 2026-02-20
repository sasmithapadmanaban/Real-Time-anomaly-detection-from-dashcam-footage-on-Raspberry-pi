import cv2
import csv
import os
import time
from datetime import datetime
from collections import deque

PRE_SEC    = 5
POST_SEC   = 5
FPS        = 15
FOURCC     = cv2.VideoWriter_fourcc(*"mp4v")
FRAME_SIZE = (640, 480)


class AnomalyLogger:
    def _init_(self, log_dir="logs", footage_dir="footage",
                 snapshot_dir="snapshots"):
        self.log_dir      = log_dir
        self.footage_dir  = footage_dir
        self.snapshot_dir = snapshot_dir

        for d in [log_dir, footage_dir, snapshot_dir]:
            os.makedirs(d, exist_ok=True)

        self.pre_buffer       = deque(maxlen=PRE_SEC * FPS)
        self.post_writer      = None
        self.post_frames_left = 0
        self.clip_path        = None
        self.frame_count      = 0
        self.total_detections = 0
        self.session_start    = time.time()

        self.csv_path = os.path.join(log_dir, "anomaly_log.csv")
        self._init_csv()

        print(f"[LOGGER] CSV log    : {self.csv_path}")
        print(f"[LOGGER] Snapshots  : {snapshot_dir}/")
        print(f"[LOGGER] Clips      : {footage_dir}/")

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "Timestamp", "Frame", "Type", "Label",
                    "Confidence_%", "x1","y1","x2","y2",
                    "Snapshot", "VideoClip"
                ])

    def buffer_frame(self, frame):
        """Call every frame — maintains pre-event buffer."""
        self.frame_count += 1
        self.pre_buffer.append(frame.copy())

        if self.post_writer and self.post_frames_left > 0:
            self.post_writer.write(frame)
            self.post_frames_left -= 1
            if self.post_frames_left == 0:
                self.post_writer.release()
                self.post_writer = None
                print(f"[LOGGER] Clip saved: {self.clip_path}")

    def log_event(self, detection, frame):
        """Save snapshot + start video clip + write CSV row."""
        self.total_detections += 1
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ts_hr = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Snapshot
        snap_name = f"anomaly_{ts}.jpg"
        snap_path = os.path.join(self.snapshot_dir, snap_name)
        cv2.imwrite(snap_path, frame)

        # Video clip
        clip_name = f"clip_{ts}.mp4"
        clip_path = os.path.join(self.footage_dir, clip_name)

        if self.post_writer is None:
            writer = cv2.VideoWriter(
                clip_path, FOURCC, FPS, FRAME_SIZE)
            for f in self.pre_buffer:
                writer.write(f)
            self.post_writer      = writer
            self.post_frames_left = POST_SEC * FPS
            self.clip_path        = clip_path
        else:
            # Extend existing clip
            self.post_frames_left = POST_SEC * FPS
            clip_name = os.path.basename(self.clip_path)
            clip_path = self.clip_path

        # CSV
        x1, y1, x2, y2 = detection["bbox"]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts_hr, self.frame_count,
                detection.get("type", "unknown"),
                detection["label"],
                f"{detection['confidence']*100:.1f}",
                x1, y1, x2, y2, snap_name, clip_name
            ])

    def save_manual_snapshot(self, frame):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.snapshot_dir, f"manual_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def print_summary(self):
        dur = time.time() - self.session_start
        print("\n" + "="*52)
        print("  SESSION SUMMARY")
        print("="*52)
        print(f"  Frames processed  : {self.frame_count}")
        print(f"  Total detections  : {self.total_detections}")
        print(f"  Duration          : {dur:.1f}s")
        print(f"  Avg FPS           : {self.frame_count/max(dur,1):.1f}")
        print(f"  CSV log           : {self.csv_path}")
        print(f"  Snapshots         : {self.snapshot_dir}/")
        print(f"  Video clips       : {self.footage_dir}/")
        print("="*52)
