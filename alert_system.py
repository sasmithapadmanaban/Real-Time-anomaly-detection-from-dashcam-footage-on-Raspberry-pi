import cv2
import os
import json
from datetime import datetime
from collections import deque

class AlertSystem:
    """
    Saves video clips automatically when anomaly is detected.
    Includes 5 seconds BEFORE the event (pre-buffer) and
    5 seconds AFTER the event (post-buffer).
    """

    def init(self, output_dir='alerts', pre_buffer_seconds=5, fps=15):
        self.output_dir = output_dir
        self.fps = fps
        self.pre_buffer_size = pre_buffer_seconds * fps
        self.pre_buffer = deque(maxlen=self.pre_buffer_size)
        self.post_buffer_size = 5 * fps

        self.is_recording = False
        self.post_frames_remaining = 0
        self.current_writer = None
        self.current_clip_path = None
        self.alert_count = 0
        self.alert_log_path = os.path.join(output_dir, 'alert_log.json')
        self.alerts = []

        os.makedirs(output_dir, exist_ok=True)
        print(f"[ALERT] Alert system ready. Saving clips to: {output_dir}/")

    def add_to_buffer(self, frame):
        """Continuously buffer recent frames."""
        self.pre_buffer.append(frame.copy())

    def trigger_alert(self, frame, anomaly_result):
        """Start recording a clip when anomaly is detected."""
        if not self.is_recording:
            self.alert_count += 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            clip_name = f"anomaly_{self.alert_count:04d}_{timestamp}.avi"
            self.current_clip_path = os.path.join(self.output_dir, clip_name)

            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.current_writer = cv2.VideoWriter(
                self.current_clip_path, fourcc, self.fps, (w, h)
            )

            # Write pre-event buffer (what happened BEFORE anomaly)
            for buffered_frame in self.pre_buffer:
                self.current_writer.write(buffered_frame)

            self.is_recording = True
            self.post_frames_remaining = self.post_buffer_size

            # Save alert metadata
            alert_entry = {
                'alert_id': self.alert_count,
                'timestamp': anomaly_result['timestamp'],
                'score': anomaly_result['anomaly_score'],
                'flags': anomaly_result['flags'],
                'clip_file': clip_name
            }
            self.alerts.append(alert_entry)
            self._save_log()

            print(f"[ALERT ⚠️ ] Alert #{self.alert_count} triggered!")
            print(f"[SAVING]    {self.current_clip_path}")

    def process_frame(self, frame, anomaly_result):
        """Call this every frame — handles buffering and recording."""
        self.add_to_buffer(frame)

        if anomaly_result['is_anomaly']:
            self.trigger_alert(frame, anomaly_result)

        if self.is_recording:
            self.current_writer.write(frame)
            self.post_frames_remaining -= 1

            if self.post_frames_remaining <= 0:
                self.current_writer.release()
                self.current_writer = None
                self.is_recording = False
                print(f"[SAVED] ✓ Clip saved successfully.")

    def _save_log(self):
        """Save alert log to JSON file."""
        with open(self.alert_log_path, 'w') as f:
            json.dump(self.alerts, f, indent=2)

    def cleanup(self):
        """Release resources on shutdown."""
        if self.current_writer:
            self.current_writer.release()
        self._save_log()
        print(f"[ALERT] Shutdown complete. Total alerts: {self.alert_count}")
