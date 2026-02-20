#!/usr/bin/env python3
"""
camera_module.py
Threaded USB webcam capture.
Runs in background thread so main loop never stuters."""

import cv2
import threading
import time


class CameraModule:
    def _init_(self, camera_index=0, width=640, height=480, fps=15):
        self.index   = camera_index
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.cap     = None
        self.frame   = None
        self.running = False
        self.lock    = threading.Lock()

    def start(self):
        print(f"[CAMERA] Opening USB webcam (index {self.index})...")
        self.cap = cv2.VideoCapture(self.index,cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.index}. "
                "Try: ls /dev/video* and change camera_index.")

        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Opened at {aw}x{ah}")

        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        time.sleep(0.5)   # warm up

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[CAMERA] Released.")
