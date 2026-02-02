import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque

# ==============================
# CONFIG (EDGE / DASHCAM STYLE)
# ==============================
MOTION_HISTORY = 40
FLOW_THRESHOLD = 2.0        # optical flow anomaly
MOTION_THRESHOLD = 1.0      # pixel motion
CONFIRM_FRAMES = 3
MIN_BOX_AREA = 0.02
MAX_BOX_AREA = 0.6

# ==============================
# LOAD TFLITE MODEL (CNN)
# ==============================
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_H = input_details[0]['shape'][1]
MODEL_W = input_details[0]['shape'][2]

print("Model loaded")
print("Output shape:", output_details[0]['shape'])

# ==============================
# WEBCAM (DASHCAM SIM)
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ==============================
# TEMPORAL STATE (LSTM-LIKE)
# ==============================
prev_gray = None
motion_buffer = deque(maxlen=MOTION_HISTORY)
flow_buffer = deque(maxlen=MOTION_HISTORY)
anomaly_counter = 0

# ==============================
# MAIN LOOP
# ==============================
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------
    # OPTICAL FLOW (Motion Field)
    # --------------------------
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mean = np.mean(mag)
    else:
        flow_mean = 0.0

    prev_gray = gray
    flow_buffer.append(flow_mean)

    # --------------------------
    # BASIC MOTION (PIXEL DIFF)
    # --------------------------
    if len(motion_buffer) > 0:
        motion = abs(flow_mean - np.mean(flow_buffer))
    else:
        motion = 0.0

    motion_buffer.append(motion)

    # --------------------------
    # AI INFERENCE (CNN)
    # --------------------------
    resized = cv2.resize(frame, (MODEL_W, MODEL_H))
    input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    detections = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes = []

    for det in detections:
        xmin, ymin, xmax, ymax = det

        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        if x2 <= x1 or y2 <= y1:
            continue

        area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
        if area_ratio < MIN_BOX_AREA or area_ratio > MAX_BOX_AREA:
            continue

        boxes.append((x1, y1, x2, y2))

    # --------------------------
    # ANOMALY DECISION (HYBRID)
    # --------------------------
    anomaly = False

    if (
        flow_mean > FLOW_THRESHOLD and
        motion > MOTION_THRESHOLD and
        len(boxes) > 0
    ):
        anomaly_counter += 1
    else:
        anomaly_counter = max(0, anomaly_counter - 1)

    if anomaly_counter >= CONFIRM_FRAMES:
        anomaly = True

    # --------------------------
    # VISUALIZATION
    # --------------------------
    for (x1, y1, x2, y2) in boxes:
        color = (0, 0, 255) if anomaly else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if anomaly:
        cv2.putText(frame, "DASHCAM ANOMALY DETECTED",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3)

    fps = 1.0 / (time.time() - start)

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Optical Flow: {flow_mean:.2f}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Real-Time Dashcam Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()
