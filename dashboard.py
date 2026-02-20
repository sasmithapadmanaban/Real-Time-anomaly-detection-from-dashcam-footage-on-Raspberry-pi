from flask import Flask, Response, render_template_string, jsonify
import cv2
import json
import os
import threading

app = Flask(name)

# Shared state between main app and dashboard
latest_frame = None
latest_stats = {
    'score': 0,
    'status': 'STARTING',
    'frames': 0,
    'flags': [],
    'alerts': 0
}
frame_lock = threading.Lock()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Dashcam Anomaly Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0d0d;
      color: #eee;
      font-family: 'Courier New', monospace;
      text-align: center;
      padding: 20px;
    }
    h1 { color: #00ff88; margin-bottom: 15px; font-size: 22px; }
    img {
      border: 3px solid #333;
      border-radius: 10px;
      max-width: 100%;
      width: 640px;
    }
    .stats-row {
      display: flex;
      justify-content: center;
      gap: 15px;
      margin: 15px auto;
      flex-wrap: wrap;
      max-width: 700px;
    }
    .card {
      background: #1a1a1a;
      border: 1px solid #333;
      padding: 12px 20px;
      border-radius: 10px;
      min-width: 130px;
      font-size: 13px;
    }
    .card .val {
      font-size: 20px;
      font-weight: bold;
      margin-top: 5px;
    }
    .normal  { color: #00cc44; }
    .warning { color: #ff9900; }
    .anomaly { color: #ff2222; }
    .flags-box {
      background: #1a1a1a;
      border-radius: 8px;
      padding: 10px 20px;
      margin: 10px auto;
      max-width: 640px;
      font-size: 13px;
      text-align: left;
    }
    #flag-list { color: #ff4444; margin-top: 5px; }
    .footer {
      margin-top: 20px;
      font-size: 11px;
      color: #555;
    }
  </style>
  <script>
    function updateStats() {
      fetch('/stats')
        .then(r => r.json())
        .then(d => {
          document.getElementById('score').innerText  = d.score || '--';
          document.getElementById('frames').innerText = d.frames || 0;
          document.getElementById('alerts').innerText = d.alerts || 0;
          document.getElementById('motion').innerText = d.motion || '--';

          const statusEl = document.getElementById('status');
          statusEl.innerText = d.status || '--';
          statusEl.className = 'val ' + (d.status === 'ANOMALY' ? 'anomaly' :
                                          d.status === 'WARNING' ? 'warning' : 'normal');

          const flags = d.flags || [];
          document.getElementById('flag-list').innerText =
            flags.length > 0 ? flags.join(' | ') : 'None';
        })
        .catch(() => {});
    }
    setInterval(updateStats, 500);
    updateStats();
  </script>
</head>
<body>
  <h1>🚗 Dashcam Anomaly Detection — Live Monitor</h1>
  <img src="/feed" alt="Live Feed" />

  <div class="stats-row">
    <div class="card">
      STATUS<br>
      <span id="status" class="val normal">--</span>
    </div>
    <div class="card">
      ANOMALY SCORE<br>
      <span id="score" class="val">--</span>
    </div>
    <div class="card">
      TOTAL ALERTS<br>
      <span id="alerts" class="val">0</span>
    </div>
    <div class="card">
      FRAMES<br>
      <span id="frames" class="val">0</span>
    </div>
    <div class="card">
      MOTION<br>
      <span id="motion" class="val">--</span>
    </div>
  </div>

  <div class="flags-box">
    <b>Active Flags:</b>
    <div id="flag-list">None</div>
  </div>

  <div class="footer">
    Raspberry Pi Real-Time Anomaly Detection System &nbsp;|&nbsp;
    Refresh: 500ms &nbsp;|&nbsp;
    Open on any device connected to the same WiFi
  </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is not None:
                ret, buffer = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 65]
                )
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n'
                           + buffer.tobytes()
                           + b'\r\n')
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def get_stats():
    return jsonify(latest_stats)

def update_feed(frame, result, alert_count=0):
    """Call this from main.py to push new frame and stats to dashboard."""
    global latest_frame, latest_stats
    with frame_lock:
        latest_frame = frame.copy()
        latest_stats = {
            'score':  result['anomaly_score'],
            'status': 'ANOMALY' if result['is_anomaly'] else
                      ('WARNING' if result['anomaly_score'] >= 0.35 else 'NORMAL'),
            'frames': result['frame_id'],
            'flags':  result['flags'],
            'motion': result['motion_score'],
            'alerts': alert_count
        }

def start_dashboard(host='0.0.0.0', port=5000):
    """Start the Flask dashboard in a background thread."""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Silence Flask request logs
    app.run(host=host, port=port, threaded=True)
