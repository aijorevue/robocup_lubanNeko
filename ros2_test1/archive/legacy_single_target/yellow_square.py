#!/usr/bin/env python3
import sys, os, cv2, time, threading, http.server, socketserver
sys.path.insert(0, "/home/cat/ros2_ws/color")
from color_detector import ColorDetector

det = ColorDetector()
frame, info, lock = None, "", threading.Lock()
running = True
PORT = 8102
COLOR = "yellow"
SHAPE = "square"

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(("<html><body style=margin:0;background:#000;text-align:center>"
            "<img src=/video style=max-width:100%>"
            "<div style=color:#fff;font-size:20px;padding:10px>" + info + "</div></body></html>").encode())
        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace;boundary=f")
            self.end_headers()
            try:
                while running:
                    with lock: f = frame.copy() if frame is not None else None
                    if f is not None:
                        _, jpg = cv2.imencode(".jpg", f, [90])
                        self.wfile.write(b"--f\r\nContent-Type:image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    time.sleep(0.04)
            except: pass
    def log_message(self, *a): pass

def loop():
    global frame, info
    cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while running:
        ret, img = cap.read()
        if not ret: continue
        objs = det.detect(img, shape=SHAPE, color=COLOR)
        vis = det.draw(img, objs)
        n = len(objs)
        info = COLOR + " " + SHAPE + ": " + ("FOUND" if n else "searching...")
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        with lock: frame = vis
    cap.release()

s = http.server.HTTPServer(("0.0.0.0", PORT), H)
threading.Thread(target=s.serve_forever, daemon=True).start()
threading.Thread(target=loop, daemon=True).start()
print("http://192.168.137.207:" + str(PORT) + "  -- " + COLOR + " " + SHAPE)
try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    running = False
    s.shutdown()
