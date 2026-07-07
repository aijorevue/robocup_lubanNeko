#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import http.server
import socketserver
import time
import os
import signal
import sys

signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

frame = None
info_text = "starting..."
lock = threading.Lock()
running = True
PORT = 8102

YELLOW_LOW = np.array([20, 80, 80])
YELLOW_HIGH = np.array([35, 255, 255])
MIN_AREA = 150
MIN_RADIUS = 10
CIRC_THRESH = 0.75


def circularity(c):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    if p == 0:
        return 0
    return 4 * np.pi * a / (p * p)


def detect_yellow_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        circ = circularity(c)
        if circ < CIRC_THRESH:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < MIN_RADIUS:
            continue
        conf = min(1.0, (circ - 0.65) / 0.35)
        results.append({
            'cx': int(x), 'cy': int(y), 'r': int(r),
            'circ': round(circ, 3), 'conf': round(conf, 2)
        })
    return results, mask


def camera_loop():
    global frame, info_text
    cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while running:
        ret, img = cap.read()
        if not ret:
            continue
        objs, mask = detect_yellow_ball(img)
        vis = img.copy()
        n = len(objs)
        if n == 0:
            info_text = "Yellow Ball: searching..."
            cv2.putText(vis, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            o = objs[0]
            cv2.circle(vis, (o['cx'], o['cy']), o['r'], (0, 255, 255), 2)
            cv2.circle(vis, (o['cx'], o['cy']), 4, (0, 255, 255), -1)
            label = "Yellow Ball  conf={:.2f}  circ={:.2f}".format(o['conf'], o['circ'])
            cv2.putText(vis, label, (o['cx'] + o['r'] + 5, o['cy']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            info_text = "Yellow Ball: FOUND x={} y={} r={} conf={:.2f}".format(
                o['cx'], o['cy'], o['r'], o['conf'])
            if n > 1:
                info_text += " (+{} more)".format(n - 1)
        with lock:
            frame = vis
    cap.release()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html;charset=utf-8")
            self.end_headers()
            html = ("<html><body style='margin:0;background:#111;"
                    "text-align:center;color:#fff;font-family:sans-serif'>"
                    "<h2 style='margin:10px'>Yellow Ball Detection</h2>"
                    "<img src=/video style='max-width:100%;border-radius:8px'>"
                    "<div id=info style='padding:10px;font-size:18px'>" + info_text + "</div>"
                    "<script>"
                    "setInterval(function(){"
                    "fetch('/status').then(function(r){return r.text();}).then(function(t){"
                    "document.getElementById('info').textContent=t;"
                    "});"
                    "},500)"
                    "</script>"
                    "</body></html>")
            self.wfile.write(html.encode())
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "text/plain;charset=utf-8")
            self.end_headers()
            self.wfile.write(info_text.encode())
        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Content-type",
                             "multipart/x-mixed-replace;boundary=f")
            self.end_headers()
            try:
                while running:
                    with lock:
                        f = frame.copy() if frame is not None else None
                    if f is not None:
                        _, jpg = cv2.imencode(
                            ".jpg", f,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        self.wfile.write(b"--f\r\nContent-Type:image/jpeg\r\n\r\n")
                        self.wfile.write(jpg.tobytes())
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except:
                pass

    def log_message(self, *a):
        pass


s = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
threading.Thread(target=lambda: (time.sleep(0.5), s.serve_forever()),
                 daemon=True).start()
threading.Thread(target=camera_loop, daemon=True).start()
print("http://192.168.137.207:{}   -- Yellow Ball Detection".format(PORT))
print("Press Ctrl+C to stop")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    running = False
    s.shutdown()
