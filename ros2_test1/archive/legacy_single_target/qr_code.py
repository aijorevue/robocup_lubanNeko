#!/usr/bin/env python3
"""QR detect via OpenCV detect() + match red/blue templates"""
import cv2, numpy as np, threading, http.server, time, sys, signal

signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

RED_TMPL = cv2.imread("red.png", 0)
BLUE_TMPL = cv2.imread("blue.png", 0)
if RED_TMPL is None or BLUE_TMPL is None:
    print("ERROR: red.png or blue.png not found!")
    sys.exit(1)

frame = None; info_text = "starting..."; lock = threading.Lock(); running = True
PORT = 8120
qr_det = cv2.QRCodeDetector()

def camera_loop():
    global frame, info_text
    cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while running:
        ret, img = cap.read()
        if not ret: continue
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        retval, pts = qr_det.detect(gray)
        found = None
        scores = {}

        if pts is not None:
            for ip in pts:
                ip_int = ip.astype(int)
                x, y, w, h = cv2.boundingRect(ip_int)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
                if x2 <= x1 or y2 <= y1: continue
                roi = gray[y1:y2, x1:x2]

                scores = {}
                for label, tmpl in [("red", RED_TMPL), ("blue", BLUE_TMPL)]:
                    scaled = cv2.resize(tmpl, (roi.shape[1], roi.shape[0]))
                    res = cv2.matchTemplate(roi, scaled, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(res)
                    scores[label] = mv

                best = max(scores, key=scores.get) if scores else None
                if best and scores[best] > 0.35:
                    found = best
                    best_conf = scores[best]

            if found:
                color = (0, 0, 255) if found == "red" else (255, 0, 0)
                cv2.polylines(vis, [ip_int], True, color, 2)
                cx, cy = int(ip_int[:, 0].mean()), int(ip_int[:, 1].mean())
                cv2.putText(vis, found, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                info_text = "QR: {}  R:{:.0f}% B:{:.0f}%".format(
                    found, scores.get("red", 0) * 100, scores.get("blue", 0) * 100)
            else:
                sr, sb = scores.get("red", 0), scores.get("blue", 0)
                info_text = "QR detected  R:{:.0f}% B:{:.0f}%".format(sr * 100, sb * 100)
        else:
            info_text = "QR: searching..."

        cv2.putText(vis, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        with lock: frame = vis
    cap.release()


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-type","text/html;charset=utf-8"); self.end_headers()
            html = ("<html><body style='margin:0;background:#111;text-align:center;color:#fff;font-family:sans-serif'>"
                    "<h2>QR: red / blue</h2>"
                    "<img src=/video style='max-width:100%'>"
                    "<div id=info style='padding:10px;font-size:18px'>" + info_text + "</div>"
                    "<script>setInterval(function(){fetch('/status').then(function(r){return r.text()}).then(function(t){document.getElementById('info').textContent=t})},500)</script>"
                    "</body></html>")
            self.wfile.write(html.encode())
        elif self.path == "/status":
            self.send_response(200); self.send_header("Content-type","text/plain;charset=utf-8"); self.end_headers()
            self.wfile.write(info_text.encode())
        elif self.path == "/video":
            self.send_response(200); self.send_header("Content-type","multipart/x-mixed-replace;boundary=f"); self.end_headers()
            try:
                while running:
                    with lock: f=frame.copy() if frame is not None else None
                    if f is not None:
                        _,jpg=cv2.imencode(".jpg",f,[90])
                        self.wfile.write(b"--f\r\nContent-Type:image/jpeg\r\n\r\n"+jpg.tobytes()+b"\r\n")
                    time.sleep(0.03)
            except: pass
    def log_message(self,*a): pass

s = http.server.HTTPServer(("0.0.0.0", PORT), H)
threading.Thread(target=s.serve_forever, daemon=True).start()
threading.Thread(target=camera_loop, daemon=True).start()
print("http://192.168.137.207:{}  -- QR Detection".format(PORT))
try:
    while True: time.sleep(1)
except KeyboardInterrupt: running=False; s.shutdown()
