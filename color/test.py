#!/usr/bin/env python3
"""ColorDetector web test — browser live view + shape/color buttons"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, time, threading, http.server, socketserver
from color_detector import ColorDetector, DEFAULT_COLORS

CUSTOM = {}
shape_mode = 'ball'
color_mode = 'yellow'

frame, info_text = None, ""
lock = threading.Lock()
running = True
det = ColorDetector(**CUSTOM)

PAGE = '\n'.join([
    '<html><head><meta charset="utf-8"><title>Color Detector</title></head>',
    '<body style="margin:0;background:#000;text-align:center">',
    '<img src="/video" style="max-width:100%">',
    '<div style="color:#fff;font-size:16px;padding:8px;font-family:monospace" id="info">---</div>',
    '<button onclick="fetch(\'/cmd?m=yellow-ball\')">Yellow Ball</button>',
    '<button onclick="fetch(\'/cmd?m=blue-ball\')">Blue Ball</button>',
    '<button onclick="fetch(\'/cmd?m=green-ring\')">Green Ring</button>',
    '<button onclick="fetch(\'/cmd?m=red-square\')">Red Square</button>',
    '<button onclick="fetch(\'/cmd?m=auto\')">Auto</button>',
    '<script>var es=new EventSource("/stats");es.onmessage=function(e){document.getElementById("info").innerHTML=e.data}</script>',
    '</body></html>',
])

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global shape_mode, color_mode
        if self.path == '/':
            self.send_response(200); self.send_header('Content-type','text/html'); self.end_headers()
            self.wfile.write(PAGE.encode())
        elif self.path.startswith('/cmd'):
            m = self.path.split('=')[-1]
            if m == 'auto': color_mode = 'auto'
            elif '-' in m:
                cm, sm = m.split('-')
                color_mode = cm; shape_mode = sm
            self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
        elif self.path == '/video':
            self.send_response(200); self.send_header('Content-type','multipart/x-mixed-replace;boundary=f'); self.end_headers()
            try:
                while running:
                    with lock: img = frame.copy() if frame is not None else None
                    if img is not None:
                        _, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        self.wfile.write(b'--f\r\nContent-Type:image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
                    time.sleep(0.04)
            except: pass
        elif self.path == '/stats':
            self.send_response(200); self.send_header('Content-type','text/event-stream'); self.end_headers()
            try:
                while running:
                    self.wfile.write(('data: ' + info_text + '\n\n').encode()); self.wfile.flush(); time.sleep(0.3)
            except: pass
    def log_message(self, *a): pass

def loop():
    global frame, info_text, shape_mode, color_mode
    cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_FPS, 30)

    while running:
        ret, img = cap.read()
        if not ret: continue

        # -- 自动状态机 --
        if color_mode == 'auto':
            yellow_balls = det.detect(img, shape='ball', color='yellow')
            if yellow_balls:
                shape_mode = 'ball'; color_mode = 'yellow'
            else:
                shape_mode = 'ball'; color_mode = 'blue'

        objs = det.detect(img, shape=shape_mode, color=color_mode)
        vis = det.draw(img, objs)

        color_cnts = ' | '.join(['{}:{}'.format(cn, sum(1 for o in objs if o['color']==cn)) for cn in DEFAULT_COLORS])
        info_text = 'State: {}-{} | {} | Auto:{}'.format(color_mode, shape_mode, color_cnts, 'ON' if color_mode=='auto' else 'OFF')
        cv2.putText(vis, info_text.replace('<br>',' '), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        with lock: frame = vis
    cap.release()

def main():
    global running
    s = http.server.HTTPServer(('0.0.0.0', 8081), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    print('\n  http://198.18.0.207:8081\n  Auto-mode: yellow ball -> blue ball\n')
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        running = False; s.shutdown()

if __name__ == '__main__':
    main()
