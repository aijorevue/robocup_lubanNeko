#!/usr/bin/env python3
"""通用颜色+形状检测模块

用法:
    from color import ColorDetector

    det = ColorDetector(
        yellow={'low':(20,80,80), 'high':(35,255,255)},
        blue  ={'low':(95,50,50), 'high':(135,255,255)},
    )
    balls   = det.detect(frame, shape='ball',   color='yellow')
    squares = det.detect(frame, shape='square', color='blue')
    rings   = det.detect(frame, shape='ring',   color='green')
    all_objs = det.detect_all(frame)

    # 绘制结果
    det.draw(frame, balls)
"""
import cv2
import numpy as np

# ===================== 默认颜色 =====================
DEFAULT_COLORS = {
    'yellow': {'low': (20, 80, 80),  'high': (35, 255, 255)},
    'blue':   {'low': (95, 50, 50),  'high': (135, 255, 255)},
    'green':  {'low': (40, 80, 80),  'high': (80, 255, 255)},
    'red':    {'low': (0, 80, 80),   'high': (10, 255, 255),
               'low2': (160, 80, 80), 'high2': (180, 255, 255)},
    'white':  {'low': (0, 0, 140),   'high': (180, 50, 255)},
}

DRAW_COLORS = {
    'yellow': (0, 255, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
    'red':    (0, 0, 255),   'white': (255, 255, 255),
}

# ===================== 检测器类 =====================
class ColorDetector:
    def __init__(self, **color_overrides):
        self.colors = {}
        for name, cfg in DEFAULT_COLORS.items():
            self.colors[name] = dict(cfg)
        for name, cfg in color_overrides.items():
            if name in self.colors:
                self.colors[name].update(cfg)
            else:
                self.colors[name] = cfg

    # ------ 掩码 ------
    def _mask(self, hsv, cname):
        cfg = self.colors[cname]
        m1 = cv2.inRange(hsv, np.array(cfg['low']), np.array(cfg['high']))
        if 'low2' in cfg:
            m2 = cv2.inRange(hsv, np.array(cfg['low2']), np.array(cfg['high2']))
            m1 = cv2.bitwise_or(m1, m2)
        m1 = cv2.erode(m1, None, iterations=1)
        m1 = cv2.dilate(m1, None, iterations=2)
        return m1

    # ------ 工具 ------
    @staticmethod
    def _circularity(c):
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0: return 0
        return 4 * np.pi * area / (peri * peri)

    @staticmethod
    def _min_circle(c):
        (x, y), r = cv2.minEnclosingCircle(c)
        return int(x), int(y), int(r)

    # ------ 球体检测 ------
    def _detect_balls(self, hsv, cname):
        mask = self._mask(hsv, cname)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in cnts:
            if cv2.contourArea(c) < 150: continue
            if self._circularity(c) < 0.55: continue
            x, y, r = self._min_circle(c)
            if r < 10: continue
            results.append({'center': (x, y), 'radius': r, 'color': cname, 'type': 'ball'})
        return results

    # ------ 方块检测 ------
    def _detect_squares(self, hsv, cname):
        mask = self._mask(hsv, cname)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in cnts:
            if cv2.contourArea(c) < 500: continue
            if self._circularity(c) > 0.7: continue
            peri = cv2.arcLength(c, True)
            poly = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(poly) != 4: continue
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), angle = rect
            if min(w, h) < 15: continue
            area_ratio = cv2.contourArea(c) / (w * h) if w * h > 0 else 0
            if area_ratio < 0.7: continue
            results.append({'center': (int(cx), int(cy)),
                           'width': int(w), 'height': int(h),
                           'angle': round(angle, 1),
                           'color': cname, 'type': 'square'})
        return results

    # ------ 圆环检测 ------
    def _detect_rings(self, hsv, cname):
        mask = self._mask(hsv, cname)
        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None: return []
        results = []
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < 300: continue
            circ = self._circularity(c)
            if circ < 0.65: continue
            x, y, outer_r = self._min_circle(c)
            if outer_r < 15: continue
            has_hole = False; inner_r = 0
            if hierarchy[0][i][2] != -1:
                child = cnts[hierarchy[0][i][2]]
                cx, cy, cr = self._min_circle(child)
                dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if (self._circularity(child) > 0.5 and cr > 5 and
                    dist < outer_r * 0.3 and cr < outer_r * 0.7):
                    has_hole = True; inner_r = cr
            if has_hole:
                results.append({'center': (x, y), 'outer_r': outer_r,
                               'inner_r': inner_r, 'color': cname, 'type': 'ring'})
        return results

    # ------ 统一接口 ------
    def detect(self, frame, shape='ball', color=None):
        """检测指定形状+颜色。shape: ball/square/ring, color: 颜色名或None=全部"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colors = [color] if color else list(self.colors.keys())
        if shape == 'ball':   fn = self._detect_balls
        elif shape == 'square': fn = self._detect_squares
        elif shape == 'ring':   fn = self._detect_rings
        else: raise ValueError('shape must be ball/square/ring, got: {}'.format(shape))
        results = []
        for cn in colors:
            if cn not in self.colors: continue
            results.extend(fn(hsv, cn))
        return results

    def detect_all(self, frame, colors=None):
        """检测所有形状"""
        if colors is None: colors = list(self.colors.keys())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        all_ = []
        for cn in colors:
            all_.extend(self._detect_balls(hsv, cn))
            all_.extend(self._detect_squares(hsv, cn))
            all_.extend(self._detect_rings(hsv, cn))
        return all_

    # ------ 可视化 ------
    def draw(self, frame, objects):
        """在图像上画出检测结果，返回结果帧"""
        img = frame.copy()
        for obj in objects:
            c = DRAW_COLORS.get(obj['color'], (0, 255, 0))
            bgr = c
            if obj['type'] == 'ball':
                x, y = obj['center']; r = obj['radius']
                cv2.circle(img, (x, y), r, bgr, 2)
                cv2.putText(img, obj['color'][0].upper(), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
            elif obj['type'] == 'square':
                rect = cv2.minAreaRect(np.array([
                    [obj['center'][0] - obj['width'] // 2, obj['center'][1] - obj['height'] // 2],
                    [obj['center'][0] + obj['width'] // 2, obj['center'][1] - obj['height'] // 2],
                    [obj['center'][0] + obj['width'] // 2, obj['center'][1] + obj['height'] // 2],
                    [obj['center'][0] - obj['width'] // 2, obj['center'][1] + obj['height'] // 2]],
                    dtype=np.float32))
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(img, [box], 0, bgr, 2)
                cv2.putText(img, obj['color'][0].upper(),
                           (obj['center'][0] + 5, obj['center'][1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
            elif obj['type'] == 'ring':
                x, y = obj['center']
                cv2.circle(img, (x, y), obj['outer_r'], bgr, 2)
                if obj['inner_r'] > 0:
                    cv2.circle(img, (x, y), obj['inner_r'], bgr, 1)
                cv2.putText(img, obj['color'][0].upper(),
                           (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
        return img

    def annotate(self, frame, objects):
        """就地标注(不拷贝)"""
        self.draw(frame, objects)
