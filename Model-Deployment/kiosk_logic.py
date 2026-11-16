import math
import threading
import time

KEYWORDS = {
    'helmet': ('hardhat', 'hard hat', 'hard-hat', 'helmet'),
    'vest': ('vest',),
    'mask': ('mask', 'respirator'),
}
ORDER = ['helmet', 'vest', 'mask']

# ปรับค่าได้ที่นี่ หรือใส่ใน settings.json เป็น kiosk_<ชื่อ> เช่น kiosk_check_seconds
DEFAULTS = dict(
    check_seconds=4.0,       # เวลาตรวจต่อรอบ
    pass_hold=3.0,           # แสดง "ผ่าน" อย่างน้อยกี่วินาที
    fail_hold=4.0,           # แสดง "ไม่ผ่าน" กี่วินาที แล้วตรวจใหม่ถ้ายังยืนอยู่
    leave_seconds=2.0,       # ไม่เห็นคนกี่วินาทีถึงกลับไปรอคนถัดไป
    enter_seconds=0.6,       # ต้องเห็นคนต่อเนื่องกี่วินาทีถึงเริ่มตรวจ
    lost_grace=1.0,          # คนหายจากภาพชั่วคราวได้กี่วินาที
    min_ratio=0.5,           # ต้องเห็นอุปกรณ์ในกี่ส่วนของเฟรมถึงนับว่าใส่
    conf_min=0.5,            # ความมั่นใจขั้นต่ำของอุปกรณ์
    person_conf_min=0.5,
    min_person_height=0.35,  # คนต้องสูงอย่างน้อยกี่ส่วนของภาพ (ไกลเกินไม่ตรวจ)
    min_frames=3,
)


def classify(name):
    """คืน (item, is_negative) เช่น 'NO-Safety Vest' -> ('vest', True)"""
    n = name.lower().strip()
    neg = n.startswith(('no-', 'no ', 'no_'))
    for item, words in KEYWORDS.items():
        if any(w in n for w in words):
            return item, neg
    return None, False


class KioskSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.state = 'idle'
            self.since = time.time()
            self.last_seen = 0.0
            self.present_since = None
            self.hint = None
            self.result = {}
            self.missing = []
            self._clear_tally()
            self.snap = self._snapshot(time.time(), [], DEFAULTS)

    def restart(self):
        """เริ่มนับเวลาตรวจใหม่ (ใช้เมื่อเพิ่งอ่านรหัสพนักงาน)"""
        with self.lock:
            if self.state == 'checking':
                self._begin(time.time())

    def _clear_tally(self):
        self.total = 0
        self.start = None
        self.pos = {k: 0 for k in ORDER}
        self.neg = {k: 0 for k in ORDER}

    def _go(self, state, now):
        self.state = state
        self.since = now
        if state in ('idle', 'checking'):
            self.result, self.missing = {}, []
        if state == 'idle':
            self._clear_tally()

    def _begin(self, now):
        self._go('checking', now)
        self._clear_tally()
        self.start = now

    def _subject(self, dets, frame_h, cfg):
        best, best_area, too_small, n = None, 0, False, 0
        for d in dets:
            if d['class'].lower() != 'person' or d['confidence'] < cfg['person_conf_min']:
                continue
            x1, y1, x2, y2 = d['bbox']
            if (y2 - y1) < cfg['min_person_height'] * frame_h:
                too_small = True
                continue
            n += 1
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best, best_area = d, area
        return best, too_small, n

    def _tally(self, dets, subject, cfg):
        x1, y1, x2, y2 = subject['bbox']
        mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        pos, neg = set(), set()
        for d in dets:
            item, is_neg = classify(d['class'])
            if not item or d['confidence'] < cfg['conf_min']:
                continue
            bx1, by1, bx2, by2 = d['bbox']
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
                (neg if is_neg else pos).add(item)
        self.total += 1
        for i in pos:
            self.pos[i] += 1
        for i in neg:
            self.neg[i] += 1

    def _worn(self, item, cfg):
        return (self.total > 0
                and self.pos[item] / self.total >= cfg['min_ratio']
                and self.pos[item] > self.neg[item])

    def _finish(self, required, cfg, now):
        self.result = {i: ('ok' if self._worn(i, cfg) else 'missing') for i in required}
        self.missing = [i for i, v in self.result.items() if v == 'missing']
        self._go('fail' if self.missing else 'pass', now)

    def update(self, detections, frame_shape, settings=None, now=None):
        now = time.time() if now is None else now
        s = settings or {}
        cfg = {k: s.get('kiosk_' + k, v) for k, v in DEFAULTS.items()}
        req_cfg = s.get('required_ppe') or {'helmet': True, 'vest': True}
        required = [k for k in ORDER if req_cfg.get(k)]
        with self.lock:
            subject, too_small, n = self._subject(detections, frame_shape[0], cfg)
            crowded = n > 1  # มีคนใกล้กล้องมากกว่าหนึ่งคน
            present = subject is not None
            if present:
                self.last_seen = now
                if self.present_since is None:
                    self.present_since = now
            elif now - self.last_seen > cfg['lost_grace']:
                self.present_since = None
            if crowded and self.state in ('idle', 'checking'):
                self.hint = 'multiple'
            else:
                self.hint = 'too_far' if (too_small and not present and self.state == 'idle') else None

            if self.state == 'idle':
                if present and not crowded and now - self.present_since >= cfg['enter_seconds']:
                    self._begin(now)
            elif self.state == 'checking':
                if crowded:
                    self._begin(now)  # เริ่มนับใหม่ รอจนเหลือคนเดียว
                elif present:
                    self._tally(detections, subject, cfg)
                elapsed = now - self.start
                if not present and now - self.last_seen > cfg['lost_grace']:
                    self._go('idle', now)
                elif elapsed >= cfg['check_seconds']:
                    if self.total >= cfg['min_frames']:
                        self._finish(required, cfg, now)
                    elif elapsed >= 2 * cfg['check_seconds']:
                        self._go('idle', now)
            elif self.state == 'pass':
                if now - self.since >= cfg['pass_hold'] and now - self.last_seen >= cfg['leave_seconds']:
                    self._go('idle', now)
            elif self.state == 'fail':
                if now - self.since >= cfg['fail_hold']:
                    if present:
                        self._begin(now)
                    elif now - self.last_seen >= cfg['leave_seconds']:
                        self._go('idle', now)

            self.snap = self._snapshot(now, required, cfg)
            return self.snap

    def _snapshot(self, now, required, cfg):
        checking = self.state == 'checking' and self.start is not None
        if self.state in ('pass', 'fail'):
            items = dict(self.result)
        else:
            items = {i: ('ok' if checking and self.total >= 2 and self._worn(i, cfg) else 'pending')
                     for i in required}
        elapsed = (now - self.start) if checking else 0
        return {
            'state': self.state,
            'progress': 1.0 if self.state in ('pass', 'fail') else round(min(1.0, elapsed / cfg['check_seconds']), 3),
            'seconds_left': max(0, math.ceil(cfg['check_seconds'] - elapsed)) if checking else 0,
            'required': required,
            'items': items,
            'missing': list(self.missing),
            'hint': self.hint,
            'updated_at': time.time(),
        }

    def get_state(self):
        with self.lock:
            s = dict(self.snap)
        s['stale'] = time.time() - s['updated_at'] > 2.5
        return s
