import re
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime

DB = 'detections.db'  # ไฟล์เดียวกับ database.py
CODE_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
CODE_TTL = 60  # วินาที: รหัสที่รับไว้ตอนยังไม่มีใครเข้าตรวจ ใช้ได้นานเท่านี้


def _q(sql, args=()):
    with closing(sqlite3.connect(DB)) as c:
        rows = c.execute(sql, args).fetchall()
        c.commit()
        return rows


def init():
    _q('CREATE TABLE IF NOT EXISTS employees (code TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT "")')
    _q('CREATE TABLE IF NOT EXISTS check_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, emp_code TEXT, '
       'checked_at TEXT NOT NULL, result TEXT NOT NULL, missing TEXT NOT NULL DEFAULT "")')
    _q('CREATE INDEX IF NOT EXISTS idx_check_logs_time ON check_logs (checked_at)')
    _q('CREATE TABLE IF NOT EXISTS check_images (log_id INTEGER PRIMARY KEY, jpg BLOB NOT NULL)')


def valid_code(code):
    return bool(code) and bool(CODE_RE.match(code))


def name_of(code):
    rows = _q('SELECT name FROM employees WHERE code = ?', (code,))
    return rows[0][0] if rows else ''


def add_employees(text):
    """หนึ่งบรรทัดต่อคน: รหัส,ชื่อ  คืนจำนวนที่เพิ่มหรืออัปเดต"""
    n = 0
    for line in (text or '').splitlines():
        parts = re.split(r'[,\t ]+', line.strip(), maxsplit=1)
        code = parts[0] if parts else ''
        if valid_code(code):
            _q('INSERT OR REPLACE INTO employees (code, name) VALUES (?, ?)',
               (code, parts[1].strip() if len(parts) > 1 else ''))
            n += 1
    return n


def delete_employee(code):
    _q('DELETE FROM employees WHERE code = ?', (code,))


def log_check(code, result, missing):
    """บันทึกผลหนึ่งครั้ง คืน id ของแถวที่เพิ่ม"""
    with closing(sqlite3.connect(DB)) as c:
        cur = c.execute('INSERT INTO check_logs (emp_code, checked_at, result, missing) VALUES (?, ?, ?, ?)',
                        (code, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), result, ','.join(missing or [])))
        c.commit()
        return cur.lastrowid


def save_image(log_id, jpg):
    """เก็บภาพ JPEG ของการตรวจครั้งนั้นลงฐานข้อมูล (ตารางแยก เพื่อไม่ให้ตารางผลหนัก)"""
    _q('INSERT OR REPLACE INTO check_images (log_id, jpg) VALUES (?, ?)', (log_id, sqlite3.Binary(jpg)))


def get_image(log_id):
    rows = _q('SELECT jpg FROM check_images WHERE log_id = ?', (log_id,))
    return bytes(rows[0][0]) if rows else None


def recent_logs(limit=50):
    rows = _q('SELECT l.id, l.checked_at, l.emp_code, e.name, l.result, l.missing, '
              '(SELECT 1 FROM check_images i WHERE i.log_id = l.id) FROM check_logs l '
              'LEFT JOIN employees e ON e.code = l.emp_code ORDER BY l.id DESC LIMIT ?', (limit,))
    return [{'id': i, 'time': t, 'code': c, 'name': n or '', 'result': r,
             'missing': m.split(',') if m else [], 'has_image': bool(img)}
            for i, t, c, n, r, m, img in rows]


def today_summary():
    """สถานะของพนักงานแต่ละคนวันนี้: ใช้ผลล่าสุดของวัน ถ้าไม่มีผลเลยคือ 'none' (ยังไม่ตรวจ)"""
    today = datetime.now().strftime('%Y-%m-%d')
    latest = {r[0]: r for r in _q(
        'SELECT emp_code, result, missing, checked_at FROM check_logs WHERE id IN '
        '(SELECT MAX(id) FROM check_logs WHERE checked_at >= ? AND emp_code IS NOT NULL GROUP BY emp_code)',
        (today,))}
    rows, counts = [], {'pass': 0, 'fail': 0, 'none': 0}
    for code, name in _q('SELECT code, name FROM employees ORDER BY code'):
        r = latest.get(code)
        status = r[1] if r else 'none'
        counts[status] += 1
        rows.append({'code': code, 'name': name, 'status': status,
                     'time': r[3] if r else None, 'missing': r[2].split(',') if r and r[2] else []})
    unknown = dict(_q('SELECT result, COUNT(*) FROM check_logs WHERE checked_at >= ? AND '
                      '(emp_code IS NULL OR emp_code NOT IN (SELECT code FROM employees)) GROUP BY result',
                      (today,)))
    return {'date': today, 'counts': counts, 'employees': rows, 'unknown': unknown}


class CheckRecorder:
    """จำรหัสพนักงานของรอบตรวจปัจจุบัน และบันทึกผลเมื่อสถานะเปลี่ยนจาก checking เป็น pass/fail"""

    def __init__(self):
        self.lock = threading.Lock()
        self.code = None
        self.name = ''
        self.code_time = 0.0
        self.prev = 'idle'

    def set_code(self, code):
        """คืน True ถ้าเป็นรหัสใหม่ที่ใช้ได้"""
        code = (code or '').strip()
        if not valid_code(code):
            return False
        name = name_of(code)
        with self.lock:
            same = code == self.code
            self.code, self.name, self.code_time = code, name, time.time()
            return not same

    def current(self):
        with self.lock:
            if self.code and self.prev == 'idle' and time.time() - self.code_time > CODE_TTL:
                self.code = None
            return {'code': self.code, 'name': self.name} if self.code else None

    def observe(self, st):
        """คืน {'id', 'result'} ของแถวที่เพิ่งบันทึก ถ้ารอบนี้เพิ่งตัดสินผล ไม่งั้นคืน None"""
        s = st['state']
        logged = None
        with self.lock:
            if s == 'checking' and self.prev == 'idle' and self.code and time.time() - self.code_time > CODE_TTL:
                self.code = None
            if self.prev == 'checking' and s in ('pass', 'fail'):
                logged = {'id': log_check(self.code, s, st['missing']), 'result': s}
            if s == 'idle' and self.prev != 'idle':
                self.code = None  # จบรอบแล้ว คนถัดไปต้องแสดงรหัสใหม่
            self.prev = s
        return logged
