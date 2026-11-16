import glob
import os
import threading
import time

try:
    import serial
except ImportError:
    serial = None

PATTERNS = ('/dev/cu.usbmodem*', '/dev/cu.usbserial*', '/dev/cu.wchusbserial*',
            '/dev/cu.SLAB_USBtoUART*', '/dev/ttyUSB*', '/dev/ttyACM*')


class HardwareLink:
    """ส่งสถานะให้ Arduino/ESP32 ผ่าน USB serial: IDLE / CHECK / PASS / FAIL <รายการที่ขาด>
    ตั้งพอร์ตเองได้ด้วย PPE_SERIAL_PORT=/dev/cu.usbmodemXXXX ไม่มีบอร์ดก็ทำงานได้ตามปกติ"""

    def __init__(self, baud=115200):
        self.baud = baud
        self.ser = None
        self.port = None
        self.want = None
        self.sent = False
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        if self.started:
            return
        self.started = True
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _find_port(self):
        env = os.environ.get('PPE_SERIAL_PORT')
        if env:
            return env
        for p in PATTERNS:
            found = sorted(glob.glob(p))
            if found:
                return found[0]
        return None

    def _keepalive(self):
        while True:
            if self.ser is None and serial is not None:
                port = self._find_port()
                if port:
                    try:
                        ser = serial.Serial(port, self.baud, timeout=0.1)
                        time.sleep(2)  # Arduino รีเซ็ตตัวเองตอนเปิดพอร์ต
                        with self.lock:
                            self.ser, self.port, self.sent = ser, port, False
                            self._flush()
                        print(f'[Hardware] connected: {port}')
                    except Exception as e:
                        print(f'[Hardware] cannot open {port}: {e}')
            time.sleep(3)

    def _flush(self):
        if self.ser and self.want and not self.sent:
            try:
                self.ser.write((self.want + '\n').encode())
                self.sent = True
            except Exception:
                self.ser, self.port = None, None

    def update(self, st):
        self.start()
        if st['state'] == 'fail':
            msg = 'FAIL ' + ','.join(st['missing'])
        else:
            msg = {'idle': 'IDLE', 'checking': 'CHECK', 'pass': 'PASS'}[st['state']]
        with self.lock:
            if msg != self.want:
                self.want, self.sent = msg, False
            self._flush()

    def status(self):
        return {'connected': self.ser is not None, 'port': self.port,
                'pyserial': serial is not None}
