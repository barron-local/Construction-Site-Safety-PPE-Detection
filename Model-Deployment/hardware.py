import glob
import os
import threading
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

PATTERNS = ('/dev/cu.usbmodem*', '/dev/cu.usbserial*', '/dev/cu.wchusbserial*',
            '/dev/cu.SLAB_USBtoUART*', '/dev/ttyUSB*', '/dev/ttyACM*')


class HardwareLink:
    """ส่งสถานะให้ Arduino/ESP32 ผ่าน USB serial: IDLE / CHECK / PASS / FAIL <รายการที่ขาด>
    ตั้งพอร์ตเองได้ด้วย PPE_SERIAL_PORT=COM5 (Windows) หรือ /dev/cu.usbmodemXXXX (Mac/Linux)
    และตั้ง Baud rate ด้วย PPE_SERIAL_BAUD=9600 หรือ 115200 (Default 115200)"""

    def __init__(self, baud=None):
        if baud is None:
            baud = int(os.environ.get('PPE_SERIAL_BAUD', 115200))
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
        # 1. ค้นหาอัตโนมัติผ่าน pyserial (รองรับทั้ง Windows COM*, Linux /dev/tty*, macOS /dev/cu*)
        if serial and hasattr(serial, 'tools') and hasattr(serial.tools, 'list_ports'):
            try:
                ports = list(serial.tools.list_ports.comports())
                # ค้นหาบอร์ดไมโครคอนโทรลเลอร์ที่พบบ่อย (Arduino, CH340, FTDI, CP210x, ESP32)
                for p in ports:
                    desc = ((p.description or '') + ' ' + (p.hwid or '')).lower()
                    if any(k in desc for k in ('ch340', 'arduino', 'usb serial', 'ftdi', 'cp210', 'wch', 'usb-to-serial')):
                        return p.device
                if ports:
                    return ports[0].device
            except Exception:
                pass
        # 2. Fallback to Unix patterns
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
                self.ser.flush()
                self.sent = True
                desc = '1 (PASS -> ไฟเขียว)' if self.want == '1' else '0 (FAIL -> ไฟแดง)' if self.want == '0' else '2 (IDLE -> ดับไฟ)'
                print(f'[Hardware] Sent to Arduino: {desc}')
            except Exception as e:
                print(f'[Hardware] Serial write error: {e}')
                self.ser, self.port = None, None

    def update(self, st):
        self.start()
        state = st.get('state', 'idle')
        # ส่งค่ารหัสตัวเลขให้ Arduino: '1' = ผ่าน (ไฟเขียว), '0' = ไม่ผ่าน (ไฟแดง), '2' = ดับไฟ (IDLE/CHECK)
        if state == 'pass':
            msg = '1'
        elif state == 'fail':
            msg = '0'
        else:
            msg = '2'
        with self.lock:
            if msg != self.want:
                self.want, self.sent = msg, False
            self._flush()

    def status(self):
        return {'connected': self.ser is not None, 'port': self.port,
                'pyserial': serial is not None}
