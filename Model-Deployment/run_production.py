"""
Production WSGI Server runner using Waitress
No development server warning, optimized for Windows deployment.
"""
from waitress import serve
import app as ppe_app
import os

if __name__ == '__main__':
    print("[+] Initializing Database & PPE AI Models on GPU...")
    ppe_app.db.init_db()
    ppe_app.load_model()
    ppe_app.instance_detector.update_settings(ppe_app.current_settings)
    
    port = int(os.environ.get('PORT', 3333))
    print(f"[+] [PRODUCTION WSGI] Serving application with Waitress on http://localhost:{port}")
    print("[+] Press Ctrl+C to stop.")
    serve(ppe_app.app, host='0.0.0.0', port=port, threads=8)
