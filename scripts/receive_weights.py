"""
scripts/receive_weights.py
==========================
Receives uploaded model weight files over HTTP PUT and saves them directly to ml/deepfake/models/.
"""

import os
from http.server import HTTPServer, BaseHTTPRequestHandler

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "deepfake", "models")
os.makedirs(DEST_DIR, exist_ok=True)

class UploadHandler(BaseHTTPRequestHandler):
    def do_PUT(self):
        filename = os.path.basename(self.path.lstrip("/"))
        if not filename:
            filename = "uploaded_weights.pt"
            
        dest_path = os.path.join(DEST_DIR, filename)
        content_length = int(self.headers.get("Content-Length", 0))
        
        print(f"[*] Receiving {filename} ({content_length / (1024*1024):.2f} MB)...")
        bytes_read = 0
        chunk_size = 65536
        
        with open(dest_path, "wb") as f:
            while bytes_read < content_length:
                to_read = min(chunk_size, content_length - bytes_read)
                chunk = self.rfile.read(to_read)
                if not chunk:
                    break
                f.write(chunk)
                bytes_read += len(chunk)
                
        print(f"[OK] Successfully saved: {dest_path} ({bytes_read / (1024*1024):.2f} MB)")
        
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(f"OK: Saved {filename}".encode("utf-8"))

    def log_message(self, format, *args):
        # Clean logging
        print(f"[HTTP] {format % args}")

def run():
    server = HTTPServer(("0.0.0.0", 9999), UploadHandler)
    print(f"[*] Receiver listening on port 9999... Saving directly to {DEST_DIR}")
    server.serve_forever()

if __name__ == "__main__":
    run()
