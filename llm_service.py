#!/usr/bin/env python3
"""Read-only loopback inference API for existing STN-Labz LM checkpoints.
Explicit start only. Never trains, installs, or manages the service.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess

MAX_BODY = 8192

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=Path(__file__).resolve().parent / "micro_llm")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if args.host != "127.0.0.1":
        ap.error("only loopback binding is supported")
    if not 1024 <= args.port <= 65535:
        ap.error("invalid port")
    if not args.binary.is_file() or not args.checkpoint.is_file():
        ap.error("compiled model and checkpoint must exist")
    model = str(args.binary.resolve())
    checkpoint = str(args.checkpoint.resolve())
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, obj):
            payload = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ready", "model": "micro_llm"})
            else:
                self.send_json(404, {"error": "not found"})
        def do_POST(self):
            if self.path != "/generate":
                self.send_json(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= MAX_BODY:
                    self.send_json(413, {"error": "invalid request size"})
                    return
                obj = json.loads(self.rfile.read(size))
                prompt = obj.get("prompt")
                length = obj.get("max_bytes", 64)
                seed = obj.get("seed", 913)
                if (not isinstance(prompt, str) or not 1 <= len(prompt) <= 2048 or
                    "\x00" in prompt or not isinstance(length, int) or
                    isinstance(length, bool) or not 1 <= length <= 256 or
                    not isinstance(seed, int) or isinstance(seed, bool) or
                    not 0 <= seed <= 2147483647):
                    raise ValueError("invalid generation parameters")
                result = subprocess.run([model, "generate", checkpoint, prompt,
                                         str(length), str(seed)], capture_output=True,
                                         timeout=30, check=False)
                if result.returncode:
                    self.send_json(503, {"error": "model generation failed"})
                    return
                self.send_json(200, {"text": result.stdout.decode("utf-8", errors="replace").rstrip("\n")})
            except (ValueError, TypeError, json.JSONDecodeError):
                self.send_json(400, {"error": "invalid request"})
            except subprocess.TimeoutExpired:
                self.send_json(504, {"error": "model generation timeout"})
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"LM inference listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
if __name__ == "__main__":
    main()
