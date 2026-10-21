#!/usr/bin/env python3
"""Experimental loopback-only HTTP adapter for STN-Labz native model.

AI-assisted. No agent integration, deployment hooks, third-party packages,
remote service calls, or training. Optional data ingestion writes to a
private JSONL file only when explicitly configured. Operator starts and stops this foreground
process explicitly. Do not expose it through a public proxy.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

ENCODED = re.compile(r"(?:\\x[0-9a-f]{2})+")


def generate(binary, checkpoint, request):
    if not isinstance(request, dict) or set(request) - {"prompt", "length", "seed", "temperature", "penalty"}:
        raise ValueError("unsupported request fields")
    prompt = request.get("prompt")
    count = request.get("length", 128)
    seed = request.get("seed", 913)
    temperature = request.get("temperature", 0.65)
    penalty = request.get("penalty", 0)
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 1024 or "\x00" in prompt:
        raise ValueError("prompt must be 1..1024 UTF-8 bytes and contain no NUL")
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("length must be an integer from 1 to 256")
    if type(seed) is not int or not 1 <= seed <= 2**64 - 1:
        raise ValueError("seed must be a nonzero uint64")
    if (type(temperature) not in (float, int) or not 0.25 <= temperature <= 2 or
            type(penalty) not in (float, int) or not 0 <= penalty <= 2):
        raise ValueError("temperature or penalty out of range")
    result = subprocess.run(
        [binary, "sample", checkpoint, prompt, str(count), str(seed),
         str(temperature), str(penalty), "utf8"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    encoded = result.stdout.strip()
    if len(encoded) != 4 * count or not ENCODED.fullmatch(encoded):
        raise RuntimeError("invalid model output")
    output = bytes.fromhex(encoded.replace("\\x", "")).decode("utf-8")
    return {"text": output, "generated_bytes": count, "model": "stn-labz-gru",
            "experimental": True}


def validate_learning(request):
    """Validate a bounded, explicitly approved Digit exchange."""
    required = {"input", "response", "source", "approved"}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("learn requires input, response, source, approved")
    if request["approved"] is not True:
        raise ValueError("record must be explicitly approved")
    for key, limit in (("input", 1024), ("response", 2048)):
        value = request[key]
        if (not isinstance(value, str) or not value.strip() or
                "\x00" in value or len(value.encode("utf-8")) > limit):
            raise ValueError(f"{key} must be nonempty UTF-8, max {limit} bytes, without NUL")
    if request["source"] != "digit":
        raise ValueError("source must be digit")
    return {"input": request["input"], "response": request["response"],
            "source": "digit", "approved": True}


def append_learning(path, record):
    """Append private JSONL records; no training or checkpoint update."""
    line = (json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not os.path.isfile(path):
            raise OSError("learning path must be a regular file")
        if os.write(descriptor, line) != len(line):
            raise OSError("incomplete learning write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Handler(BaseHTTPRequestHandler):
    binary = "./model"
    checkpoint = ""
    learn_file = None

    def _respond(self, code, data):
        payload = json.dumps(data, ensure_ascii=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            return self._respond(200, {"status": "ok", "experimental": True})
        return self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/generate", "/learn"):
            return self._respond(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            size = -1
        if not 1 <= size <= 4096:
            self.close_connection = True
            return self._respond(413, {"error": "request size must be 1..4096 bytes"})
        try:
            data = json.loads(self.rfile.read(size).decode("utf-8"))
            if self.path == "/learn":
                if not self.learn_file:
                    return self._respond(503, {"error": "learning ingestion disabled"})
                record = validate_learning(data)
                append_learning(self.learn_file, record)
                return self._respond(202, {"accepted": True, "trained": False,
                                            "status": "queued_for_review"})
            result = generate(self.binary, self.checkpoint, data)
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            return self._respond(400, {"error": str(exc)})
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError):
            return self._respond(502, {"error": "model inference failed"})
        return self._respond(200, result)

    def log_message(self, format, *args):
        # Do not log prompts or generated content.
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--learn-file", help="explicit private JSONL ingestion destination")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be 1024..65535")
    if not pathlib.Path(args.model).is_file() or not pathlib.Path(args.checkpoint).is_file():
        parser.error("model executable or checkpoint missing")
    Handler.learn_file = args.learn_file
    Handler.binary = args.model
    Handler.checkpoint = args.checkpoint
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"STN-Labz experimental API listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
