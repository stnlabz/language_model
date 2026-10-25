#!/usr/bin/env python3
"""gru_learn: approved JSONL -> prepared corpus -> CPU LM checkpoint.
The component name is historical; underlying LM architecture is Transformer.
No service management, agent mutations, or deployment side effects.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, default=ROOT / "digit_approved_learning.jsonl")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "llm.stn")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--binary", type=Path, default=ROOT / "micro_llm")
    args = p.parse_args()
    if args.steps < 1 or args.steps > 200000:
        p.error("steps must be between 1 and 200000")
    if not args.jsonl.is_file() or not args.binary.is_file():
        p.error("approved JSONL and compiled model binary must exist")
    if args.checkpoint.exists():
        p.error("checkpoint exists; refusing to overwrite")
    digest = hashlib.sha256(args.jsonl.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="stn-gru-learn-") as tmp:
        corpus = Path(tmp) / "corpus.txt"
        subprocess.run([sys.executable, str(ROOT / "prepare_jsonl.py"),
                        str(args.jsonl), str(corpus)], check=True)
        subprocess.run([str(args.binary.resolve()), "train", str(corpus),
                        str(args.checkpoint.resolve()), str(args.steps)], check=True)
    print(json.dumps({"status": "trained", "checkpoint": str(args.checkpoint),
                      "approved_jsonl_sha256": digest, "steps": args.steps}))
if __name__ == "__main__":
    main()
