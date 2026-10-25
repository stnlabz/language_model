#!/usr/bin/env python3
"""Convert explicitly approved Digit JSONL to language-model training text.
Never modify the input. Fail on malformed or unapproved records.
"""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    args = ap.parse_args()
    src, dst = Path(args.source), Path(args.output)
    if not src.is_file() or dst.exists() or src.resolve() == dst.resolve():
        ap.error("source must exist and output must not exist")
    count = 0
    text = []
    with src.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                ap.error(f"invalid JSON at line {n}: {e}")
            if not isinstance(rec, dict) or rec.get("approved") is not True:
                continue
            question, response = rec.get("input"), rec.get("response")
            if not isinstance(response, str) or not response.strip():
                continue
            if isinstance(question, str) and not question.startswith((
                "Digit Corpus record ", "Digit Context record ",
                "Digit Observation record "
            )):
                text.append(f"Question: {question.strip()}\nAnswer: {response.strip()}\n")
                count += 1
            elif isinstance(question, str) and question.startswith("Digit Corpus record "):
                text.append(response.strip() + "\n")
                count += 1
            elif rec.get("type") == "statement" and isinstance(rec.get("text"), str):
                text.append(rec["text"].strip() + "\n")
                count += 1
    if count < 5:
        ap.error("fewer than five approved usable records")
    with dst.open("x", encoding="utf-8") as f:
        f.writelines(text)
    print(f"approved_training_records={count} bytes={dst.stat().st_size}")

if __name__ == "__main__":
    main()
