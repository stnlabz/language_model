#!/usr/bin/env python3
"""Convert an explicit, immutable snapshot of Digit /learn JSONL to GRU text.

Does not train, alter checkpoints, or touch the source. Review produced text
before explicitly calling the existing prepare executable.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import sys

MAX_BYTES = 16 * 1024 * 1024
MAX_INPUT = 64 * 1024 * 1024
KEYS = {"input", "response", "source", "approved"}


def clean_field(value, limit, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    if "\x00" in value or len(value.encode("utf-8")) > limit:
        raise ValueError(f"{label} exceeds bounds or contains NUL")
    return " ".join(value.split())


def convert(blob, mode="all"):
    """Strict snapshot parser: never silently skip malformed/incomplete lines."""
    if not blob or len(blob) > MAX_INPUT or not blob.endswith(b"\n"):
        raise ValueError("snapshot must be nonempty, <=64 MiB, and end on a complete line")
    rows = []
    seen = set()
    stats = collections.Counter()
    for index, raw in enumerate(blob.splitlines(), 1):
        if not raw:
            raise ValueError(f"line {index}: blank record")
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"line {index}: invalid JSON/UTF-8: {exc}") from exc
        if not isinstance(record, dict) or set(record) != KEYS:
            raise ValueError(f"line {index}: unexpected record schema")
        if record["approved"] is not True or record["source"] != "digit":
            raise ValueError(f"line {index}: not an approved Digit record")
        try:
            question = clean_field(record["input"], 1024, "input")
            answer = clean_field(record["response"], 2048, "response")
        except ValueError as exc:
            raise ValueError(f"line {index}: {exc}") from exc
        stats["read"] += 1
        is_structured = question.startswith((
            "Digit Context record ", "Digit Corpus record ",
            "Digit Observation record "))
        if is_structured:
            stats["structured_records"] += 1
        if mode == "dialogue" and is_structured:
            stats["excluded_structured"] += 1
            continue
        pair = (question, answer)
        if pair in seen:
            stats["duplicates"] += 1
            continue
        seen.add(pair)
        # Keep each record separate while retaining source approval checks.
        # Structured records are approved statements, not dialogue turns.
        # Preserve their approved response text without adding Q/A markers.
        if is_structured:
            rows.append(f"{answer}\n")
        else:
            rows.append(f"Question: {question} Answer: {answer}\n")
    data = "".join(rows).encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("converted corpus exceeds existing prepare.c 16 MiB limit")
    stats["written"] = len(rows)
    stats["output_bytes"] = len(data)
    return data, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="frozen JSONL snapshot")
    parser.add_argument("--output", required=True, help="new UTF-8 corpus filename")
    parser.add_argument("--mode", choices=("all", "dialogue"), default="all",
                        help="dialogue excludes Digit Context/Corpus/Observation record labels")
    args = parser.parse_args()
    source = pathlib.Path(args.input)
    target = pathlib.Path(args.output)
    if source.resolve() == target.resolve():
        parser.error("input and output must differ")
    if not source.is_file() or target.exists():
        parser.error("input must exist and output must not already exist")
    blob = source.read_bytes()
    data, stats = convert(blob, args.mode)
    if stats["written"] < 5:
        raise ValueError("fewer than five records; not enough for corpus preparation")
    # Refuse overwrite even if another process creates the output meanwhile.
    with target.open("xb") as output:
        output.write(data)
    print("migration=digit_jsonl_to_gru_utf8")
    print(f"mode={args.mode} source_sha256={hashlib.sha256(blob).hexdigest()}")
    for key in ("read", "structured_records", "excluded_structured", "duplicates",
                "written", "output_bytes"):
        print(f"{key}={stats[key]}")
    print(f"output_sha256={hashlib.sha256(data).hexdigest()}")
    print("status=converted_only training_started=no")
    print("note=Inspect output and source approval; this tool does not train the GRU.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
