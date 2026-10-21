#!/usr/bin/env python3
"""Controlled, read-only comparison of generation-only sampling settings.

Scores describe output form; they do not assess comprehension or fluency.
"""
import argparse
import pathlib
import re
import subprocess
import sys

PROMPTS = (
    "The engineer ",
    "Digit ",
    "A question ",
    "The engineer checks the sensor before ",
    "The controller receives a request and ",
)
CONFIGS = (
    ("legacy", None, None),
    ("cool", "0.65", "0"),
    ("balanced", "0.90", "0.15"),
    ("warm", "1.20", "0.15"),
    ("anti_repeat", "0.90", "0.50"),
)
ENCODED = re.compile(r"(?:\\x[0-9a-f]{2})+")


def run(binary, checkpoint, prefix, count, seed, temperature, penalty):
    command = [binary, "generate" if temperature is None else "sample",
               checkpoint, prefix, str(count), str(seed)]
    if temperature is None:
        command.append("utf8")
    else:
        command.extend((temperature, penalty, "utf8"))
    process = subprocess.run(command, text=True, capture_output=True,
                             check=True, timeout=60)
    encoded = process.stdout.strip()
    if len(encoded) != count * 4 or not ENCODED.fullmatch(encoded):
        raise ValueError("malformed encoded generation")
    result = bytes.fromhex(encoded.replace("\\x", ""))
    result.decode("utf-8")  # Fail on malformed output.
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--bytes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=913)
    args = parser.parse_args()
    if not 1 <= args.bytes <= 256 or not 1 <= args.seed <= 2**64-1:
        parser.error("bytes must be 1..256 and seed must be nonzero uint64")
    words = {w.lower() for w in re.findall(r"[A-Za-z]{2,}",
             pathlib.Path(args.train).read_text(encoding="utf-8"))}
    print("evaluation=sampling_comparison")
    print(f"checkpoint={args.checkpoint} seed={args.seed} bytes={args.bytes}")
    for label, temperature, penalty in CONFIGS:
        outputs = []
        hits = tokens = 0
        for prompt in PROMPTS:
            sample = run(args.model, args.checkpoint, prompt, args.bytes,
                         args.seed, temperature, penalty)
            outputs.append(sample)
            generated = sample.decode("utf-8")
            units = re.findall(r"[A-Za-z]{2,}", generated)
            tokens += len(units)
            hits += sum(w.lower() in words for w in units)
            print(f"setting={label} prompt={prompt!r} text={generated!r}")
        print(f"setting_summary={label} unique_outputs={len(set(outputs))}/{len(PROMPTS)} "
              f"vocabulary_hits={hits}/{tokens}")
    print("note=More diversity or vocabulary overlap does not demonstrate coherent meaning.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print(f"sampling comparison failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
