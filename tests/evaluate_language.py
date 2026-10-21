#!/usr/bin/env python3
"""STN-Labz Milestone 4 read-only language evaluation.

AI-assisted. Does not train, overwrite checkpoints, or access Digit/RictusD.
Requires Python 3 standard library and the existing compiled ./model.
"""
import argparse
import collections
import math
import pathlib
import re
import subprocess
import sys
import time


def read_data(path):
    data = pathlib.Path(path).read_bytes()
    if len(data) < 2:
        raise ValueError(f"{path}: fewer than two bytes")
    return data


def score_model(binary, checkpoint, validation):
    result = subprocess.run(
        [binary, "eval", checkpoint, validation],
        capture_output=True, text=True, check=True, timeout=120,
    )
    match = re.search(r"^validation_loss=([0-9]+(?:\.[0-9]+)?)$", result.stdout, re.M)
    if match is None:
        raise ValueError("model returned no validation loss")
    return float(match.group(1))


def generate(binary, checkpoint, prompt, count):
    result = subprocess.run(
        [binary, "generate", checkpoint, prompt, str(count)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    encoded = result.stdout.strip()
    if not re.fullmatch(r"(?:\\x[0-9a-f]{2}){" + str(count) + r"}", encoded):
        raise ValueError("unexpected generation format")
    return bytes(int(encoded[k + 2:k + 4], 16) for k in range(0, len(encoded), 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model")
    parser.add_argument("--checkpoint", default="prototype.stn")
    parser.add_argument("--train", default="train.bin")
    parser.add_argument("--validation", default="validation.bin")
    args = parser.parse_args()
    train = read_data(args.train)
    validation = read_data(args.validation)
    if pathlib.Path(args.checkpoint).stat().st_size == 0:
        raise ValueError("empty checkpoint")

    # Frequency-only training baseline with Laplace smoothing.
    counts = collections.Counter(train[:-1])
    denom = len(train) - 1 + 256
    baseline = -sum(math.log((counts[b] + 1) / denom) for b in validation[1:]) / (len(validation) - 1)
    uniform = math.log(256)
    start = time.monotonic()
    trained = score_model(args.model, args.checkpoint, args.validation)
    elapsed = time.monotonic() - start

    print("evaluation=milestone4")
    print(f"train_bytes={len(train)} validation_bytes={len(validation)}")
    print(f"uniform_loss={uniform:.6f}")
    print(f"unigram_train_baseline_loss={baseline:.6f}")
    print(f"trained_model_validation_loss={trained:.6f}")
    print(f"model_beats_unigram={'yes' if trained < baseline else 'no'}")
    print(f"model_eval_seconds={elapsed:.3f}")

    for prompt in ("The engineer ", "Digit ", "A question "):
        generated = generate(args.model, args.checkpoint, prompt, 64)
        valid = True
        try:
            text = generated.decode("utf-8")
        except UnicodeDecodeError:
            valid = False
            text = generated.decode("utf-8", "replace")
        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in generated)
        diversity = len(set(generated))
        repeats = max((len(list(g)) for _, g in __import__("itertools").groupby(generated)), default=0)
        print(f"prompt={prompt!r} utf8_valid={valid} printable_ascii={printable}/64 distinct_bytes={diversity}/64 longest_byte_run={repeats}")
        print(f"  generated={text!r}")
    print("note=Generation uses fixed code-level RNG and may not substantially vary by prompt.")
    print("note=Scores on 369 validation bytes cannot establish language fluency or generalization.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
