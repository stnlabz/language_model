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


def generate(binary, checkpoint, prompt, count, seed=None, utf8=False):
    command = [binary, "generate", checkpoint, prompt, str(count)]
    if seed is not None or utf8:
        command.append(str(20261020 if seed is None else seed))
    if utf8:
        command.append("utf8")
    result = subprocess.run(
        command,
        capture_output=True, text=True, check=True, timeout=120,
    )
    encoded = result.stdout.strip()
    if not re.fullmatch(r"(?:\\x[0-9a-f]{2}){" + str(count) + r"}", encoded):
        raise ValueError("unexpected generation format")
    return bytes(int(encoded[k + 2:k + 4], 16) for k in range(0, len(encoded), 4))


def probe(binary, checkpoint, prompt):
    result = subprocess.run(
        [binary, "probe", checkpoint, prompt],
        capture_output=True, text=True, check=True, timeout=120,
    )
    values = [float(item) for item in result.stdout.strip().split(",")]
    if len(values) != 256 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid probability vector from probe")
    if abs(sum(values) - 1.0) > 0.0001:
        raise ValueError("probe probabilities not normalized")
    return values


def total_variation(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


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
    # Preserve legacy baseline, then compare prompts under identical seed.
    prompts = ("The engineer ", "Digit ", "A question ")
    seed = 913
    controlled = [generate(args.model, args.checkpoint, prompt, 64, seed=seed)
                  for prompt in prompts]
    distributions = [probe(args.model, args.checkpoint, p) for p in prompts]
    repeat_distribution = probe(args.model, args.checkpoint, prompts[0])
    if distributions[0] != repeat_distribution:
        raise ValueError("probe not reproducible")
    for i in range(len(prompts)):
        top = max(range(256), key=lambda b: distributions[i][b])
        print(f"probe_prompt={prompts[i]!r} top_byte=0x{top:02x} top_probability={distributions[i][top]:.6f}")
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            print(f"prompt_pair={i + 1},{j + 1} total_variation={total_variation(distributions[i], distributions[j]):.9f}")
    print("note=TV=0 means identical next-byte distributions; TV=1 means disjoint distributions.")
    print(f"controlled_seed={seed}")
    print(f"controlled_prompt_unique_outputs={len(set(controlled))}/3")
    print(f"controlled_first_byte_diversity={len({sample[0] for sample in controlled})}/3")
    different_seed = generate(args.model, args.checkpoint, prompts[0], 64, seed=914)
    print(f"seed_changes_output={different_seed != controlled[0]}")
    repeat = generate(args.model, args.checkpoint, prompts[0], 64, seed=seed)
    if repeat != controlled[0]:
        raise ValueError("generation not reproducible with identical seed")
    print("seed_reproducible=yes")
    for prompt in prompts:
        sample = generate(args.model, args.checkpoint, prompt, 64,
                          seed=seed, utf8=True)
        try:
            decoded = sample.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"utf8 mode produced invalid output: {exc}")
        print(f"utf8_prompt={prompt!r} utf8_valid=yes output_bytes={len(sample)}")
        print(f"  utf8_generated={decoded!r}")
    print("note=Legacy generation uses a fixed code-level RNG; controlled comparisons use explicit seeds.")
    print("note=Scores on 369 validation bytes cannot establish language fluency or generalization.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
