#!/usr/bin/env python3
"""Read-only, deterministic sample-quality diagnostics for STN-Labz byte GRU.

These scores measure format and lexical resemblance, not comprehension, factual
accuracy, instruction following, or agent readiness. No corpus modifications.
"""
import argparse
import collections
import pathlib
import re
import subprocess
import sys

PROMPTS = (
    "The engineer ",
    "Digit ",
    "A question ",
    "The controller ",
    "A report ",
)
WORD = re.compile(r"[A-Za-z]+")
ENCODED = re.compile(r"(?:\\x[0-9a-f]{2})+")


def decode_generated(raw, count):
    encoded = raw.strip()
    if not ENCODED.fullmatch(encoded) or len(encoded) != 4 * count:
        raise ValueError("invalid encoded generation length or format")
    return bytes.fromhex(encoded.replace("\\x", ""))


def sample(binary, checkpoint, prompt, seed, count):
    output = subprocess.run(
        [binary, "generate", checkpoint, prompt, str(count), str(seed), "utf8"],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return decode_generated(output.stdout, count)


def vocabulary(train):
    text = pathlib.Path(train).read_text(encoding="utf-8")
    return {word.lower() for word in WORD.findall(text) if len(word) >= 2}


def analyze(payload, vocab):
    text = payload.decode("utf-8")
    tokens = [token.lower() for token in WORD.findall(text)]
    eligible = [token for token in tokens if len(token) >= 2]
    matched = sum(token in vocab for token in eligible)
    invalid_controls = sum(byte < 32 and byte not in (9, 10, 13) or byte == 127
                           for byte in payload)
    punctuation = sum(char in ".!?" for char in text)
    return {
        "words": len(eligible),
        "vocab_hits": matched,
        "vocab_ratio": matched / len(eligible) if eligible else 0.0,
        "controls": invalid_controls,
        "sentence_marks": punctuation,
        "printable_ascii": sum(byte in (9, 10, 13) or 32 <= byte <= 126
                               for byte in payload),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--bytes", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.bytes <= 256:
        parser.error("--bytes must be 1..256")
    if not 1 <= args.seed <= 2**64 - 1:
        parser.error("--seed must be a nonzero unsigned 64-bit integer")
    vocab = vocabulary(args.train)
    if not vocab:
        raise ValueError("empty word vocabulary")
    print("evaluation=language_quality_diagnostic")
    print(f"checkpoint={args.checkpoint} prompts={len(PROMPTS)} seed={args.seed}")
    print(f"reference_vocabulary_size={len(vocab)}")
    samples = []
    totals = collections.Counter()
    for index, prompt in enumerate(PROMPTS, 1):
        payload = sample(args.model, args.checkpoint, prompt, args.seed, args.bytes)
        again = sample(args.model, args.checkpoint, prompt, args.seed, args.bytes)
        if payload != again:
            raise ValueError("nonreproducible sample")
        stats = analyze(payload, vocab)
        totals.update({key: value for key, value in stats.items() if key != "vocab_ratio"})
        samples.append(payload)
        print(f"prompt_index={index} prompt={prompt!r} " +
              " ".join(f"{key}={value:.3f}" if isinstance(value, float)
                       else f"{key}={value}" for key, value in stats.items()))
        print(f"  text={payload.decode('utf-8')!r}")
    print(f"unique_outputs={len(set(samples))}/{len(samples)}")
    print(f"total_vocabulary_hits={totals['vocab_hits']}/{totals['words']}")
    print(f"invalid_controls={totals['controls']}")
    print("note=Vocabulary hits measure spelling resemblance, not semantic correctness.")
    print("note=No automatic fluency or comprehension pass/fail is asserted.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print(f"quality evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
