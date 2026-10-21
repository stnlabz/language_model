#!/bin/sh
# STN-Labz Milestone 3 test: source build and bounded experiment.
# AI-assisted; execute in the separate language_model repository only.
set -eu
cd "$(dirname "$0")/.."
d="$(mktemp -d)"
trap 'rm -rf "$d"' EXIT HUP INT TERM
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic model.c -lm -o "$d/model"
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic prepare.c -o "$d/prepare"
"$d/prepare" prepare samples/synthetic_language.txt "$d/train.bin" "$d/valid.bin"
"$d/model" train "$d/train.bin" "$d/valid.bin" 200 "$d/prototype.stn" > "$d/training.log"
test -s "$d/prototype.stn"
# Re-running identical sequence training must reproduce metrics and weights.
"$d/model" train "$d/train.bin" "$d/valid.bin" 200 "$d/repeat.stn" > "$d/repeat.log"
cmp "$d/prototype.stn" "$d/repeat.stn"
cmp "$d/training.log" "$d/repeat.log"
# A learnable dataset should reduce the initial held-out loss.
awk '
    /^parameters=/ {
        split($2, part, "="); initial = part[2]
    }
    /^steps=/ {
        split($3, part, "="); final = part[2]
    }
    END {
        if (initial == "" || final == "" || !(final < initial)) exit 1
    }
' "$d/training.log"
grep -q '^parameters=' "$d/training.log"
grep -q '^steps=200' "$d/training.log"
# Regression gate against the reported broken 1,000-step BPTT result
# (validation loss 5.541507). A passing result is not language fluency.
"$d/model" train "$d/train.bin" "$d/valid.bin" 1000 "$d/scale.stn" > "$d/scale.log"
awk '
    /^steps=1000 / {
        split($3, part, "=")
        if (!(part[2] + 0 < 5.541507)) exit 1
        found = 1
    }
    END { if (!found) exit 1 }
' "$d/scale.log"
"$d/model" eval "$d/prototype.stn" "$d/valid.bin" | grep -q '^validation_loss='
"$d/model" generate "$d/prototype.stn" "Digit " 16 > "$d/generated.txt"
test "$(wc -c < "$d/generated.txt" | tr -d ' ')" -eq 65
grep -Fq '\x' "$d/generated.txt"
if "$d/model" train "$d/train.bin" "$d/valid.bin" 200 "$d/prototype.stn" >/dev/null 2>&1; then
    echo "FAIL: existing checkpoint overwritten" >&2; exit 1
fi
cp "$d/prototype.stn" "$d/corrupt.stn"
printf '\000' | dd of="$d/corrupt.stn" bs=1 seek=50 count=1 conv=notrunc 2>/dev/null
if "$d/model" eval "$d/corrupt.stn" "$d/valid.bin" >/dev/null 2>&1; then
    echo "FAIL: corrupt checkpoint accepted" >&2; exit 1
fi
if "$d/model" generate "$d/prototype.stn" "x" 257 >/dev/null 2>&1; then
    echo "FAIL: unbounded generation accepted" >&2; exit 1
fi
# Stateful training: identical weights for uninterrupted and resumed
# 128 + 128 positions, using the same 64-position update boundaries.
"$d/model" train-state "$d/train.bin" "$d/valid.bin" 256 "$d/full_state.stn" > "$d/full_state.log"
"$d/model" train-state "$d/train.bin" "$d/valid.bin" 128 "$d/first_state.stn" > "$d/first_state.log"
"$d/model" resume "$d/first_state.stn" "$d/train.bin" "$d/valid.bin" 128 "$d/resumed_state.stn" > "$d/resumed_state.log"
cmp "$d/full_state.stn" "$d/resumed_state.stn"
"$d/model" eval "$d/resumed_state.stn" "$d/valid.bin" | grep -q '^validation_loss='
grep -q 'total_positions=256' "$d/resumed_state.log"
if "$d/model" resume "$d/prototype.stn" "$d/train.bin" "$d/valid.bin" 64 "$d/legacy_wrong.stn" >/dev/null 2>&1; then
    echo "FAIL: legacy checkpoint incorrectly accepted as resumable" >&2; exit 1
fi
if "$d/model" resume "$d/first_state.stn" "$d/valid.bin" "$d/train.bin" 64 "$d/wrong_data.stn" >/dev/null 2>&1; then
    echo "FAIL: changed training data incorrectly accepted" >&2; exit 1
fi
if "$d/model" resume "$d/first_state.stn" "$d/train.bin" "$d/valid.bin" 63 "$d/wrong_count.stn" >/dev/null 2>&1; then
    echo "FAIL: unaligned training steps accepted" >&2; exit 1
fi
if "$d/model" resume "$d/first_state.stn" "$d/train.bin" "$d/valid.bin" 64 "$d/full_state.stn" >/dev/null 2>&1; then
    echo "FAIL: existing state checkpoint overwritten" >&2; exit 1
fi
# Generation-only controls must be deterministic, bounded and UTF-8 safe.
"$d/model" sample "$d/prototype.stn" "The engineer " 64 913 0.8 0.2 utf8 > "$d/sample_a.txt"
"$d/model" sample "$d/prototype.stn" "The engineer " 64 913 0.8 0.2 utf8 > "$d/sample_b.txt"
cmp "$d/sample_a.txt" "$d/sample_b.txt"
test "$(wc -c < "$d/sample_a.txt" | tr -d ' ')" -eq 257
if "$d/model" sample "$d/prototype.stn" "x" 64 913 0 0 utf8 >/dev/null 2>&1; then
    echo "FAIL: invalid temperature accepted" >&2; exit 1
fi
if "$d/model" sample "$d/prototype.stn" "x" 64 913 0.8 3 utf8 >/dev/null 2>&1; then
    echo "FAIL: invalid penalty accepted" >&2; exit 1
fi
echo "Milestone 3 model checks: PASS"
