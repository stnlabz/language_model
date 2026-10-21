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
grep -q '^parameters=' "$d/training.log"
grep -q '^steps=200' "$d/training.log"
"$d/model" eval "$d/prototype.stn" "$d/valid.bin" | grep -q '^validation_loss='
"$d/model" generate "$d/prototype.stn" "Digit " 16 | grep -Eq '^(\\x[0-9a-f]{2}){16}$'
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
echo "Milestone 3 model checks: PASS"
