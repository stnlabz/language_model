#!/bin/sh
# STN-Labz language model Milestone 2 smoke and negative tests.
# AI-assisted; no production VPS access or agent changes.
set -eu
cd "$(dirname "$0")/.."
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT HUP INT TERM
cc -std=c11 -Wall -Wextra -Werror -pedantic -O2 prepare.c -o "$dir/prepare"
"$dir/prepare" prepare samples/synthetic_language.txt "$dir/train.bin" "$dir/valid.bin" > "$dir/record.txt"
test -s "$dir/train.bin"
test -s "$dir/valid.bin"
grep -q 'vocabulary=256' "$dir/record.txt"
"$dir/prepare" prepare samples/synthetic_language.txt "$dir/train2.bin" "$dir/valid2.bin" > "$dir/record2.txt"
cmp "$dir/train.bin" "$dir/train2.bin"
cmp "$dir/valid.bin" "$dir/valid2.bin"
cmp "$dir/record.txt" "$dir/record2.txt"
if "$dir/prepare" prepare samples/synthetic_language.txt "$dir/train.bin" "$dir/new.bin" >/dev/null 2>&1; then
    echo "ERROR: overwrite was allowed" >&2
    exit 1
fi
test ! -e "$dir/new.bin"
printf '\377\n' > "$dir/invalid.txt"
if "$dir/prepare" prepare "$dir/invalid.txt" "$dir/badtrain.bin" "$dir/badvalid.bin" >/dev/null 2>&1; then
    echo "ERROR: invalid UTF-8 was accepted" >&2
    exit 1
fi
test ! -e "$dir/badtrain.bin"
test ! -e "$dir/badvalid.bin"
: > "$dir/empty.txt"
if "$dir/prepare" prepare "$dir/empty.txt" "$dir/emptytrain.bin" "$dir/emptyvalid.bin" >/dev/null 2>&1; then
    echo "ERROR: empty input was accepted" >&2
    exit 1
fi
echo "Milestone 2 tokenizer/data-pipeline checks: PASS"
