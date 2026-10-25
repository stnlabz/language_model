#!/bin/sh
# STN-Labz 20261025.1: fixed-step alignment qualification, separate held-out data.
set -eu
cd "$(dirname "$0")/.."
run=$(mktemp -d ./tests/qualification-XXXXXX)
cc -std=c11 -O2 -Wall -Wextra -pedantic micro_llm.c -lm -o "$run/micro_llm"
cc -std=c11 -O2 -Wall -Wextra -pedantic tests/model_math.c -lm -o "$run/model_math"
cc -std=c11 -O2 -Wall -Wextra -pedantic tests/context_alignment.c -lm -o "$run/context_alignment"
"$run/model_math" > "$run/math.log"
"$run/context_alignment" > "$run/context.log"
checkpoint=${1:-$run/model.stn}
if [ "$#" -eq 0 ]; then
 "$run/micro_llm" train tests/fixtures/alignment/train.txt "$checkpoint" 800 tests/fixtures/alignment/heldout.txt > "$run/train.log" 2>&1
fi
# Evaluation records are never optimized. Check all training prompts and all
# held-out prompts, even when an earlier one fails. No sample cherry-picking.
printf 'Qualification artifacts: %s\n' "$run"
sh tests/evaluate_alignment.sh "$run/micro_llm" "$checkpoint" "$run/outputs" > "$run/generation.log" && result=0 || result=$?
cat "$run/generation.log"
exit "$result"
