#!/bin/sh
# Synthetic fixtures only. All artifacts stay inside a fresh workspace folder.
# This is memorization/prompt differentiation, NOT a general-language test.
set -eu
cd "$(dirname "$0")/.."
run=$(mktemp -d ./tests/qualification-XXXXXX)
cc -std=c11 -O2 -Wall -Wextra -pedantic micro_llm.c -lm -o "$run/micro_llm"
cc -std=c11 -O2 -Wall -Wextra -pedantic tests/model_math.c -lm -o "$run/model_math"
"$run/model_math" > "$run/math.log"
i=0
while [ "$i" -lt 100 ]; do
 printf 'Question: What is an engineer?\nAnswer: An engineer designs systems.\n\nQuestion: What is a gardener?\nAnswer: A gardener grows plants.\n\n'
 i=$((i+1))
done > "$run/english.txt"
checkpoint=${1:-$run/model.stn}
if [ "$#" -eq 0 ]; then
 "$run/micro_llm" train "$run/english.txt" "$checkpoint" 4000 > "$run/train.log"
fi
printf ' An engineer designs systems.\n' > "$run/engineer.expected"
printf ' A gardener grows plants.\n' > "$run/gardener.expected"
"$run/micro_llm" generate "$checkpoint" "$(printf 'Question: What is an engineer?\nAnswer:')" 29 913 greedy > "$run/engineer.actual"
"$run/micro_llm" generate "$checkpoint" "$(printf 'Question: What is a gardener?\nAnswer:')" 25 913 greedy > "$run/gardener.actual"
cmp "$run/engineer.expected" "$run/engineer.actual"
cmp "$run/gardener.expected" "$run/gardener.actual"
printf 'PASS: exact trained continuations; artifacts=%s\n' "$run"
