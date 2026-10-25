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
# Held-out paraphrases are never included in the synthetic training corpus.
# Report independent outcomes; a memorization pass alone must not qualify
# generalization. Keep all outputs in the test-specific directory.
"$run/micro_llm" generate "$checkpoint" "$(printf 'Question: Explain what an engineer does.\nAnswer:')" 29 913 greedy > "$run/engineer-paraphrase.actual"
"$run/micro_llm" generate "$checkpoint" "$(printf 'Question: Describe the work of a gardener.\nAnswer:')" 25 913 greedy > "$run/gardener-paraphrase.actual"
printf 'Held-out engineer: '; cat "$run/engineer-paraphrase.actual"
printf 'Held-out gardener: '; cat "$run/gardener-paraphrase.actual"
if cmp -s "$run/engineer.expected" "$run/engineer-paraphrase.actual" &&
   cmp -s "$run/gardener.expected" "$run/gardener-paraphrase.actual"; then
 printf 'PASS: held-out paraphrases reproduced expected answers\n'
else
 printf 'BASELINE FAIL: unseen phrasing (two-question training); artifacts=%s\n' "$run" >&2
fi

# Separate controlled generalization experiment: four distinct phrasings
# per concept. Held-out prompts below never appear in the training fixture.
cat > "$run/variants-once.txt" <<'EOF'
Question: What is an engineer?
Answer: An engineer designs systems.

Question: Tell me what an engineer does.
Answer: An engineer designs systems.

Question: What work does an engineer perform?
Answer: An engineer designs systems.

Question: Describe an engineer's job.
Answer: An engineer designs systems.

Question: What is a gardener?
Answer: A gardener grows plants.

Question: Tell me what a gardener does.
Answer: A gardener grows plants.

Question: What work does a gardener perform?
Answer: A gardener grows plants.

Question: Describe a gardener's job.
Answer: A gardener grows plants.

EOF
i=0
while [ "$i" -lt 100 ]; do
 cat "$run/variants-once.txt"
 i=$((i+1))
done > "$run/variants-corpus.txt"
"$run/micro_llm" train "$run/variants-corpus.txt" "$run/variants.stn" 4000 > "$run/variants-train.log"
"$run/micro_llm" generate "$run/variants.stn" "$(printf 'Question: Explain what an engineer does.\nAnswer:')" 29 913 greedy > "$run/variants-engineer.actual"
"$run/micro_llm" generate "$run/variants.stn" "$(printf 'Question: Describe the work of a gardener.\nAnswer:')" 25 913 greedy > "$run/variants-gardener.actual"
printf 'Variant-trained held-out engineer: '; cat "$run/variants-engineer.actual"
printf 'Variant-trained held-out gardener: '; cat "$run/variants-gardener.actual"
if cmp -s "$run/engineer.expected" "$run/variants-engineer.actual" &&
   cmp -s "$run/gardener.expected" "$run/variants-gardener.actual"; then
 printf 'PASS: controlled held-out paraphrases; artifacts=%s\n' "$run"
else
 printf 'FAIL: controlled held-out paraphrases; artifacts=%s\n' "$run" >&2
 exit 1
fi
