#!/bin/sh
# Run export, train (optional), and read-only evaluation on one DB snapshot.
set -eu
if [ "$#" -ne 4 ]; then
 echo "usage: sqlite_model_pipeline.sh DB CHECKPOINT STEPS train|evaluate" >&2
 exit 2
fi
database=$1
checkpoint=$2
steps=$3
mode=$4
temp=$(mktemp /tmp/stn-sqlite-corpus-XXXXXX)
trap 'rm -f "$temp"' EXIT HUP INT TERM
./sqlite_corpus "$database" > "$temp"
if [ "$mode" = train ]; then
 ./micro_llm train "$temp" "$checkpoint" "$steps"
elif [ "$mode" != evaluate ]; then
 echo "unsupported pipeline mode" >&2
 exit 2
fi
./micro_llm evaluate "$temp" "$checkpoint"

# Observe native model output, without supplying any canned answer.
# The source prompt is extracted from the SQLite export, not a fixture.
# A failed degeneration check is a language-test failure, never PASS.
prompt=$(LC_ALL=C awk 'BEGIN { RS=""; ORS="" } NF { gsub(/[\r\n\t]/," "); print substr($0,1,80); exit }' "$temp")
if [ -z "$prompt" ]; then
 echo "FAIL: no database-derived prompt available" >&2
 exit 1
fi
sample=$(mktemp /tmp/stn-language-output-XXXXXX)
trap 'rm -f "$temp" "$sample"' EXIT HUP INT TERM
echo "=== ACTUAL CHECKPOINT LANGUAGE GENERATION ==="
printf 'checkpoint: %s\nsource_prompt: %s\n' "$checkpoint" "$prompt"
# 256 is the existing native CLI's maximum output per invocation; this
# bounded observation is not a restriction on the model or its corpus.
./micro_llm generate "$checkpoint" "$prompt" 256 913 greedy > "$sample"
printf 'model_continuation:\n'
cat "$sample"
printf '\n'
# Reject the observed repeating-character failure, not claim that a
# structural check proves grammar, comprehension, or language fluency.
if ! LC_ALL=C awk '
 { for (i=1;i<=length($0);i++) {
    c=substr($0,i,1)
    if(c==prev)run++;else run=1
    if(run>=12)degenerate=1
    if(c==" ")spaces++
    prev=c
   }
 }
 END { if(degenerate || spaces==0)exit 1 }
' "$sample"; then
 echo "FAIL: actual model output is degenerate (long character repetition or no word separators)" >&2
 exit 1
fi
echo "Generation nondegeneracy check: PASS; language competency still requires inspection."
