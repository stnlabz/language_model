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

# Display actual language from this checkpoint, seeded by the database
# records in the same train/held-out split used by micro_llm evaluate.
# These are qualitative observations, never canned answers or substituted data.
record_count=$(awk 'BEGIN { RS=""; } NF { n++ } END { print n+0 }' "$temp")
training_count=$((record_count * 9 / 10))
if [ "$record_count" -lt 2 ] || [ "$training_count" -lt 1 ]; then
 echo "Cannot select database-derived generation prompts" >&2
 exit 1
fi
echo "=== ACTUAL STN LANGUAGE OUTPUT (unfiltered greedy generation) ==="
echo "checkpoint=$checkpoint database=$database"
for record in 1 "$training_count" "$((training_count + 1))" "$record_count"; do
 prompt=$(awk -v target="$record" 'BEGIN { RS=""; ORS="" } NF { i++; if(i==target){gsub(/[\r\n\t]/," "); print substr($0,1,64); exit} }' "$temp")
 if [ -z "$prompt" ]; then
  echo "FAIL: empty database-derived generation prompt" >&2
  exit 1
 fi
 if [ "$record" -le "$training_count" ]; then part=training; else part=heldout; fi
 printf '\npartition=%s record=%s\n' "$part" "$record"
 printf 'source_prompt: %s\n' "$prompt"
 printf 'model_continuation: '
 ./micro_llm generate "$checkpoint" "$prompt" 64 913 greedy
done
echo "Language output displayed. Generation coherence has NOT been qualified by next-byte metrics."

