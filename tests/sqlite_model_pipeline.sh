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
