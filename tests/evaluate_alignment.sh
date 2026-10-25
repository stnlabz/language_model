#!/bin/sh
set -eu
binary=$1
checkpoint=$2
dir=$3
mkdir -p "$dir"
fail=0
for fixture in train heldout; do
 count=0;pass=0
 while IFS= read -r question; do
  [ -n "$question" ] || continue
  IFS= read -r answer
  expected=${answer#Answer:}
  count=$((count+1))
  printf '%s\n' "$expected" > "$dir/$fixture-$count.expected"
  "$binary" generate "$checkpoint" "$(printf '%s\nAnswer:' "$question")" "${#expected}" 913 greedy > "$dir/$fixture-$count.actual"
  printf '%s %s: ' "$fixture" "$question"
  cat "$dir/$fixture-$count.actual"
  if cmp -s "$dir/$fixture-$count.expected" "$dir/$fixture-$count.actual"; then pass=$((pass+1));else fail=$((fail+1));fi
 done < "tests/fixtures/alignment/$fixture.txt"
 printf '%s exact_match=%s/%s\n' "$fixture" "$pass" "$count"
 done
[ "$fail" -eq 0 ]
