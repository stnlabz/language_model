#!/bin/sh
# STN-LABZ prose-language qualification: independent of Q&A.
# Entirely synthetic corpus. Never accesses Digit or user lesson files.
set -eu
cd "$(dirname "$0")/.."
run=$(mktemp -d ./tests/prose-qualification-XXXXXX)
printf 'Prose qualification artifacts: %s\n' "$run"
cc -std=c11 -O2 -Wall -Wextra -pedantic micro_llm.c -lm -o "$run/micro_llm"
cc -std=c11 -O2 -Wall -Wextra -pedantic tests/context_utilization.c -lm -o "$run/context_utilization"
# Each paragraph is a distinct record. The final two records are held out
# by the general trainer's record-partition logic.
cat > "$run/corpus.txt" <<'EOF'
The blue river flows through the quiet valley. Trees stand along the riverbank. Birds cross the sky above the water.

A careful engineer inspects each part of the machine. The engineer records the result before making a change.

The morning air is cool and clear. A gardener opens the greenhouse and waters the growing plants.

A builder measures the wooden frame before cutting the boards. Accurate measurements prevent unnecessary errors.

A reader opens a book and studies the first chapter. Every paragraph explains a different idea in ordinary language.

The pilot reviews the weather before the aircraft departs. Clear instructions help the crew prepare for the flight.

A teacher asks the students to explain their answers. The students use complete sentences to describe their reasoning.

The mechanic checks the engine and examines each connection. A careful inspection reveals the source of the problem.

The scientist records the experiment and compares the observations. The report explains how the evidence was collected.

The mountain path curves around a small lake. The walking trail continues through the forest toward the village.

A baker prepares the dough before the morning customers arrive. The bread is baked slowly and allowed to cool.

The artist draws a simple sketch and adds detail to the picture. Light falls across the table near the window.

The farmer plants new seeds in the field. Rain nourishes the soil and encourages the crops to grow.

The carpenter selects a strong piece of wood. Each joint must fit before the furniture can be assembled.

The library has shelves filled with history and science books. Visitors read quietly beside the large windows.

The captain checks the map before the journey begins. The ship moves steadily across the calm harbor.

The physician reads the examination notes carefully. Clear records help explain the patient's treatment plan.

The programmer writes a small test for the new function. A passing test demonstrates only the behavior it checks.

The station clock strikes noon as the train arrives. Travelers carry their bags toward the platform.

The wind moves through the tall grass in the field. Clouds gather above the distant hills.
EOF
# More than 1KiB, and held-out records are separate. No Q&A labels anywhere.
if grep -q 'Question:\|Answer:' "$run/corpus.txt"; then
 echo 'FAIL: prose fixture contains Q&A labels' >&2; exit 1
fi
"$run/micro_llm" inspect "$run/corpus.txt" > "$run/inspect.log"
"$run/micro_llm" train "$run/corpus.txt" "$run/model.stn" 200 > "$run/train.log" 2>&1
# Evaluate genuine unfiltered greedy continuation; save raw output.
"$run/micro_llm" generate "$run/model.stn" 'The morning air is' 48 913 greedy > "$run/generation.txt"
"$run/context_utilization" "$run/model.stn" > "$run/context.log"
cat "$run/context.log"
# Record evidence, not fabricated fluency qualification.
cat "$run/inspect.log"
grep 'step=' "$run/train.log"
printf 'Prose continuation: '
cat "$run/generation.txt"
# Basic integrity gate. Coherence and held-out improvement remain distinct.
if ! grep -q 'step=200 ' "$run/train.log"; then
 echo 'FAIL: prose training incomplete' >&2; exit 1
fi
if [ ! -s "$run/generation.txt" ]; then
 echo 'FAIL: no prose continuation' >&2; exit 1
fi
# Detect a concrete autoregressive failure mode, not merely a nonempty file.
# Three or more repetitions of a multi-character phrase indicate collapse.
if grep -Eq '([[:alpha:]]{2,}[[:space:]]+)(\1){2,}' "$run/generation.txt"; then
 echo 'FAIL: degenerate repeated prose continuation' >&2
 exit 1
fi
echo 'Prose execution: PASS; language generalization remains separately unqualified'
