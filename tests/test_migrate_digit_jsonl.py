#!/usr/bin/env python3
"""Tests for the standalone, non-training Digit JSONL converter."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

module_path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "migrate_digit_jsonl.py"
spec = importlib.util.spec_from_file_location("migration", module_path)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def record(prompt, response="Answer"):
    return json.dumps({"input": prompt, "response": response, "source": "digit",
                       "approved": True}).encode() + b"\n"


class MigrationTests(unittest.TestCase):
    def test_pair_format_and_whitespace(self):
        data, stats = migration.convert(record("Hello\nthere", "Good\nanswer"))
        self.assertEqual(data, b"Question: Hello there Answer: Good answer\n")
        self.assertEqual(stats["written"], 1)

    def test_structured_records_are_statements_not_dialogue(self):
        source = (record("Digit Corpus record 10, category technical:", "A verified statement.") +
                  record("Digit Observation record 11:", "An observed event.") +
                  record("Digit Context record 12:", "Context for recall.") +
                  record("Digit, explain yourself", "I am Digit."))
        converted, stats = migration.convert(source, "all")
        self.assertEqual(converted,
                         b"A verified statement.\n"
                         b"An observed event.\n"
                         b"Context for recall.\n"
                         b"Question: Digit, explain yourself Answer: I am Digit.\n")
        self.assertEqual(stats["structured_records"], 3)
        self.assertEqual(stats["written"], 4)
        dialogue, dialogue_stats = migration.convert(source, "dialogue")
        self.assertEqual(dialogue,
                         b"Question: Digit, explain yourself Answer: I am Digit.\n")
        self.assertEqual(dialogue_stats["excluded_structured"], 3)

    def test_filter_and_dedup(self):
        data = (record("Digit Context record 1, category utterance.observed:") +
                record("What is PHP?", "A language.") +
                record("What is PHP?", "A language."))
        output, stats = migration.convert(data, "dialogue")
        self.assertEqual(output, b"Question: What is PHP? Answer: A language.\n")
        self.assertEqual(stats["excluded_structured"], 1)
        self.assertEqual(stats["duplicates"], 1)

    def test_reject_invalid_or_partial_input(self):
        for blob in (b"", record("Hi")[:-1], b"not json\n",
                     record("Hi") + b"\n", b"\xff\n"):
            with self.subTest(blob=blob), self.assertRaises(ValueError):
                migration.convert(blob)
        unauthorized = b'{"input":"a","response":"b","source":"digit","approved":false}\n'
        with self.assertRaises(ValueError):
            migration.convert(unauthorized)

    def test_snapshot_unchanged(self):
        original = b"".join(record(f"Question {i}") for i in range(6))
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "snapshot.jsonl"
            source.write_bytes(original)
            output, stats = migration.convert(source.read_bytes())
            destination = pathlib.Path(directory) / "training.txt"
            destination.write_bytes(output)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(stats["written"], 6)
            self.assertEqual(output.count(b"\n"), 6)


if __name__ == "__main__":
    unittest.main()
