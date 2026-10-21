#!/usr/bin/env python3
"""Unit checks for read-only generation diagnostics; no model or VPS required."""
import importlib.util
import pathlib
import unittest

module_path = pathlib.Path(__file__).with_name("evaluate_quality.py")
spec = importlib.util.spec_from_file_location("evaluate_quality", module_path)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


class QualityTests(unittest.TestCase):
    def test_valid_encoded_output(self):
        self.assertEqual(quality.decode_generated("\\x41\\x20\\x42\n", 3), b"A B")

    def test_reject_malformed_or_short_output(self):
        for raw in ("\\x41\\x4g", "\\x41", "\\x41extra", ""):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    quality.decode_generated(raw, 2)

    def test_lexical_counts(self):
        stats = quality.analyze(b"Clear report.\nBad\x01", {"clear", "report"})
        self.assertEqual(stats["words"], 3)
        self.assertEqual(stats["vocab_hits"], 2)
        self.assertEqual(stats["controls"], 1)
        self.assertEqual(stats["sentence_marks"], 1)

    def test_empty_text(self):
        self.assertEqual(quality.analyze(b" \n", {"test"})["vocab_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
