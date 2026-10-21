#!/usr/bin/env python3
"""Unit tests for loopback HTTP adapter, with no VPS socket or model required."""
import importlib.util
import pathlib
import subprocess
import tempfile
import json
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("http_api", pathlib.Path(__file__).resolve().parents[1] / "http_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class HttpApiTests(unittest.TestCase):
    def test_valid_generation(self):
        fake = subprocess.CompletedProcess([], 0, stdout="\\x68\\x69\\x21\n", stderr="")
        with patch.object(api.subprocess, "run", return_value=fake) as run:
            result = api.generate("./model", "weights.stn", {"prompt": "Hello", "length": 3})
            self.assertEqual(result["text"], "hi!")
            self.assertEqual(result["generated_bytes"], 3)
            self.assertIn("sample", run.call_args.args[0])

    def test_reject_invalid_requests_without_running_model(self):
        invalid = [
            {}, {"prompt": ""}, {"prompt": "x", "length": 257},
            {"prompt": "x", "length": True}, {"prompt": "x", "seed": 0},
            {"prompt": "x", "temperature": 0}, {"prompt": "x", "penalty": 3},
            {"prompt": "x", "extra": 1}, {"prompt": "\x00"},
            {"prompt": "a" * 1025},
        ]
        with patch.object(api.subprocess, "run") as run:
            for value in invalid:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    api.generate("./model", "weights.stn", value)
            run.assert_not_called()

    def test_learning_record_validation(self):
        valid = {"input": "Why is PHP difficult?", "response": "Its syntax is unfamiliar.",
                 "source": "digit", "approved": True}
        self.assertEqual(api.validate_learning(valid), valid)
        for changed in ({"approved": False}, {"source": "other"},
                        {"response": ""}, {"extra": 1}, {"input": "a" * 1025}):
            bad = dict(valid, **changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                api.validate_learning(bad)

    def test_learning_append(self):
        record = api.validate_learning(
            {"input": "Question", "response": "Answer", "source": "digit", "approved": True})
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "queue.jsonl"
            api.append_learning(str(path), record)
            api.append_learning(str(path), record)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), record)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_reject_malformed_output(self):
        fake = subprocess.CompletedProcess([], 0, stdout="\\x00\n", stderr="")
        with patch.object(api.subprocess, "run", return_value=fake):
            with self.assertRaises(RuntimeError):
                api.generate("./model", "weights.stn", {"prompt": "Hello", "length": 3})


if __name__ == "__main__":
    unittest.main()
