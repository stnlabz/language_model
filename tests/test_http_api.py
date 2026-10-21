#!/usr/bin/env python3
"""Unit tests for loopback HTTP adapter, with no VPS socket or model required."""
import importlib.util
import pathlib
import subprocess
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

    def test_reject_malformed_output(self):
        fake = subprocess.CompletedProcess([], 0, stdout="\\x00\n", stderr="")
        with patch.object(api.subprocess, "run", return_value=fake):
            with self.assertRaises(RuntimeError):
                api.generate("./model", "weights.stn", {"prompt": "Hello", "length": 3})


if __name__ == "__main__":
    unittest.main()
