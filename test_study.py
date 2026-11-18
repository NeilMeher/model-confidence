import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import study


class StudyTests(unittest.TestCase):
    def test_pair_integrity_and_independent_key(self):
        cases = list(study.items(914, 60, 21, 44))
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({expr for expr, *_ in cases}), 60)
        for expr, choices, answer, wrong in cases:
            self.assertEqual(study.alternate_key(*expr), answer)
            self.assertEqual(len(set(choices)), 4)
            self.assertNotEqual(wrong, answer)
            neutral = study.prompt(expr, choices, None)
            misleading = study.prompt(expr, choices, wrong)
            self.assertEqual(neutral.splitlines()[:2], misleading.splitlines()[:2])

    def test_report(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            study.run(root)
            summary = json.loads((root / "summary.json").read_text())
            with (root / "pairs.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary["pairs"], 60)
            self.assertEqual(len(rows), 120)
            self.assertTrue(all(0.25 <= float(row["confidence"]) <= 1 for row in rows))
            self.assertIn("Reliability", (root / "calibration.svg").read_text())
