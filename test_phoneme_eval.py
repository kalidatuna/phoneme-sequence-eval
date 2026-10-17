import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from phoneme_eval import edit_operations, evaluate, phones, read_examples


class PhoneEditTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(edit_operations(["AA", "B"], ["AA", "B"])["distance"], 0)

    def test_one_substitution(self):
        self.assertEqual(edit_operations(["AA"], ["AE"])["substitutions"], 1)

    def test_one_deletion(self):
        self.assertEqual(edit_operations(["AA", "B"], ["AA"])["deletions"], 1)

    def test_one_insertion(self):
        self.assertEqual(edit_operations(["AA"], ["AA", "B"])["insertions"], 1)

    def test_empty_prediction(self):
        self.assertEqual(edit_operations(["A", "B"], [])["distance"], 2)

    def test_empty_reference_internal(self):
        self.assertEqual(edit_operations([], ["A"])["insertions"], 1)

    def test_both_empty_internal(self):
        self.assertEqual(edit_operations([], [])["distance"], 0)

    def test_multi_edit_minimum(self):
        self.assertEqual(edit_operations(["A", "B", "C"], ["B", "C", "D"])["distance"], 2)

    def test_edit_sum_invariant(self):
        for ref, hyp in [("A B C", "A B C D"), ("A B C D", "B D"), ("A B", "X Y")]:
            op = edit_operations(ref.split(), hyp.split())
            self.assertEqual(op["distance"], sum(op[k] for k in ("substitutions", "deletions", "insertions")))

    def test_stress_aware_by_default(self):
        self.assertEqual(phones("AH0 T EH1"), ["AH0", "T", "EH1"])

    def test_stress_ignored(self):
        self.assertEqual(phones("AH0 T EH1", True), ["AH", "T", "EH"])

    def test_unicode_normalization(self):
        self.assertEqual(phones("e\u0301"), ["é"])


class EvaluationTests(unittest.TestCase):
    DATA = [
        {"item": "a", "locale": "en", "reference_phones": "AH0 B", "predicted_phones": "AH0 B"},
        {"item": "b", "locale": "en", "reference_phones": "D IY1", "predicted_phones": "D EH1"},
        {"item": "c", "locale": "fr", "reference_phones": "A B C", "predicted_phones": "A B"},
    ]

    def test_counts(self):
        x = evaluate(self.DATA)["overall"]
        self.assertEqual((x["items"], x["reference_phones"], x["total_edits"]), (3, 7, 2))

    def test_phone_error_rate(self):
        self.assertAlmostEqual(evaluate(self.DATA)["overall"]["phone_error_rate"], 2 / 7)

    def test_exact_accuracy(self):
        self.assertAlmostEqual(evaluate(self.DATA)["overall"]["exact_sequence_accuracy"], 1 / 3)

    def test_locale_breakdown(self):
        x = evaluate(self.DATA)["by_locale"]
        self.assertEqual(x["fr"]["deletions"], 1)
        self.assertEqual(x["en"]["substitutions"], 1)

    def test_stress_switch_really_changes_results(self):
        data = [{"item": "s", "locale": "en", "reference_phones": "AH0", "predicted_phones": "AH1"}]
        self.assertEqual(evaluate(data)["overall"]["total_edits"], 1)
        self.assertEqual(evaluate(data, ignore_stress=True)["overall"]["total_edits"], 0)

    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            evaluate([{"item": "bad"}])

    def test_missing_locale(self):
        bad = dict(self.DATA[0], locale="")
        with self.assertRaisesRegex(ValueError, "item and locale"):
            evaluate([bad])

    def test_empty_reference(self):
        bad = dict(self.DATA[0], reference_phones="")
        with self.assertRaisesRegex(ValueError, "empty reference"):
            evaluate([bad])

    def test_zero_rows(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate([])

    def test_empty_prediction_per_is_one(self):
        bad = dict(self.DATA[0], predicted_phones="")
        self.assertEqual(evaluate([bad])["overall"]["phone_error_rate"], 1)

    def test_non_model_flag(self):
        self.assertIs(evaluate(self.DATA)["model_inference_performed"], False)
        self.assertIsNone(evaluate(self.DATA)["billing_observed"])

    def test_deterministic_tied_errors(self):
        a = evaluate(self.DATA)
        b = evaluate(self.DATA)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_read_and_cli(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.csv"
            with inp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.DATA[0]))
                writer.writeheader()
                writer.writerows(self.DATA)
            self.assertEqual(len(read_examples(inp)), 3)
            prg = Path(__file__).parent / "phoneme_eval.py"
            result = subprocess.run(
                [sys.executable, str(prg), str(inp), "--top-errors", "2"],
                capture_output=True, text=True, check=True,
            )
            data = json.loads(result.stdout)
            self.assertEqual(data["overall"]["items"], 3)
            self.assertEqual(len(data["worst_items"]), 2)

    def test_read_rejects_wrong_columns(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            path.write_text("a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError, "CSV must contain"):
                read_examples(path)


if __name__ == "__main__":
    unittest.main()
