#!/usr/bin/env python3
"""Offline, model-agnostic phoneme-sequence evaluation.

All metrics are calculated from provided reference/prediction sequences.
This does NOT call or evaluate any speech recognition or TTS model by itself.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REQUIRED_COLUMNS = ("item", "locale", "reference_phones", "predicted_phones")


def phones(value: str, ignore_stress: bool = False) -> list[str]:
    if not isinstance(value, str):
        raise TypeError("A phoneme sequence must be text")
    items = unicodedata.normalize("NFC", value).split()
    if ignore_stress:
        items = [re.sub(r"[012]$", "", item) for item in items]
    return items


def edit_operations(reference: list[str], prediction: list[str]) -> dict[str, int]:
    """Unit-cost Levenshtein with deterministic S/D/I traceback.

    Returns minimal edit distance and operation counts. Tie-breaking prefers
    diagonal substitution, then deletion, then insertion.
    """
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = int(reference[i - 1] != prediction[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    edits = {"substitutions": 0, "deletions": 0, "insertions": 0}
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + int(reference[i - 1] != prediction[j - 1]):
            if reference[i - 1] != prediction[j - 1]:
                edits["substitutions"] += 1
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            edits["deletions"] += 1
            i -= 1
        else:
            assert j and dp[i][j] == dp[i][j - 1] + 1
            edits["insertions"] += 1
            j -= 1
    edits["distance"] = dp[n][m]
    assert edits["distance"] == sum(edits[k] for k in ("substitutions", "deletions", "insertions"))
    return edits


def _metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    entries = list(items)
    n = len(entries)
    ref_tokens = sum(e["reference_phone_count"] for e in entries)
    edits = sum(e["distance"] for e in entries)
    matches = sum(e["exact_match"] for e in entries)
    operations = {
        k: sum(e[k] for e in entries)
        for k in ("substitutions", "deletions", "insertions")
    }
    return {
        "items": n,
        "reference_phones": ref_tokens,
        "total_edits": edits,
        **operations,
        "exact_matches": matches,
        "exact_sequence_accuracy": (matches / n) if n else None,
        "phone_error_rate": (edits / ref_tokens) if ref_tokens else None,
    }


def evaluate(rows: Iterable[dict[str, str]], *, ignore_stress: bool = False) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    for line_no, row in enumerate(rows, 2):
        missing = set(REQUIRED_COLUMNS) - row.keys()
        if missing:
            raise ValueError(f"Row {line_no}: missing columns {', '.join(sorted(missing))}")
        item = (row["item"] or "").strip()
        locale = (row["locale"] or "").strip()
        if not item or not locale:
            raise ValueError(f"Row {line_no}: item and locale must be present")
        ref = phones(row["reference_phones"] or "", ignore_stress)
        predicted = phones(row["predicted_phones"] or "", ignore_stress)
        if not ref:
            raise ValueError(f"Row {line_no}: an empty reference cannot define a phone error rate")
        operations = edit_operations(ref, predicted)
        out.append({
            "item": item,
            "locale": locale,
            "reference_phone_count": len(ref),
            "predicted_phone_count": len(predicted),
            **operations,
            "exact_match": int(ref == predicted),
            "reference_phones": ref,
            "predicted_phones": predicted,
        })
    if not out:
        raise ValueError("At least one evaluation record is required")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in out:
        grouped[row["locale"]].append(row)
    return {
        "metric_definition": "Levenshtein unit-cost phone error rate, micro-averaged by reference phone count",
        "phone_tokenization": "Unicode NFC plus whitespace; stress stripped only when requested",
        "ignore_stress": ignore_stress,
        "overall": _metrics(out),
        "by_locale": {k: _metrics(v) for k, v in sorted(grouped.items())},
        "worst_items": sorted(out, key=lambda r: (-r["distance"] / r["reference_phone_count"], r["item"])),
        "model_inference_performed": False,
        "billing_observed": None,
    }


def read_examples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as inp:
        reader = csv.DictReader(inp)
        if not reader.fieldnames or any(col not in reader.fieldnames for col in REQUIRED_COLUMNS):
            raise ValueError("CSV must contain: " + ", ".join(REQUIRED_COLUMNS))
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="One source row per item; whitespace-delimited phone tokens")
    parser.add_argument("--ignore-stress", action="store_true", help="Ignore trailing ARPAbet stress 0, 1, 2")
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout")
    parser.add_argument("--top-errors", type=int, default=10)
    args = parser.parse_args()
    if args.top_errors < 0:
        parser.error("--top-errors must be >= 0")
    result = evaluate(read_examples(args.input_csv), ignore_stress=args.ignore_stress)
    result["worst_items"] = result["worst_items"][:args.top_errors]
    contents = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(contents, encoding="utf-8")
    else:
        print(contents, end="")


if __name__ == "__main__":
    main()
