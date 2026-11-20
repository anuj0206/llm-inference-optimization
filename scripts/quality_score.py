"""Grade answer content separately from the requested JSON output shape."""

import argparse
import json
from pathlib import Path


def same_value(actual, expected):
    numeric = (isinstance(expected, (int, float)) and not isinstance(expected, bool)
               and isinstance(actual, (int, float)) and not isinstance(actual, bool))
    return actual == expected and (numeric or type(actual) is type(expected))


def score_response(content, expected):
    valid_json = True
    normalization = "none"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        valid_json = False
        try:
            parsed = json.loads("{" + content.strip() + "}")
            normalization = "wrapped bare JSON key/value fragment in braces"
        except json.JSONDecodeError:
            parsed = None
    direct_match = valid_json and same_value(parsed, expected)
    actual = parsed
    if isinstance(actual, dict) and len(actual) == 1 and not isinstance(expected, dict):
        actual = next(iter(actual.values()))
        normalization += "; extracted single object value"
    return {"actual": actual, "correct": same_value(actual, expected),
            "strict_json_valid": valid_json, "requested_value_match": direct_match,
            "normalization": normalization}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    for record in data["records"]:
        record.update(score_response(record["content"], record["expected"]))
    data["scoring"] = "Answer correctness permits extracting a single JSON object value or a bare key/value fragment. requested_value_match is the original strict match; strict_json_valid records syntax separately. Raw outputs are retained."
    args.result.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for cache in ("f16", "q8_0"):
        records = [r for r in data["records"] if r["cache_type"] == cache]
        print(cache, "answers", sum(r["correct"] for r in records), "/", len(records), "strict matches", sum(r["requested_value_match"] for r in records))
