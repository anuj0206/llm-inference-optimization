"""Screen n-gram drafting policies on saved greedy output tokens."""

import argparse
from collections import Counter
from itertools import islice
import json
from pathlib import Path


def match_ends(history, sampled, length):
    suffix = history[-(length - 1):] + [sampled]
    # Match the pinned ngram-simple search bounds. The sampled token is still outside history.
    for start in range(len(history) - length - 1, 0, -1):
        if history[start:start + length] == suffix:
            yield start + length


def fixed_draft(history, sampled, length, limit):
    if len(history) <= length + limit + 1:
        return []
    end = next(match_ends(history, sampled, length), len(history))
    draft = history[end:end + limit]
    return draft if len(draft) >= length else []


def consensus_draft(history, sampled):
    draft = fixed_draft(history, sampled, 6, 8)
    if draft:
        return draft
    for length in (4, 3, 2):
        supporters = [history[end:end + 3] for end in islice(match_ends(history, sampled, length), 16)]
        for index in range(3):
            votes = Counter(candidate[index] for candidate in supporters if index < len(candidate))
            if not votes:
                break
            token, count = votes.most_common(1)[0]
            if count < 3 or count * 4 < len(supporters) * 3:
                break
            draft.append(token)
            supporters = [candidate for candidate in supporters if index < len(candidate) and candidate[index] == token]
        if draft:
            return draft
    return []


def replay(prompt, output, propose):
    drafted = accepted = verification_calls = target_calls = 0
    position = 0
    # Prefill supplies output[0]. Every subsequent target call emits one token plus its verified prefix.
    while position < len(output) - 1:
        remaining = len(output) - position - 2
        draft = propose(prompt + output[:position], output[position])[:remaining]
        prefix = 0
        for candidate, actual in zip(draft, output[position + 1:]):
            if candidate != actual:
                break
            prefix += 1
        drafted += len(draft)
        accepted += prefix
        verification_calls += bool(draft)
        target_calls += 1
        position += 1 + prefix
    return drafted, accepted, verification_calls, target_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="8K n-gram benchmark JSON with saved prompt and output token IDs")
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    if data["status"] != "passed":
        raise SystemExit("Use a completed n-gram benchmark.")

    for case, prompt in data["prompts"].items():
        runs = [(session["mode"], row) for session in data["sessions"]
                for row in session["measurements"] if row["case"] == case]
        output = next(row["tokens"] for mode, row in runs if mode == "none")
        for mode, row in runs:
            if row["tokens"] != output:
                raise SystemExit(f"Greedy outputs differ between runs for {case}.")
        stock = replay(prompt, output, lambda history, sampled: fixed_draft(history, sampled, 6, 8))
        backoff = replay(prompt, output, consensus_draft)
        by_mode = {"ngram-simple": stock, "ngram-backoff": backoff}
        for mode, row in runs:
            if mode in by_mode and by_mode[mode][:3] != (row["drafted"], row["accepted"], row["verifications"]):
                raise SystemExit(f"{mode} replay counters do not match the measured counters for {case}.")
        print(f"\n{case} | policy | drafted | accepted | verification calls | target calls")
        shorter = replay(prompt, output, lambda history, sampled: fixed_draft(history, sampled, 3, 4))
        for label, counts in [("fixed 6/8", stock), ("fixed 3/4", shorter), ("consensus backoff", backoff)]:
            print(" | ".join((label, *(str(value) for value in counts))))


if __name__ == "__main__":
    main()
