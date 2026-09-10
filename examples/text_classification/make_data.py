"""Generate the sentiment corpus used by the text-classification example.

A small synthetic review corpus with a deliberately imperfect signal:
class-indicative words appear with overlapping frequencies, some reviews
contain negation that flips their polarity, and a fraction of labels are
noisy.  The result is a task where naive Bayes and logistic regression
land at genuinely different, genuinely non-trivial accuracies rather than
both saturating at 100%.

The train/test split is disjoint by construction, and the generator says
so -- so Fabric's contamination check has something true to verify rather
than something to take on faith.
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

POSITIVE = ("excellent", "loved", "brilliant", "gripping", "warm", "clever",
            "delightful", "moving", "sharp", "assured")
NEGATIVE = ("dull", "tedious", "clumsy", "shallow", "muddled", "lifeless",
            "clunky", "predictable", "hollow", "overlong")
NEUTRAL = ("film", "story", "cast", "script", "scene", "ending", "pacing",
           "direction", "score", "runtime", "the", "a", "was", "with", "its")
NEGATORS = ("not", "hardly", "never")

VERSIONS = {
    # version -> (train size, test size, label noise, negation rate)
    "v1": (1200, 400, 0.06, 0.15),
    "v2": (1200, 400, 0.12, 0.25),
}


def make_review(rng: random.Random, label: int, negation_rate: float) -> tuple[str, int]:
    """Build one review and the label it should carry."""
    polarity = label
    words: list[str] = []
    strong = POSITIVE if polarity == 1 else NEGATIVE
    opposite = NEGATIVE if polarity == 1 else POSITIVE

    length = rng.randint(8, 22)
    negated = rng.random() < negation_rate
    for _ in range(length):
        draw = rng.random()
        if draw < 0.18:
            words.append(rng.choice(strong))
        elif draw < 0.26:
            # Words from the other class: the reason this task is not trivial.
            words.append(rng.choice(opposite))
        else:
            words.append(rng.choice(NEUTRAL))

    if negated:
        # "not excellent" reads positive to a unigram model and negative to a
        # bigram one, which is exactly what the bigram ablation measures.
        index = next((i for i, w in enumerate(words) if w in strong), None)
        if index is not None:
            words.insert(index, rng.choice(NEGATORS))
            polarity = 1 - polarity

    if rng.random() < 0.3:
        words[0] = words[0].capitalize()
    return " ".join(words) + rng.choice((".", "!", "")), polarity


def build(version: str) -> dict[str, list[dict[str, object]]]:
    train_size, test_size, noise, negation_rate = VERSIONS[version]
    rng = random.Random(f"sentiment:{version}")
    rows: list[dict[str, object]] = []
    for index in range(train_size + test_size):
        intended = index % 2
        text, label = make_review(rng, intended, negation_rate)
        if rng.random() < noise:
            label = 1 - label
        rows.append({"id": f"r{index:05d}", "text": text, "label": label})
    rng.shuffle(rows)
    return {"train": rows[:train_size], "test": rows[train_size:]}


def write(version: str) -> str:
    directory = os.path.join(HERE, "data", f"sentiment-{version}")
    os.makedirs(directory, exist_ok=True)
    splits = build(version)
    for name, rows in splits.items():
        with open(os.path.join(directory, f"{name}.jsonl"), "w",
                  encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    train_ids = {r["id"] for r in splits["train"]}
    test_ids = {r["id"] for r in splits["test"]}
    manifest = {
        "version": version,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "positive_fraction": round(
            sum(int(r["label"]) for r in splits["train"]) / len(splits["train"]), 4),
        "id_overlap": len(train_ids & test_ids),
        "generator": "examples/text_classification/make_data.py",
    }
    with open(os.path.join(directory, "manifest.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return directory


def main() -> None:
    for version in VERSIONS:
        print(f"wrote {write(version)}")


if __name__ == "__main__":
    main()
