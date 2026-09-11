"""Generate the spiral classification dataset for the PyTorch example.

A genuinely non-linear task -- two interleaved spirals -- so a linear
baseline cannot solve it and the network has something real to learn.
Generated deterministically from the version string and written to disk as
JSONL, so the dataset is content-addressed like any other.

``v1`` and ``v2`` differ only in label noise, which makes the dataset axis
of the reproducibility study a controlled perturbation rather than a
different task.
"""

from __future__ import annotations

import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# version -> (points per class, turns, gaussian jitter, label noise)
VERSIONS = {
    "v1": (700, 2.2, 0.16, 0.02),
    "v2": (700, 2.2, 0.16, 0.08),
}


def build(version: str) -> dict[str, list[dict[str, object]]]:
    per_class, turns, jitter, noise = VERSIONS[version]
    rng = random.Random(f"spiral:{version}")
    rows: list[dict[str, object]] = []
    for label in (0, 1):
        for index in range(per_class):
            fraction = index / per_class
            radius = 0.2 + 4.0 * fraction
            angle = turns * math.pi * fraction + label * math.pi
            x = radius * math.cos(angle) + rng.gauss(0.0, jitter)
            y = radius * math.sin(angle) + rng.gauss(0.0, jitter)
            observed = label if rng.random() >= noise else 1 - label
            rows.append({
                "id": f"{label}-{index:04d}",
                "x": round(x, 6),
                "y": round(y, 6),
                "label": observed,
            })
    rng.shuffle(rows)
    split = int(0.75 * len(rows))
    return {"train": rows[:split], "test": rows[split:]}


def write(version: str) -> str:
    directory = os.path.join(HERE, "data", f"spiral-{version}")
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
        "label_noise": VERSIONS[version][3],
        "id_overlap": len(train_ids & test_ids),
        "generator": "examples/torch_mlp/make_data.py",
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
