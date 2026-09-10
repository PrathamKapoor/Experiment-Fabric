"""Generate the synthetic retrieval-QA benchmark used by the example.

Run ``python examples/agent_pipeline/make_data.py`` to (re)create the two
benchmark versions under ``data/``.  Generation is fully deterministic
from the version string, so the files can be regenerated anywhere and
will hash identically -- which is what lets Fabric treat them as
content-addressed datasets.

``v1`` and ``v2`` differ in the question mix: v2 has more multi-hop and
more distractor-heavy questions.  That difference is the point.  It makes
"how much does a conclusion move when the benchmark version changes?" a
measurable quantity rather than a worry.
"""

from __future__ import annotations

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# Version -> (number of questions, P(multi-hop), mean distractors, P(trap))
VERSIONS: dict[str, tuple[int, float, float, float]] = {
    "v1": (400, 0.30, 1.8, 0.15),
    "v2": (400, 0.42, 2.6, 0.18),
}

TOPICS = (
    "photosynthesis", "monetary policy", "protein folding", "roman law",
    "tectonic drift", "graph colouring", "antibiotic resistance", "opera",
    "supply chains", "orbital mechanics", "copyright", "immunology",
)


def build(version: str) -> list[dict[str, object]]:
    """Build one benchmark version."""
    count, multi_hop_probability, mean_distractors, trap_probability = VERSIONS[version]
    rng = random.Random(f"agent-benchmark:{version}")
    questions: list[dict[str, object]] = []
    for index in range(count):
        hops = 2 if rng.random() < multi_hop_probability else 1
        distractors = min(6, int(rng.expovariate(1.0 / mean_distractors)))
        questions.append({
            "id": f"q{index:04d}",
            "topic": rng.choice(TOPICS),
            "hops": hops,
            "distractors": distractors,
            "is_repeat": rng.random() < 0.20,
            "is_trap": rng.random() < trap_probability,
            "difficulty": round(rng.random(), 6),
        })
    return questions


def write(version: str) -> str:
    """Write one version to disk and return its directory."""
    directory = os.path.join(HERE, "data", f"benchmark-{version}")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "questions.jsonl")
    questions = build(version)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for question in questions:
            fh.write(json.dumps(question, sort_keys=True) + "\n")
    manifest = {
        "version": version,
        "n_questions": len(questions),
        "multi_hop_fraction": round(
            sum(1 for q in questions if q["hops"] > 1) / len(questions), 4
        ),
        "mean_distractors": round(
            sum(int(q["distractors"]) for q in questions) / len(questions), 4
        ),
        "trap_fraction": round(
            sum(1 for q in questions if q["is_trap"]) / len(questions), 4
        ),
        "generator": "examples/agent_pipeline/make_data.py",
    }
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return directory


def main() -> None:
    for version in VERSIONS:
        directory = write(version)
        print(f"wrote {directory}")


if __name__ == "__main__":
    main()
