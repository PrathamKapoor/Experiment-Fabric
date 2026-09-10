"""A five-component retrieval-QA agent, built to be ablated.

This is the system under study in the worked example.  It is a
*simulation*: there is no language model behind it, and the numbers it
produces are properties of this code, not predictions about any real
agent.  That is deliberate -- the example has to run in seconds on a
laptop with no GPU and no API key, while still exercising every part of
Fabric on genuinely noisy, genuinely structured data.

What is real: the question set is loaded from content-addressed files on
disk, every component genuinely changes the outcome, seed variance is
real sampling variance over 400 questions, and the accuracy/latency
trade-offs between components are not hand-written constants but
consequences of the model below.

The component contract is the interesting part:

    if ctx.enabled("retrieval"):
        ...

Write the system once with those guards and every ablation in the
specification becomes runnable without touching this file.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from fabric.execution.adapters import RunContext

COMPONENTS = ("retrieval", "reranking", "memory", "planning", "verification")

# Latency in milliseconds attributable to each stage.  These are the only
# hand-set constants in the model; everything else follows from them.
LATENCY_BASE = 18.0
LATENCY_RETRIEVAL_PER_DOC = 6.4
LATENCY_RERANK_PER_DOC = 2.1
LATENCY_MEMORY = 9.0
LATENCY_PLANNING_PER_HOP = 15.5
LATENCY_VERIFICATION = 26.0


@dataclass(frozen=True)
class Question:
    """One benchmark item."""

    id: str
    topic: str
    hops: int
    distractors: int
    is_repeat: bool
    is_trap: bool
    difficulty: float

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Question:
        return Question(
            id=str(data["id"]),
            topic=str(data.get("topic", "")),
            hops=int(data.get("hops", 1)),
            distractors=int(data.get("distractors", 0)),
            is_repeat=bool(data.get("is_repeat", False)),
            is_trap=bool(data.get("is_trap", False)),
            difficulty=float(data.get("difficulty", 0.5)),
        )


def load_questions(dataset: dict[str, Any]) -> list[Question]:
    """Load the benchmark named by the run's dataset record."""
    path = str(dataset.get("local_path") or "")
    if not path:
        raise FileNotFoundError(
            "the dataset has no local_path; point the specification at the "
            "generated benchmark directory (see examples/agent_pipeline/make_data.py)"
        )
    questions_path = os.path.join(path, "questions.jsonl")
    if not os.path.isfile(questions_path):
        raise FileNotFoundError(
            f"{questions_path} does not exist. Run "
            "'python examples/agent_pipeline/make_data.py' to generate it."
        )
    questions = []
    with open(questions_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(Question.from_dict(json.loads(line)))
    return questions


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def answer_logit(
    question: Question,
    *,
    enabled: frozenset[str],
    top_k: int,
    rerank_depth: int,
    memory_size: int,
    temperature: float,
    capability: float,
    stress: float = 0.0,
) -> float:
    """Log-odds that the pipeline answers this question correctly.

    Every term corresponds to a component doing something specific:

    * **retrieval** supplies evidence, and its quality degrades as
      distractors crowd the retrieved set out;
    * **reranking** suppresses distractors, so it only helps when there
      are any;
    * **memory** helps on repeated questions and nothing else;
    * **planning** is what makes multi-hop questions tractable, and its
      absence actively hurts them;
    * **verification** catches traps, and its absence lets them through.

    ``stress`` is the robustness perturbation: harder questions, more
    distractors.
    """
    distractors = question.distractors + (3.0 * stress)
    difficulty = min(1.0, question.difficulty + 0.15 * stress)

    logit = -0.85 + capability

    if "retrieval" in enabled:
        suppression = 0.35 if "reranking" in enabled else 1.0
        effective_noise = distractors * suppression
        depth = max(1, top_k)
        recall = depth / (depth + effective_noise)
        logit += 2.15 * recall
        if "reranking" in enabled and distractors > 0:
            # Reranking pays off in proportion to how much it has to sort.
            logit += 0.55 * min(1.0, rerank_depth / max(1.0, depth * 2.0))
    else:
        # Parametric knowledge only.
        logit -= 0.45

    if question.is_repeat:
        if "memory" in enabled:
            logit += 0.85 * min(1.0, memory_size / 32.0)
        else:
            logit -= 0.15

    if question.hops > 1:
        if "planning" in enabled:
            logit += 1.05
        else:
            logit -= 1.30

    if question.is_trap:
        if "verification" in enabled:
            logit += 0.70
        else:
            logit -= 1.15

    logit -= 1.40 * difficulty
    logit -= 0.80 * max(0.0, temperature - 0.30)
    return logit


def latency_ms(question: Question, *, enabled: frozenset[str], top_k: int,
               rerank_depth: int, jitter: float) -> float:
    """Wall-clock cost of answering one question, in milliseconds."""
    total = LATENCY_BASE
    if "retrieval" in enabled:
        total += LATENCY_RETRIEVAL_PER_DOC * top_k
    if "reranking" in enabled:
        total += LATENCY_RERANK_PER_DOC * rerank_depth
    if "memory" in enabled:
        total += LATENCY_MEMORY
    if "planning" in enabled:
        total += LATENCY_PLANNING_PER_HOP * question.hops
    if "verification" in enabled:
        total += LATENCY_VERIFICATION
    return total * (1.0 + jitter)


def evaluate(
    questions: Sequence[Question],
    ctx: RunContext,
    *,
    enabled: frozenset[str],
    stress: float = 0.0,
    stream: str = "eval",
) -> dict[str, Any]:
    """Run the pipeline over a question set and score it."""
    rng = ctx.rng(stream)
    top_k = int(ctx.get("top_k", 5))
    rerank_depth = int(ctx.get("rerank_depth", 10))
    memory_size = int(ctx.get("memory_size", 32))
    temperature = float(ctx.get("temperature", 0.2))
    threshold = float(ctx.get("verification_threshold", 0.45))
    capability = float((ctx.model.get("config") or {}).get("capability", 0.0))

    correct = 0
    answered = 0
    latencies: list[float] = []
    per_question: list[dict[str, Any]] = []

    for question in questions:
        logit = answer_logit(
            question, enabled=enabled, top_k=top_k, rerank_depth=rerank_depth,
            memory_size=memory_size, temperature=temperature, capability=capability,
            stress=stress,
        )
        probability = _sigmoid(logit)
        is_correct = rng.random() < probability
        # Verification abstains when its own confidence is too low, trading
        # coverage for precision.  Without verification the system always
        # answers.
        confidence = probability if not is_correct else max(probability, 1.0 - probability)
        abstained = "verification" in enabled and confidence < threshold
        latency = latency_ms(question, enabled=enabled, top_k=top_k,
                             rerank_depth=rerank_depth,
                             jitter=rng.uniform(-0.08, 0.08))
        latencies.append(latency)
        if not abstained:
            answered += 1
            if is_correct:
                correct += 1
        per_question.append({
            "id": question.id,
            "probability": round(probability, 6),
            "correct": bool(is_correct and not abstained),
            "abstained": abstained,
            "latency_ms": round(latency, 3),
        })

    total = len(questions)
    accuracy = correct / total if total else 0.0
    precision = correct / answered if answered else 0.0
    recall = accuracy
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "abstention_rate": 1.0 - (answered / total if total else 0.0),
        "latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "predictions": per_question,
    }


def _enabled_components(ctx: RunContext, allowed: Iterable[str] = COMPONENTS) -> frozenset[str]:
    return frozenset(c for c in allowed if ctx.enabled(c))


def _finish(ctx: RunContext, clean: dict[str, Any], stressed: dict[str, Any],
            enabled: frozenset[str]) -> dict[str, float]:
    """Record artifacts and return the metric set."""
    ctx.write_json_artifact("predictions.json", {
        "enabled_components": sorted(enabled),
        "config": ctx.config,
        "clean": clean["predictions"],
        "stressed": stressed["predictions"],
    })
    ctx.log(f"components enabled: {sorted(enabled) or 'none'}")
    ctx.log(f"clean accuracy {clean['accuracy']:.4f}, stressed {stressed['accuracy']:.4f}")
    return {
        "accuracy": clean["accuracy"],
        "f1": clean["f1"],
        "latency_ms": clean["latency_ms"],
        "robustness": stressed["accuracy"],
        "abstention_rate": clean["abstention_rate"],
    }


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def run(ctx: RunContext) -> dict[str, float]:
    """The full proposed system, minus whatever this condition ablates."""
    questions = load_questions(ctx.dataset)
    enabled = _enabled_components(ctx)
    clean = evaluate(questions, ctx, enabled=enabled, stress=0.0, stream="clean")
    stressed = evaluate(questions, ctx, enabled=enabled, stress=1.0, stream="stress")
    return _finish(ctx, clean, stressed, enabled)


def run_reordered(ctx: RunContext) -> dict[str, float]:
    """An alternative implementation of the same method.

    Scores the benchmark in a different order.  Mathematically it computes
    the same quantity; because the random stream is consumed in a
    different sequence, it does not produce bit-identical numbers.  That
    is exactly the kind of "surely irrelevant" implementation difference
    the reproducibility axis exists to quantify.
    """
    questions = sorted(load_questions(ctx.dataset), key=lambda q: (q.topic, q.id))
    enabled = _enabled_components(ctx)
    clean = evaluate(questions, ctx, enabled=enabled, stress=0.0, stream="clean")
    stressed = evaluate(questions, ctx, enabled=enabled, stress=1.0, stream="stress")
    return _finish(ctx, clean, stressed, enabled)


def zero_shot(ctx: RunContext) -> dict[str, float]:
    """Baseline: the model alone, with no agent scaffolding at all."""
    questions = load_questions(ctx.dataset)
    enabled: frozenset[str] = frozenset()
    clean = evaluate(questions, ctx, enabled=enabled, stress=0.0, stream="clean")
    stressed = evaluate(questions, ctx, enabled=enabled, stress=1.0, stream="stress")
    return _finish(ctx, clean, stressed, enabled)


def rag_classic(ctx: RunContext) -> dict[str, float]:
    """Baseline: ordinary retrieval-augmented generation.

    Retrieval and reranking, no memory, planning or verification -- the
    configuration most papers actually compare against.
    """
    questions = load_questions(ctx.dataset)
    enabled = frozenset({"retrieval", "reranking"}) - set(ctx.ablated)
    clean = evaluate(questions, ctx, enabled=enabled, stress=0.0, stream="clean")
    stressed = evaluate(questions, ctx, enabled=enabled, stress=1.0, stream="stress")
    return _finish(ctx, clean, stressed, enabled)


def retrieval_only(ctx: RunContext) -> dict[str, float]:
    """Baseline: retrieval with no reranking."""
    questions = load_questions(ctx.dataset)
    enabled = frozenset({"retrieval"}) - set(ctx.ablated)
    clean = evaluate(questions, ctx, enabled=enabled, stress=0.0, stream="clean")
    stressed = evaluate(questions, ctx, enabled=enabled, stress=1.0, stream="stress")
    return _finish(ctx, clean, stressed, enabled)


def flaky(ctx: RunContext) -> dict[str, float]:
    """A deliberately unreliable arm, used to exercise failure analysis.

    Fails with an out-of-memory error at large ``top_k``.  Nothing in
    Fabric special-cases it: the failure is classified from its message
    exactly as a real OOM would be.
    """
    if int(ctx.get("top_k", 5)) >= 16:
        raise MemoryError("CUDA out of memory: tried to allocate 4.20 GiB")
    return run(ctx)
