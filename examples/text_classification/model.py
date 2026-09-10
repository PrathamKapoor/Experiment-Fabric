"""Real algorithms on real data, with no Fabric import anywhere.

The agent example demonstrates ablation on a simulated system.  This one
demonstrates the other half of the claim: that Fabric's core understands
*experiments*, not neural networks, and that the code under study need
not know Fabric exists.

Nothing in this file imports ``fabric``.  The entrypoints take a plain
object with ``config``, ``dataset``, ``seed`` and ``ablated`` attributes,
which is exactly what a ``RunContext`` provides -- so the same file can be
called by Fabric, by a notebook, or by a plain script.

The classifiers are genuine implementations (multinomial naive Bayes and
logistic regression trained by gradient descent) over a real bag-of-words
feature space, so the accuracies, the seed variance, and the effect of
the learning rate are all real measurements of real algorithms.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Sequence
from typing import Any, Protocol


class Context(Protocol):
    """The narrow interface this file needs from whatever runs it."""

    seed: int
    config: dict[str, Any]
    dataset: dict[str, Any]
    ablated: tuple[str, ...]

    def get(self, key: str, default: Any = None) -> Any: ...
    def rng(self, stream: str = "") -> random.Random: ...
    def enabled(self, component: str) -> bool: ...
    def log(self, message: str) -> None: ...
    def write_json_artifact(self, name: str, value: Any, *, role: str = "output") -> str: ...


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_split(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Read one split of the corpus named by the run's dataset record."""
    path = str(dataset.get("local_path") or "")
    if not path:
        raise FileNotFoundError(
            "the dataset has no local_path; generate it with "
            "'python examples/text_classification/make_data.py'"
        )
    file_path = os.path.join(path, f"{split}.jsonl")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"{file_path} does not exist. Run "
            "'python examples/text_classification/make_data.py' first."
        )
    rows = []
    with open(file_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenise(text: str, *, lowercase: bool, bigrams: bool) -> list[str]:
    """Split text into features.

    ``lowercase`` and ``bigrams`` are the two ablatable preprocessing
    components: both are things practitioners assume help, and this
    experiment measures whether they do.
    """
    cleaned = text.lower() if lowercase else text
    words = [w.strip(".,!?;:'\"()") for w in cleaned.split()]
    words = [w for w in words if w]
    if not bigrams:
        return words
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def vectorise(rows: Sequence[dict[str, Any]], *, lowercase: bool,
              bigrams: bool) -> list[tuple[dict[str, int], int]]:
    documents = []
    for row in rows:
        counts: dict[str, int] = {}
        for token in tokenise(str(row["text"]), lowercase=lowercase, bigrams=bigrams):
            counts[token] = counts.get(token, 0) + 1
        documents.append((counts, int(row["label"])))
    return documents


# --------------------------------------------------------------------------
# classifiers
# --------------------------------------------------------------------------


class NaiveBayes:
    """Multinomial naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.log_prior: dict[int, float] = {}
        self.log_likelihood: dict[int, dict[str, float]] = {}
        self.vocabulary: set[str] = set()
        self.class_totals: dict[int, float] = {}

    def fit(self, documents: Sequence[tuple[dict[str, int], int]]) -> NaiveBayes:
        counts: dict[int, dict[str, int]] = {}
        class_counts: dict[int, int] = {}
        for features, label in documents:
            class_counts[label] = class_counts.get(label, 0) + 1
            bucket = counts.setdefault(label, {})
            for token, value in features.items():
                bucket[token] = bucket.get(token, 0) + value
                self.vocabulary.add(token)
        total_documents = len(documents) or 1
        vocabulary_size = len(self.vocabulary) or 1
        for label, bucket in counts.items():
            total = sum(bucket.values())
            self.class_totals[label] = total
            denominator = total + self.alpha * vocabulary_size
            self.log_prior[label] = math.log(class_counts[label] / total_documents)
            self.log_likelihood[label] = {
                token: math.log((value + self.alpha) / denominator)
                for token, value in bucket.items()
            }
        return self

    def _default(self, label: int) -> float:
        denominator = self.class_totals.get(label, 0.0) + self.alpha * (len(self.vocabulary) or 1)
        return math.log(self.alpha / denominator) if denominator else 0.0

    def predict(self, features: dict[str, int]) -> int:
        best_label, best_score = 0, -math.inf
        for label, prior in self.log_prior.items():
            score = prior
            likelihood = self.log_likelihood[label]
            fallback = self._default(label)
            for token, value in features.items():
                if token in self.vocabulary:
                    score += value * likelihood.get(token, fallback)
            if score > best_score:
                best_label, best_score = label, score
        return best_label


class LogisticRegression:
    """Binary logistic regression trained by minibatch gradient descent."""

    def __init__(self, *, learning_rate: float = 0.1, epochs: int = 8,
                 batch_size: int = 16, l2: float = 0.0, seed: int = 0) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.weights: dict[str, float] = {}
        self.bias = 0.0
        self.final_loss = math.nan

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z < -60:
            return 0.0
        if z > 60:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def _score(self, features: dict[str, int]) -> float:
        total = self.bias
        for token, value in features.items():
            weight = self.weights.get(token)
            if weight:
                total += weight * value
        return total

    def fit(self, documents: Sequence[tuple[dict[str, int], int]]) -> LogisticRegression:
        rng = random.Random(self.seed)
        order = list(range(len(documents)))
        for _epoch in range(self.epochs):
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = [documents[i] for i in order[start:start + self.batch_size]]
                gradients: dict[str, float] = {}
                bias_gradient = 0.0
                for features, label in batch:
                    error = self._sigmoid(self._score(features)) - label
                    bias_gradient += error
                    for token, value in features.items():
                        gradients[token] = gradients.get(token, 0.0) + error * value
                scale = self.learning_rate / max(1, len(batch))
                self.bias -= scale * bias_gradient
                for token, gradient in gradients.items():
                    current = self.weights.get(token, 0.0)
                    self.weights[token] = current - scale * (gradient + self.l2 * current)
        self.final_loss = self._loss(documents)
        return self

    def _loss(self, documents: Sequence[tuple[dict[str, int], int]]) -> float:
        if not documents:
            return math.nan
        total = 0.0
        for features, label in documents:
            probability = min(max(self._sigmoid(self._score(features)), 1e-12), 1 - 1e-12)
            total -= label * math.log(probability) + (1 - label) * math.log(1 - probability)
        return total / len(documents)

    def predict(self, features: dict[str, int]) -> int:
        return 1 if self._score(features) >= 0.0 else 0


class MajorityClass:
    """The baseline every classifier has to beat to mean anything."""

    def __init__(self) -> None:
        self.label = 0

    def fit(self, documents: Sequence[tuple[dict[str, int], int]]) -> MajorityClass:
        counts: dict[int, int] = {}
        for _features, label in documents:
            counts[label] = counts.get(label, 0) + 1
        self.label = max(counts, key=lambda k: counts[k]) if counts else 0
        return self

    def predict(self, features: dict[str, int]) -> int:
        return self.label


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def score(model: Any, documents: Sequence[tuple[dict[str, int], int]]) -> dict[str, float]:
    """Accuracy, precision, recall and F1 on a labelled split."""
    true_positive = false_positive = false_negative = correct = 0
    predictions = []
    for features, label in documents:
        predicted = model.predict(features)
        predictions.append(predicted)
        if predicted == label:
            correct += 1
        if predicted == 1 and label == 1:
            true_positive += 1
        elif predicted == 1 and label == 0:
            false_positive += 1
        elif predicted == 0 and label == 1:
            false_negative += 1
    total = len(documents) or 1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "_predictions": predictions,
    }


def _prepare(ctx: Context) -> tuple[list, list, bool, bool]:
    lowercase = ctx.enabled("lowercasing")
    bigrams = ctx.enabled("bigrams")
    train = vectorise(load_split(ctx.dataset, "train"), lowercase=lowercase, bigrams=bigrams)
    test = vectorise(load_split(ctx.dataset, "test"), lowercase=lowercase, bigrams=bigrams)
    if ctx.enabled("subsampling"):
        # Bootstrap the training set per seed, which is where run-to-run
        # variance in this pipeline actually comes from.
        rng = ctx.rng("data")
        train = [train[rng.randrange(len(train))] for _ in range(len(train))]
    return train, test, lowercase, bigrams


def _finish(ctx: Context, model: Any, train: Sequence, test: Sequence,
            started: float, extra: dict[str, float] | None = None) -> dict[str, float]:
    result = score(model, test)
    predictions = result.pop("_predictions")
    ctx.write_json_artifact("predictions.json", {
        "predictions": predictions,
        "config": dict(ctx.config),
        "n_train": len(train),
        "n_test": len(test),
    })
    ctx.log(f"trained on {len(train)} documents, evaluated on {len(test)}")
    metrics = dict(result)
    metrics["train_seconds"] = time.perf_counter() - started
    metrics["vocabulary_size"] = float(len(getattr(model, "vocabulary", None)
                                           or getattr(model, "weights", {})))
    if extra:
        metrics.update(extra)
    return metrics


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def logistic_regression(ctx: Context) -> dict[str, float]:
    """The proposed method."""
    started = time.perf_counter()
    train, test, _lowercase, _bigrams = _prepare(ctx)
    model = LogisticRegression(
        learning_rate=float(ctx.get("learning_rate", 0.1)),
        epochs=int(ctx.get("epochs", 8)),
        batch_size=int(ctx.get("batch_size", 16)),
        l2=float(ctx.get("l2", 0.0)),
        seed=ctx.seed,
    ).fit(train)
    return _finish(ctx, model, train, test, started, {"train_loss": model.final_loss})


def naive_bayes(ctx: Context) -> dict[str, float]:
    """A strong, fast, entirely conventional baseline."""
    started = time.perf_counter()
    train, test, _lowercase, _bigrams = _prepare(ctx)
    model = NaiveBayes(alpha=float(ctx.get("alpha", 1.0))).fit(train)
    return _finish(ctx, model, train, test, started)


def majority(ctx: Context) -> dict[str, float]:
    """The floor: predict the most common class every time."""
    started = time.perf_counter()
    train, test, _lowercase, _bigrams = _prepare(ctx)
    model = MajorityClass().fit(train)
    return _finish(ctx, model, train, test, started)
