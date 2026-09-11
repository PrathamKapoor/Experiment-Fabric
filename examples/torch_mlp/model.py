"""A real PyTorch training run, under Fabric.

This is the example that is not a simulation. It builds a tensor dataset,
trains a multi-layer perceptron with autograd and Adam, and evaluates it
on a held-out split. Every number it reports comes out of PyTorch.

It exists to answer a question the synthetic examples cannot: **does a
real training pipeline reproduce?** The answer, measured rather than
assumed, turns out to be interesting -- CPU training under a fixed seed is
deterministic at a given thread count, but changing the thread count
perturbs the loss at around the eighth significant figure while leaving
accuracy untouched. That is precisely the distinction between a
perturbation being *detectable* and being *acceptable*, and it is why the
report carries both a stability score and an equivalence verdict.

Requires PyTorch::

    pip install torch --index-url https://download.pytorch.org/whl/cpu

Nothing else in the project depends on it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

import torch
from torch import nn


class Context(Protocol):
    """The slice of a RunContext this file needs."""

    seed: int
    config: dict[str, Any]
    dataset: dict[str, Any]
    model: dict[str, Any]
    ablated: tuple[str, ...]

    def get(self, key: str, default: Any = None) -> Any: ...
    def enabled(self, component: str) -> bool: ...
    def log(self, message: str) -> None: ...
    def write_json_artifact(self, name: str, value: Any, *, role: str = "output") -> str: ...


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_split(dataset: dict[str, Any], split: str) -> tuple[torch.Tensor, torch.Tensor]:
    path = str(dataset.get("local_path") or "")
    if not path:
        raise FileNotFoundError(
            "the dataset has no local_path; run "
            "'python examples/torch_mlp/make_data.py' and point the "
            "specification at the generated directory"
        )
    file_path = os.path.join(path, f"{split}.jsonl")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"{file_path} does not exist. Run "
            "'python examples/torch_mlp/make_data.py' first."
        )
    features: list[list[float]] = []
    labels: list[int] = []
    with open(file_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            features.append([float(row["x"]), float(row["y"])])
            labels.append(int(row["label"]))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def model_width(ctx: Context) -> int:
    """Capacity comes from the model record, not the run configuration.

    The model entity is what a model *version* means, so swapping
    ``mlp-2x64`` for ``mlp-2x128`` has to actually change the network.
    Reading width from the run config instead would make the model
    perturbation axis a no-op that reported perfect stability.
    """
    declared = (ctx.model.get("config") or {}).get("width")
    if declared is not None:
        return int(declared)
    return int(ctx.get("width", 64))


def build_model(ctx: Context, width: int) -> nn.Module:
    """An MLP whose optional pieces are Fabric components.

    ``normalisation`` and ``dropout`` are guarded by ``ctx.enabled`` so the
    ablations in the specification are runnable without editing this file.
    """
    layers: list[nn.Module] = [nn.Linear(2, width)]
    if ctx.enabled("normalisation"):
        layers.append(nn.BatchNorm1d(width))
    layers.append(nn.ReLU())
    if ctx.enabled("dropout"):
        layers.append(nn.Dropout(float(ctx.get("dropout_rate", 0.1))))
    layers.append(nn.Linear(width, width))
    if ctx.enabled("normalisation"):
        layers.append(nn.BatchNorm1d(width))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def configure_threads(ctx: Context) -> int:
    """Respect the thread count the environment asked for.

    Fabric's environment perturbations set ``OMP_NUM_THREADS``; PyTorch
    does not always pick that up once it has initialised, so it is applied
    explicitly. This is the knob the reproducibility study turns.
    """
    requested = os.environ.get("OMP_NUM_THREADS")
    if requested:
        try:
            torch.set_num_threads(max(1, int(requested)))
        except (ValueError, RuntimeError):
            pass
    return torch.get_num_threads()


LossFn = Any  # Callable[[Tensor, Tensor], Tensor]


def fused_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The usual path: a single fused cross-entropy kernel."""
    return nn.functional.cross_entropy(logits, target)


def unfused_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """log_softmax + nll_loss: the same mathematics, a different code path."""
    return nn.functional.nll_loss(nn.functional.log_softmax(logits, dim=1), target)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             loss_fn: LossFn = fused_loss) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y)
        predicted = logits.argmax(dim=1)
    correct = (predicted == y).sum().item()
    true_positive = ((predicted == 1) & (y == 1)).sum().item()
    false_positive = ((predicted == 1) & (y == 0)).sum().item()
    false_negative = ((predicted == 0) & (y == 1)).sum().item()
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return {
        "accuracy": correct / len(y),
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "loss": float(loss),
        "predictions": predicted.tolist(),
    }


def train(ctx: Context, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
          loss_fn: LossFn = fused_loss) -> float:
    epochs = int(ctx.get("epochs", 40))
    batch_size = int(ctx.get("batch_size", 64))
    learning_rate = float(ctx.get("learning_rate", 0.01))
    weight_decay = float(ctx.get("weight_decay", 0.0)) if ctx.enabled("weight_decay") else 0.0

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = None
    if ctx.enabled("lr_schedule"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    generator = torch.Generator().manual_seed(ctx.seed)
    final_loss = float("nan")
    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if len(batch) < 2:
                continue  # BatchNorm needs more than one sample
            optimiser.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimiser.step()
            final_loss = float(loss.detach())
        if scheduler is not None:
            scheduler.step()
    return final_loss


def _run(ctx: Context, width: int, loss_fn: LossFn = fused_loss) -> dict[str, float]:
    threads = configure_threads(ctx)
    torch.manual_seed(ctx.seed)

    x_train, y_train = load_split(ctx.dataset, "train")
    x_test, y_test = load_split(ctx.dataset, "test")

    started = time.perf_counter()
    model = build_model(ctx, width)
    train_loss = train(ctx, model, x_train, y_train, loss_fn)
    elapsed = time.perf_counter() - started

    result = evaluate(model, x_test, y_test, loss_fn)
    predictions = result.pop("predictions")
    ctx.write_json_artifact("predictions.json", {
        "predictions": predictions,
        "config": dict(ctx.config),
        "threads": threads,
        "torch_version": torch.__version__,
    })
    ctx.log(f"torch {torch.__version__}, {threads} thread(s), "
            f"{sum(p.numel() for p in model.parameters())} parameters")

    return {
        "accuracy": result["accuracy"],
        "f1": result["f1"],
        # test_loss is the sensitive metric: it moves under perturbations
        # that leave accuracy untouched, which is the point of reporting both.
        "test_loss": result["loss"],
        "train_loss": train_loss,
        "train_seconds": elapsed,
        "threads": float(threads),
    }


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def mlp(ctx: Context) -> dict[str, float]:
    """The proposed method: a two-hidden-layer MLP."""
    return _run(ctx, model_width(ctx))


def mlp_fused_loss(ctx: Context) -> dict[str, float]:
    """An alternative implementation of an identical computation.

    ``log_softmax`` followed by ``nll_loss`` is mathematically the same as
    ``cross_entropy``, and PyTorch computes it along a different code path.
    Whether that changes the answer is exactly what the implementation axis
    of the reproducibility study measures.

    The loss function is passed in rather than monkeypatched: a global
    swap would leak across threads when Fabric runs conditions in
    parallel, which would corrupt other runs rather than measure this one.
    """
    return _run(ctx, model_width(ctx), unfused_loss)


def logistic_regression(ctx: Context) -> dict[str, float]:
    """Baseline: a linear model, which cannot separate the spirals."""
    threads = configure_threads(ctx)
    torch.manual_seed(ctx.seed)
    x_train, y_train = load_split(ctx.dataset, "train")
    x_test, y_test = load_split(ctx.dataset, "test")

    started = time.perf_counter()
    model = nn.Linear(2, 2)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(ctx.get("learning_rate", 0.01)))
    generator = torch.Generator().manual_seed(ctx.seed)
    batch_size = int(ctx.get("batch_size", 64))
    train_loss = float("nan")
    for _epoch in range(int(ctx.get("epochs", 40))):
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
            train_loss = float(loss.detach())
    elapsed = time.perf_counter() - started

    result = evaluate(model, x_test, y_test)
    result.pop("predictions")
    ctx.log(f"linear baseline, torch {torch.__version__}, {threads} thread(s)")
    return {
        "accuracy": result["accuracy"],
        "f1": result["f1"],
        "test_loss": result["loss"],
        "train_loss": train_loss,
        "train_seconds": elapsed,
        "threads": float(threads),
    }


def untrained(ctx: Context) -> dict[str, float]:
    """Floor: the same architecture, never trained."""
    threads = configure_threads(ctx)
    torch.manual_seed(ctx.seed)
    x_train, y_train = load_split(ctx.dataset, "train")
    x_test, y_test = load_split(ctx.dataset, "test")
    started = time.perf_counter()
    model = build_model(ctx, model_width(ctx))
    model.eval()
    elapsed = time.perf_counter() - started
    result = evaluate(model, x_test, y_test)
    result.pop("predictions")
    # The loss at initialisation, not NaN: a metric Fabric cannot compare is
    # a failed run, and "untrained" is a legitimate arm with a real value.
    initial = evaluate(model, x_train, y_train)
    return {
        "accuracy": result["accuracy"],
        "f1": result["f1"],
        "test_loss": result["loss"],
        "train_loss": initial["loss"],
        "train_seconds": elapsed,
        "threads": float(threads),
    }
