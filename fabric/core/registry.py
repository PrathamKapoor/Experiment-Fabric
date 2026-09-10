"""The entity registry.

Registration is what turns "the dataset" into *a* dataset.  Entities are
stored by content id, and an index maps human references
(``benchmark-x@v3``) onto them.

The registry's most useful behaviour is the one that annoys people: if
you register ``benchmark-x@v3`` twice with different contents, it
refuses.  A version that silently changes underneath published results
is the single most common way a reproducible-looking experiment becomes
irreproducible.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Any, Iterable, Mapping, TypeVar

from fabric.core.entities import Algorithm, Dataset, Entity, Environment, Hardware, Model
from fabric.core.store import Workspace
from fabric.util.jsonio import read_json, write_json
from fabric.util.locking import FileLock

E = TypeVar("E", bound=Entity)

KINDS: dict[str, type] = {
    "dataset": Dataset,
    "model": Model,
    "algorithm": Algorithm,
    "environment": Environment,
    "hardware": Hardware,
}


class RegistryError(RuntimeError):
    """Raised on a registration conflict or a failed lookup."""


class Registry:
    """Content-addressed storage for the entities an experiment refers to."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    # -- paths -------------------------------------------------------------

    def _kind_dir(self, kind: str) -> str:
        if kind not in KINDS:
            raise RegistryError(f"unknown entity kind {kind!r}; expected one of {sorted(KINDS)}")
        directory = os.path.join(self.workspace.registry_dir, kind)
        os.makedirs(directory, exist_ok=True)
        return directory

    def _entity_path(self, kind: str, entity_id: str) -> str:
        return os.path.join(self._kind_dir(kind), f"{entity_id}.json")

    @property
    def _index_path(self) -> str:
        return os.path.join(self.workspace.registry_dir, "index.json")

    def _index(self) -> dict[str, dict[str, Any]]:
        return read_json(self._index_path, {}) or {}

    # -- registration ------------------------------------------------------

    def register(self, kind: str, entity: E, *, force: bool = False) -> E:
        """Store an entity, refusing silent version conflicts."""
        stamped = entity
        if hasattr(entity, "registered_at") and not getattr(entity, "registered_at"):
            stamped = replace(entity, registered_at=time.time())  # type: ignore[arg-type]

        entity_id = stamped.id
        ref = getattr(stamped, "ref", entity_id)
        lock = FileLock(self._index_path)
        lock.acquire()
        try:
            return self._register_locked(kind, stamped, entity_id, ref, force=force)
        finally:
            lock.release()

    def _register_locked(self, kind: str, stamped: Any, entity_id: str, ref: str,
                         *, force: bool) -> Any:
        index = self._index()
        key = f"{kind}:{ref}"
        existing = index.get(key)
        if existing and existing.get("id") != entity_id and not force:
            raise RegistryError(
                f"{ref} is already registered as {existing['id']} with different contents.\n"
                f"  registered: {existing.get('summary', '')}\n"
                f"  incoming:   {_summarise(stamped)}\n"
                "A version must mean one thing forever. Bump the version, or pass "
                "force if you are certain the earlier registration was wrong."
            )

        path = self._entity_path(kind, entity_id)
        if not os.path.exists(path):
            write_json(path, stamped.to_dict())
        index[key] = {
            "id": entity_id,
            "kind": kind,
            "name": getattr(stamped, "name", ""),
            "version": getattr(stamped, "version", ""),
            "summary": _summarise(stamped),
            "registered_at": time.time(),
        }
        index[f"{kind}:id:{entity_id}"] = index[key]
        write_json(self._index_path, index)
        return stamped  # type: ignore[return-value]

    def register_dataset(self, dataset: Dataset, *, force: bool = False) -> Dataset:
        return self.register("dataset", dataset, force=force)

    def register_model(self, model: Model, *, force: bool = False) -> Model:
        return self.register("model", model, force=force)

    def register_algorithm(self, algorithm: Algorithm, *, force: bool = False) -> Algorithm:
        return self.register("algorithm", algorithm, force=force)

    def register_environment(self, environment: Environment) -> Environment:
        return self.register("environment", environment, force=True)

    def register_hardware(self, hardware: Hardware) -> Hardware:
        return self.register("hardware", hardware, force=True)

    # -- lookup ------------------------------------------------------------

    def get(self, kind: str, entity_id: str) -> Any:
        """Load an entity by content id."""
        data = read_json(self._entity_path(kind, entity_id))
        if data is None:
            raise RegistryError(f"no {kind} registered with id {entity_id}")
        return _rebuild(kind, data)

    def find(self, kind: str, name: str, version: str = "") -> Any:
        """Look up an entity by name and optional version."""
        index = self._index()
        if version:
            entry = index.get(f"{kind}:{name}@{version}")
            if entry:
                return self.get(kind, entry["id"])
            raise RegistryError(
                f"{kind} '{name}@{version}' is not registered. "
                f"Register it with 'fabric register {kind}' first."
            )
        candidates = [
            entry for key, entry in index.items()
            if key.startswith(f"{kind}:") and ":id:" not in key and entry.get("name") == name
        ]
        if not candidates:
            raise RegistryError(f"{kind} '{name}' is not registered")
        candidates.sort(key=lambda e: (str(e.get("version", "")), e.get("registered_at", 0)))
        return self.get(kind, candidates[-1]["id"])

    def exists(self, kind: str, entity_id: str) -> bool:
        return os.path.exists(self._entity_path(kind, entity_id))

    def list(self, kind: str) -> list[Any]:
        """Every registered entity of one kind, newest first."""
        directory = self._kind_dir(kind)
        entities = []
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            data = read_json(os.path.join(directory, name))
            if data:
                entities.append(_rebuild(kind, data))
        entities.sort(key=lambda e: getattr(e, "registered_at", 0.0), reverse=True)
        return entities

    def index_entries(self, kind: str | None = None) -> list[dict[str, Any]]:
        entries = [
            entry for key, entry in self._index().items()
            if ":id:" not in key and (kind is None or entry.get("kind") == kind)
        ]
        entries.sort(key=lambda e: e.get("registered_at", 0), reverse=True)
        return entries

    # -- verification ------------------------------------------------------

    def verify(self, kind: str, entity_id: str) -> tuple[bool, str]:
        """Recompute an entity's content id and confirm it still matches.

        This is the check behind a reproduction certificate: it proves the
        stored record has not been edited since it was written.
        """
        data = read_json(self._entity_path(kind, entity_id))
        if data is None:
            return False, f"{kind} {entity_id} is missing from the registry"
        entity = _rebuild(kind, data)
        if entity.id != entity_id:
            return False, (
                f"{kind} {entity_id} does not hash to its own identifier "
                f"(recomputed {entity.id}): the stored record has been modified"
            )
        return True, "content identifier verified"

    def verify_all(self) -> list[tuple[str, str, bool, str]]:
        results = []
        for kind in KINDS:
            directory = os.path.join(self.workspace.registry_dir, kind)
            if not os.path.isdir(directory):
                continue
            for name in sorted(os.listdir(directory)):
                if name.endswith(".json"):
                    entity_id = name[:-5]
                    ok, message = self.verify(kind, entity_id)
                    results.append((kind, entity_id, ok, message))
        return results


def _rebuild(kind: str, data: Mapping[str, Any]) -> Any:
    cls = KINDS[kind]
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    payload = {k: v for k, v in data.items() if k in fields}
    for key in ("splits", "schema", "config", "params", "packages", "lockfiles",
                "env_vars", "accelerator_runtime", "distributed"):
        if key in payload and payload[key] is None:
            payload[key] = {}
    if "gpus" in payload and payload["gpus"] is None:
        payload["gpus"] = []
    return cls(**payload)


def _summarise(entity: Any) -> str:
    for attribute in ("digest", "weights_digest", "source_digest", "commit"):
        value = getattr(entity, attribute, "")
        if value:
            return f"{attribute}={value[:23]}"
    return entity.id
