from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any


class RuntimeDependencies:
    def __init__(self, namespace: MutableMapping[str, Any]) -> None:
        object.__setattr__(self, "_namespace", namespace)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_namespace":
            object.__setattr__(self, name, value)
            return
        self._namespace[name] = value
