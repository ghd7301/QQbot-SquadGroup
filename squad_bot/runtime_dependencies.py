from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RuntimeDependencies:
    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        try:
            return self._namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
