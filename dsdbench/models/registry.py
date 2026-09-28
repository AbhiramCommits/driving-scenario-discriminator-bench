"""Name -> constructor registry so the eval harness can run any model by name."""

from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, Any] = {}


def register(name: str) -> Any:
    """Decorator registering a trainer class under ``name``."""

    def decorator(cls: Any) -> Any:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def create(name: str, config: dict[str, Any] | None = None) -> Any:
    """Instantiate the trainer registered under ``name`` with ``config``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {available()}") from None
    return cls(config if config is not None else {})


def available() -> list[str]:
    """Names of all registered models."""
    return sorted(_REGISTRY)


def registry() -> dict[str, Any]:
    """Raw name -> class mapping (for tests/introspection)."""
    return dict(_REGISTRY)
