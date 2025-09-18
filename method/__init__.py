"""Methods namespace package with lazy access to synthesis modules."""

from importlib import import_module
import sys
from types import ModuleType
from typing import Dict

_SYNTHESIS_MODULES = {
    "privsyn",
    "AIM",
    "GEM",
    "TabDDPM",
    "DP_MERF",
    "private_gsd",
    "PrivMRF",
    "RAP",
}

__all__ = sorted(_SYNTHESIS_MODULES)


def _load_module(name: str) -> ModuleType:
    module = import_module(f"{__name__}.synthesis.{name}")
    sys.modules[f"{__name__}.{name}"] = module
    return module


def __getattr__(name: str) -> ModuleType:
    if name in _SYNTHESIS_MODULES:
        return _load_module(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> Dict[str, object]:  # pragma: no cover - introspection helper
    return sorted(set(globals()) | _SYNTHESIS_MODULES)
