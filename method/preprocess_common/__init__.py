"""Method-level preprocessing utilities (PrivTree, DAWA, encoders)."""

from importlib import import_module

from .load_data_common import data_preporcesser_common
from .preprocess import discretizer, rare_merger

# Re-export modules so legacy imports like `from method.preprocess_common import privtree`
# continue to expose helper functions/classes on the module namespace.
_dawa_module = import_module(".dawa", __name__)
_privtree_module = import_module(".privtree", __name__)

dawa = _dawa_module
privtree = _privtree_module

__all__ = [
    "data_preporcesser_common",
    "discretizer",
    "rare_merger",
    "dawa",
    "privtree",
]

# Clean up helper names from module namespace.
del import_module, _dawa_module, _privtree_module
