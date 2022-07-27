import sys

from . import __docs__
from .core import (
    bootstrap_permutation,
    check_inputs,
    class_functions,
    decorators,
    exceptions,
    gsvd,
    pls,
    pls_classes,
    resample,
)
from .core.pls import PLS, methods
from .io import io
from .visualize import visualize

# __init__.py docstring assembled using blocks also used in
# other files. Docstrings found in __docs__.py
sys.modules[__name__].__doc__ = __docs__.plspy_header
sys.modules[__name__].__doc__ += __docs__.plspy_body


from . import _version

__version__ = _version.get_versions()["version"]
