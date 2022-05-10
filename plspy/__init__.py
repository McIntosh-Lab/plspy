from .core import exceptions
from .core import gsvd
from .core import decorators
from .core import check_inputs
from .core import class_functions
from .core import resample
from .core import bootstrap_permutation
from .core import pls_classes
from .core import pls
from .core.pls import PLS
from .core.pls import methods

from .io import io
from .visualize import visualize

from . import __docs__

import sys

# __init__.py docstring assembled using blocks also used in
# other files. Docstrings found in __docs__.py
sys.modules[__name__].__doc__ = __docs__.plspy_header
sys.modules[__name__].__doc__ += __docs__.plspy_body


from . import _version

__version__ = _version.get_versions()["version"]
