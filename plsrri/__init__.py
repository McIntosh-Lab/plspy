
from . import exceptions
from . import gsvd
from . import decorators
from . import check_inputs
from . import class_functions
from . import resample
from . import bootstrap_permutation
from . import pls_classes
from . import pls
from .pls import PLS
from .pls import methods
from . import __docs__

import sys

# __init__.py docstring assembled using blocks also used in
# other files. Docstrings found in __docs__.py
sys.modules[__name__].__doc__ = __docs__.plsrri_header
sys.modules[__name__].__doc__ += __docs__.plsrri_body

