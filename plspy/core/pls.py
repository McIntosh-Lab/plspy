from typing import List, Optional, Tuple, Type, Union

import numpy as np

from .. import __docs__

# project imports
from . import pls_classes

# dictionary holding PLS methods; used with help()
methods = {
    "mct": pls_classes._MeanCentreTaskPLS,
    "rb": pls_classes._RegularBehaviourPLS,
    "cst": pls_classes._ContrastTaskPLS,
    "csb": pls_classes._ContrastBehaviourPLS,
    "mb": pls_classes._MultiblockPLS,
    "cmb": pls_classes._ContrastMultiblockPLS,
}


def PLS(*args: np.ndarray, **kwargs: str) -> Type[pls_classes.PLSBase]:
    """
    Driver function for PLS. This function collects arguments from the user,
    passes them to the specified version of PLS, and returns the result.

    To read more on each flavour of PLS, see the docs for them under
    `pls_classes`.

    Parameters
    ----------
    *args : np.ndarray
        Positional (required) arguments for PLS. Passed into `pls_classes`.
    **kwargs : str
        Keyword (optional) arguments for PLS. Passed into `pls_classes`.
    Returns
    -------
    Type[pls_classes.PLSBase]
        PLS result structure from the specified PLS version. See
        docs of `pls_classes` for info on exact return values/types.
    """
    # TODO: handle first argument being pls method
    # print(f"arg1:{args[0]}")

    try:
        pls_method = kwargs.pop("pls_method")
        kwargs["pls_alg"] = pls_method
    except KeyError:
        pls_method = "mct"
        kwargs["pls_alg"] = pls_method


    if "num_split" in kwargs:
        if kwargs["num_split"] < 0 or not isinstance(kwargs["num_split"],int):
            raise ValueError(
                f"Invalid number of splits provided. Value must be a positive integer."
            )
        
        if "CI" in kwargs:
            if kwargs["CI"] is None or kwargs["CI"]< 0 or kwargs["CI"]>1:
                raise ValueError(
                    f"CI should be within 0 and 1."
                )

        if "lv" in kwargs:
            if kwargs["lv"] <=0 or not isinstance(kwargs["lv"],int):
                raise ValueError(
                    f"lv must be a positive integer greater than 0."
                )
            
    if "num_boot" in kwargs:
        if kwargs["num_boot"] < 0 or not isinstance(kwargs["num_boot"],int):
            raise ValueError(
                f"Invalid number of bootstraps provided. Value must be a positive integer."
            )
    if "num_perm" in kwargs:
        if kwargs["num_perm"] < 0 or not isinstance(kwargs["num_perm"],int):
            raise ValueError(
                f"Invalid number of permutations provided. Value must be a positive integer."
            )        

    # TODO: find a cleaner way to do this
    # if args[1] is not None:
    #     if args[1]

    # print(len(args))
    # pls_method = kwargs.get("pls_method")
    # if pls_method is None:
    #     pls_method = "mct"
    #     kwargs["pls_method"] = pls_method

    # print(kwargs)
    # return finished PLS class with user-specified method
    return pls_classes.PLSBase._create(pls_method, *args, **kwargs)


# __init__.py docstring assembled using blocks also used in
# other files. Docstrings found in __docs__.py
PLS.__doc__ = __docs__.pls_wrapper_header
PLS.__doc__ += __docs__.plspy_body
