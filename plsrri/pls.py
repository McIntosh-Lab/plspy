# import numpy as np

# project imports
from . import pls_classes

methods = {
            "mct" : pls_classes._MeanCentreTaskPLS,
            "rb" : pls_classes._RegularBehaviourPLS,
            "cst" : pls_classes._ContrastTaskPLS,
            "csb" : pls_classes._ContrastBehaviourPLS,
            "mb" : pls_classes._MultiblockPLS,
            "cmb" : pls_classes._ContrastMultiblockPLS
            }


def PLS(*args, **kwargs):
    """Front-facing wrapper function for PLS that captures user input
    and extracts user-specified PLS method. If no method is specified,
    default PLS is used.

    TODO: implement first version of PLS and document required values here

    """

    # TODO: handle first argument being pls method
    # print(f"arg1:{args[0]}")

    try:
        pls_method = kwargs.pop("pls_method")
        kwargs["pls_alg"] = pls_method
    except KeyError:
        pls_method = "mct"
        kwargs["pls_alg"] = pls_method

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
