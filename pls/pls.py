import numpy as np

# project imports
import pls_classes


def PLS(*args, **kwargs):
    """Front-facing wrapper function for PLS that captures user input
    and extracts user-specified PLS method. If no method is specified,
    default PLS is used.

    TODO: implement first version of PLS and document required values here

    """
    try:
        pls_method = kwargs.pop("pls_method")
    except KeyError:
        pls_method = "default"

    # return finished PLS class with user-specified method
    return pls_classes.PLSBase.create(pls_method, *args, **kwargs)
