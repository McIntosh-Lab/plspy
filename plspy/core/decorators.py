import functools
from time import process_time


def proctimer(fct):
    """Function wrapper that times runtime (without sleeps) of input
    function. Returns value of function passed into wrapper and
    additionally prints out process time of function.

    Parameters
    ----------
    fct : function
          Function with arbitrary number of arguments.

    Returns
    -------
    res : arbitrary
          Return value of function `fct` passed into wrapper.
    """

    @functools.wraps(fct)
    def wrap_timer(*args, **kwargs):
        start = process_time()
        res = fct(*args, **kwargs)
        end = process_time()
        total = end - start
        print("{} finished in {} seconds".format(fct.__name__, total))
        return res

    return wrap_timer
