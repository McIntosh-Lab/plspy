import numpy as np

# project imports
import gsvd
import resample


def bootstrap_test(X, Y, niter=1000, conf_int=(0.5, 0.95,)):
    """
    """
    pass


def permutation_test(X, Y, U, S, V, niter=1000, nonrotated=None):
    """
    """

    singular_totals = np.empty((niter, S.shape[0]))
