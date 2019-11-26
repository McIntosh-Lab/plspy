import numpy as np


def resample_without_replacement(matrix):
    """Resamples input matrix without replacement. This implementation
    flattens `matrix` first, resamples, and reshapes to
    original dimensions since numpy's permutation function only
    resamples along the first dimension of an input array.

    Parameters
    ----------
    matrix : array-like
             Input matrix of arbitrary dimension and type. Will be
             cast to numpy array.

    Returns
    -------
    resampled : numpy_array
                Resampled array, without replacement, of same
                shape as `matrix`.
    """
    flat = np.array(matrix).reshape(-1)
    resamp = np.random.permutation(flat)
    resampled = resamp.reshape(matrix.shape)
    return resampled


def resample_with_replacement(matrix):
    """Resamples input matrix with replacement. This implementation
    flattens `matrix` first, resamples, and reshapes to
    original dimensions since numpy's choice function only
    samples flat arrays.

    Parameters
    ----------
    matrix : array-like
             Input array of arbitrary dimension and type. Will be
             cast to numpy array.

    Returns
    -------
    resampled : numpy_array
                Resampled array, with replacement, of same
                shape as `matrix`.
    """
    flat = np.array(matrix).reshape(-1)
    resampled = np.random.choice(flat, size=matrix.shape, replace=True)
    return resampled
