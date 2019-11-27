import numpy as np

# project imports
import exceptions


def resample_without_replacement(matrix, C=None, return_indices=False):
    """Resamples input matrix without replacement. This implementation
    uses condition array `C` to shuffle the rows of `matrix` within
    conditions, without replacement. Returns `resampled` as shuffled
    `matrix`. Optionally returns shuffled indices (`shuf_indices`).

    Parameters
    ----------
    matrix : array-like
             Input matrix of arbitrary dimension and type. Will be
             cast to numpy array.
    C : array-like, optional
        Condition matrix used to shuffle rows of `matrix` according
        to condition. Should be in list form or a ndarray of shape
        (n,) where n > = 1. Initializes to `np.ones(matrix.shape[0])`
        by default.

    Returns
    -------
    resampled : numpy_array
                Resampled array, without replacement, of same
                shape as `matrix`.
    return_indices : boolean, optional
                If set to True, returns the ordering of the shuffled
                conditions list.
    """

    # initialize C to be one condition if not otherwise specified
    if C is None:
        C = np.ones(matrix.shape[0])

    if len(C.shape) > 1:
        raise exceptions.ConditionMatrixMalformedError(
            "Condition matrix has improper dimensions."
            "Must be of dimension (n,). Was {} instead.".format(C.shape)
        )
    # extract unique condition numbers
    C_vals = np.unique(C)

    # variable where shuffled indices will be calculated and stored
    shuf_indices = np.array(range(matrix.shape[0]))

    # for number of unique conditions
    for idx in range(len(C_vals)):
        # extract indices corresponding to current condition and shuffle them
        tmp = shuf_indices[C == C_vals[idx]]
        np.random.shuffle(tmp)
        # replace original indices with shuffled ones
        shuf_indices[C == C_vals[idx]] = tmp

    # shuffle matrix values according to shuffled indices
    resampled = matrix[shuf_indices, :]
    # return shuffled indices if specified
    if return_indices:
        return (resampled, shuf_indices)
    return resampled

    # flat = np.array(matrix).reshape(-1)
    # resamp = np.random.permutation(flat)
    # resampled = resamp.reshape(matrix.shape)
    # return resampled


def resample_with_replacement(matrix, C=None, return_indices=False):
    """Resamples input matrix with replacement. This implementation
    uses condition array `C` to shuffle the rows of `matrix` within
    conditions, with replacement. Returns `resampled` as shuffled
    `matrix`. Optionally returns shuffled indices (`shuf_indices`).

    Parameters
    ----------
    matrix : array-like
             Input matrix of arbitrary dimension and type. Will be
             cast to numpy array.
    C : array-like, optional
        Condition matrix used to shuffle rows of `matrix` according
        to condition. Should be in list form or a ndarray of shape
        (n,) where n > = 1. Initializes to `np.ones(matrix.shape[0])`
        by default.

    Returns
    -------
    resampled : numpy_array
                Resampled array, with replacement, of same
                shape as `matrix`.
    return_indices : boolean, optional
                If set to True, returns the ordering of the shuffled
                conditions list.
    """

    # initialize C to be one condition if not otherwise specified
    if C is None:
        C = np.ones(matrix.shape[0])

    if len(C.shape) > 1:
        raise exceptions.ConditionMatrixMalformedError(
            "Condition matrix has improper dimensions."
            "Must be of dimension (n,). Was {} instead.".format(C.shape)
        )

    # extract unique condition numbers
    C_vals = np.unique(C)

    # variable where shuffled indices will be calculated and stored
    shuf_indices = np.array(range(matrix.shape[0]))

    # for number of unique conditions
    for idx in range(len(C_vals)):
        # extract indices corresponding to current condition and shuffle them
        tmp = shuf_indices[C == C_vals[idx]]
        # shuffle with replacement
        rand_inds = np.random.randint(len(tmp), size=len(tmp))
        # replace original indices with shuffled ones
        shuf_indices[C == C_vals[idx]] = tmp[rand_inds]

    # shuffle matrix values according to shuffled indices
    resampled = matrix[shuf_indices, :]
    # return shuffled indices if specified
    if return_indices:
        return (resampled, shuf_indices)
    return resampled

    # flat = np.array(matrix).reshape(-1)
    # resampled = np.random.choice(flat, size=matrix.shape, replace=True)
    # return resampled
