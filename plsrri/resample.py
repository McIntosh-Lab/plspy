import numpy as np
import scipy
import scipy.stats

# project imports
from . import exceptions


def resample_without_replacement(
    matrix, cond_order, C=None, group_num=0, return_indices=False
):
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
    group_num : int, optional
        Group number of input matrix. Used to select corresponding condition
        matrix. Defaults to group 1.
    return_indices: boolean, optional
        Whether or not to return the shuffled indices corresponding to the
        shuffle in the returned matrix. Defaults to False.

    Returns
    -------
    resampled : numpy_array
                Resampled array, without replacement, of same
                shape as `matrix`.
    return_indices : boolean, optional
                If set to True, returns the ordering of the shuffled
                conditions list.
    """

    # # initialize C based on cond_order unless otherwise specified
    # if C is None:
    #     # C = np.ones((len(matrix), matrix[0].shape[0]))
    #     C = []
    #     for i in range(len(cond_order[group_num])):
    #         # for k in range(len(cond_order[i])):
    #         # tmp = []
    #         # for j in range(len(cond_order[i])):
    #         C.extend([i] * cond_order[group_num][i])
    #     C = np.array(C)
    # # print(C)
    # # select C array corresponding to group number
    # resampled = np.empty(matrix.shape)

    # # C = np.array(C[group_num])
    # # print(C)
    # # print(C.shape)
    # if len(C.shape) > 1:
    #     raise exceptions.ConditionMatrixMalformedError(
    #         "Condition matrix has improper dimensions."
    #         "Must be of dimension (n,). Was {} instead.".format(C.shape)
    #     )
    # # extract unique condition numbers
    # C_vals = np.unique(C)

    # # variable where shuffled indices will be calculated and stored
    # shuf_indices = np.array(range(matrix.shape[0]))

    # # for number of unique conditions
    # for idx in range(len(C_vals)):
    #     # extract indices corresponding to current condition and shuffle them
    #     tmp = shuf_indices[C == C_vals[idx]]
    #     np.random.shuffle(tmp)
    #     # replace original indices with shuffled ones
    #     shuf_indices[C == C_vals[idx]] = tmp

    # # shuffle matrix values according to shuffled indices
    # # EXPERIMENTAL
    # # shuf_indices = np.array([i for i in range(len(matrix))])
    # # np.random.shuffle(shuf_indices)
    # # resampled = matrix[shuf_indices, :]
    ncond = len(cond_order[0])
    nsub_grp = np.sum(cond_order[0])
    ngrp = len(cond_order)
    nsub_cond = int(nsub_grp / ncond)
    # generate indices list
    inds = np.array([i for i in range(len(matrix))])
    # reshape so indices are separated by group
    indsp = inds.reshape(ngrp, nsub_grp)
    # print(f"{indsp}\n")
    # reshape so subject indices are separated by group and ordered by
    # condition
    grp_split = np.array(
        [indsp[i].reshape(ncond, nsub_cond).T for i in range(len(indsp))]
    )
    # print(f"{grp_split}\n")
    # print(grp_split)
    # merge both groups into one 2-d matrix
    # if ngrp > 1:
    grp = grp_split.reshape(nsub_cond * ngrp, ncond)
    # else:
    #     grp = np.copy(grp_split)
    # print(f"{grp}\n")
    # shuffle first within subject's condition order and then
    # between all subjects
    shuff = np.random.permutation(np.random.permutation(grp.T).T)
    # print(shuff)
    # flatten
    shuf_indices = shuff.ravel()

    resampled = matrix[shuf_indices, :]
    # return shuffled indices if specified
    if return_indices:
        return (resampled, shuf_indices)
    return resampled

    # flat = np.array(matrix).reshape(-1)
    # resamp = np.random.permutation(flat)
    # resampled = resamp.reshape(matrix.shape)
    # return resampled


def resample_with_replacement(
    matrix, cond_order, C=None, group_num=0, return_indices=False
):
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
    group_num : int, optional
        Group number of input matrix. Used to select corresponding condition
        matrix. Defaults to group 1.
    return_indices: boolean, optional
        Whether or not to return the shuffled indices corresponding to the
        shuffle in the returned matrix. Defaults to False.

    Returns
    -------
    resampled : numpy_array
                Resampled array, with replacement, of same
                shape as `matrix`.
    return_indices : boolean, optional
                If set to True, returns the ordering of the shuffled
                conditions list.
    """
    # # group_num = len(cond_order) - 1
    # # initialize C based on cond_order unless otherwise specified
    # if C is None:
    #     # C = np.ones((len(matrix), matrix[0].shape[0]))
    #     C = []
    #     # cond_flat = cond_order.reshape(-1)
    #     for i in range(len(cond_order[group_num])):
    #         # for i in range(len(cond_flat)):
    #         # tmp = []
    #         # for j in range(len(cond_order[i])):
    #         C.extend([i] * cond_order[group_num][i])
    #     C = np.array(C)

    # ## initialize C to be one condition if not otherwise specified

    # # if C is None:
    # #     C = np.ones(matrix.shape[0])

    # # select C array corresponding to group number
    # # C = np.array(C[group_num - 1])

    # if len(C.shape) > 1:
    #     raise exceptions.ConditionMatrixMalformedError(
    #         "Condition matrix has improper dimensions."
    #         "Must be of dimension (n,). Was {} instead.".format(C.shape)
    #     )

    # # extract unique condition numbers
    # C_vals = np.unique(C)

    # # variable where shuffled indices will be calculated and stored
    # shuf_indices = np.array(range(matrix.shape[0]))

    # # for number of unique conditions
    # for idx in range(len(C_vals)):
    #     # extract indices corresponding to current condition and shuffle them
    #     tmp = shuf_indices[C == C_vals[idx]]
    #     # shuffle with replacement
    #     rand_inds = np.random.randint(len(tmp), size=len(tmp))
    #     # replace original indices with shuffled ones
    #     shuf_indices[C == C_vals[idx]] = tmp[rand_inds]

    # # shuffle matrix values according to shuffled indices
    # # shuf = np.array([i for i in range(len(matrix))])
    # # shuf_indices = np.random.choice(shuf, len(shuf), replace=True)
    # # resampled = matrix[shuf_indices, :]
    ncond = len(cond_order[0])
    nsub_grp = np.sum(cond_order[0])
    ngrp = len(cond_order)
    nsub_cond = int(nsub_grp / ncond)
    # generate indices list
    inds = np.array([i for i in range(len(matrix))])
    # reshape so indices are separated by group
    indsp = inds.reshape(ngrp, nsub_grp)
    # reshape so subject indices are separated by group and ordered by
    # condition
    grp_split = np.array(
        [indsp[i].reshape(nsub_cond, ncond).T for i in range(len(indsp))]
    )
    # print(grp_split)
    # merge both groups into one 2-d matrix
    grp = grp_split.reshape(nsub_cond * ngrp, ncond)
    # shuffle first within subject's condition order and then
    # between all subjects
    shuf_cond = np.array(
        [
            np.random.choice(grp[i, :], len(grp[i, :]), replace=True)
            for i in range(len(grp))
        ]
    )

    # shuff = np.random.permutation(np.random.permutation(grp.T).T)
    # flatten
    shuf_indices = shuf_cond.ravel()

    resampled = matrix[shuf_indices, :]

    # return shuffled indices if specified
    if return_indices:
        return (resampled, shuf_indices)
    return resampled

    # flat = np.array(matrix).reshape(-1)
    # resampled = np.random.choice(flat, size=matrix.shape, replace=True)
    # return resampled


def confidence_interval(matrix, conf=(0.05, 0.95)):
    """Computes element-wise confidence interval on a NumPy array.
    Requires given NumPy array to have shape (`i`, `m`, `n`) where `m` is the
    number of rows, `n` is  the number of columns, and `i` is the number of
    `m`x`n` matrices stored in the NumPy array.

    Parameters
    ----------
    matrix : np_array
             NumPy array on which to compute element-wise confidence intervals
    conf : 2-tuple, optional
           Values to use to compute the confidence interval.
           Defaults to (0.05, 0.95).

    Returns
    -------
    conf_ints : 2-tuple
                2-tuple of `m`x`n` NumPy arrays. The first array corresponds
                to the lower interval and  the second array corresponds to
                the higher interval.
    """

    # TODO: update the documentation to reflect new changes

    nrow = matrix.shape[1]
    ncol = matrix.shape[2]

    lower = np.empty((nrow, ncol))
    upper = np.empty((nrow, ncol))

    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            # use percentile function to match Matlab output
            # multiply confidence values by 100 to match
            # NumPy's required input (0,100)
            lower[i, j] = np.percentile(matrix[:, i, j], conf[0] * 100)
            upper[i, j] = np.percentile(matrix[:, i, j], conf[1] * 100)

    return (lower, upper)
    # n = len(matrix)
    # m = scipy.mean(matrix, axis=0)
    # std_err = scipy.stats.sem(matrix, axis=0)
    # h_upper = std_err * scipy.stats.t.ppf((1 + conf[1]) / 2, n - 1)
    # h_lower = std_err * scipy.stats.t.ppf((conf[0]) / 2, n - 1)
    # conf_ints = (m - h_upper, m + h_upper)
    # return conf_ints
