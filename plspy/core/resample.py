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

    inds = np.array([i for i in range(len(matrix))])
    grp_split = None
    start = 0
    # Split indices into groups and conditions based on cond_order
    for i, group_sizes in enumerate(cond_order):
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start : start + cond_size])  # Slice indices for condition
            start += cond_size
        group_split = np.column_stack(group_split)  # Stack conditions for this group

        # Concatenate into a mega-array
        if grp_split is None:
            grp_split = group_split  # Initialize with the first group
        else:
            grp_split = np.concatenate((grp_split, group_split))  # Horizontally concatenate groups
    
    grp=grp_split

    # Shuffle within each subject's condition
    within_subject_shuffle = np.apply_along_axis(np.random.permutation, axis=1, arr=grp)

    # Shuffle across all subjects
    shuff = np.copy(within_subject_shuffle.T)
    for col in range(grp.shape[1]):
        shuff[col, :] = np.random.permutation(within_subject_shuffle.T[col, :])

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
    inds = np.array([i for i in range(len(matrix))])
    start = 0

    my_resampled=None
    my_shuf_indices=None

    # Split indices into groups and conditions based on cond_order
    for i, group_sizes in enumerate(cond_order):
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start : start + cond_size])  # Slice indices for condition
            start += cond_size
        group_split = np.column_stack(group_split)  # Stack conditions for this group

        # Resample
        num_rows = group_split.shape[0]
        shuffled_indices = np.random.choice(num_rows,num_rows, replace=True)
    
        shuf_cond = []
        for col in range(group_split.shape[1]):  # Iterate through each column in grp
            shuf_cond.append(group_split[shuffled_indices, col])  # Append each column

        # Stack the list of arrays
        shuf_cond = np.vstack(shuf_cond)
        
        # flatten
        shuf_indices = shuf_cond.ravel()

        resampled = matrix[shuf_indices, :]

        if my_resampled is None:
            my_resampled = resampled  # Initialize with the first group
            my_shuf_indices=shuf_indices
        else:
            my_resampled = np.concatenate((my_resampled, resampled))  # Horizontally concatenate groups
            my_shuf_indices = np.concatenate((my_shuf_indices, shuf_indices))
        # return shuffled indices if specified

    if return_indices:
        return (my_resampled, my_shuf_indices)
    return my_resampled

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
