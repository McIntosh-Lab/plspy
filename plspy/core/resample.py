import numpy as np
import scipy
import scipy.stats

# project imports
from . import class_functions, exceptions


def resample_without_replacement(
    matrix, cond_order, C=None, group_num=0, return_indices=False, pls_alg="mct"
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

    if pls_alg in ["mct", "cst", "mb", "cmb"]:
        # For Task PLS
        # Shuffle within each subject's condition
        within_subject_shuffle = np.apply_along_axis(np.random.permutation, axis=1, arr=grp)
        
        # Shuffle across all subjects
        shuff = np.copy(within_subject_shuffle.T)
        for col in range(grp.shape[1]):
            shuff[col, :] = np.random.permutation(within_subject_shuffle.T[col, :])
        # flatten
        shuf_indices = shuff.ravel()

    else:
        # For Behavioural PLS ("rb" or "csb") & Multi-Block Behaviour
        shuf_indices = np.random.permutation(np.shape(matrix)[0])

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
            #lower[i, j] = np.percentile(matrix[:, i, j], conf[0] * 100)
            #upper[i, j] = np.percentile(matrix[:, i, j], conf[1] * 100)

            X = np.squeeze(matrix[:, i, j])  # Remove single-dimensional entries
            X = np.sort(X)  # Sort the values
            r = len(X)  # Number of elements

            # Percentile positions
            x = np.concatenate(([0], (np.arange(0.5, r - 0.5 + 1) / r) * 100, [100]))
            y = np.concatenate(([X.min()], X, [X.max()]))

            # Interpolation
            lower[i,j] = np.interp(conf[0] * 100, x, y)
            upper[i,j] = np.interp(conf[1] * 100, x, y)
    
    return (lower, upper)

def _calculate_smeanmat(X_new_T, cond_order, mctype):
    """
    Calculates the mean-centered matrix (smeanmat) based on the given mctype (for multi-block PLS).

    Parameters:
    X_new_T: Bootstrapped data matrix (samples x features) for Task portion of multi-block PLS.
    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.
    mctype: int
        Specify which mean-centring method to use.

    Returns:
    smeanmat: The mean-centered data matrix.
    """

    if mctype == 0:
        # Calculate group means
        XT_means = class_functions._get_group_means(X_new_T, cond_order)

        # Calculate the number of repeats for each group
        group_sizes = np.array([sum(g) for g in cond_order])

        # Get smeanmat
        smeanmat = X_new_T - np.repeat(XT_means, group_sizes, axis=0)

    elif mctype == 1:
        # Calculate grand condition means
        XT_means = class_functions._get_grand_condition_means(X_new_T, cond_order)

        # Calculate the number of repeats for each condition
        condition_sizes = np.array(cond_order.flatten())

        # Repeat condition means for each group
        condition_means_for_groups = np.tile(XT_means, reps=(cond_order.shape[0], 1))

        # Get smeanmat
        smeanmat = X_new_T - np.repeat(condition_means_for_groups, condition_sizes, axis=0)

    elif mctype == 2:
        smeanmat = X_new_T - np.mean(X_new_T, axis=0)

    elif mctype == 3:
        # Calculate group means, condition means, and the grand mean
        XT_group_means = class_functions._get_group_means(X_new_T, cond_order)
        XT_cond_means = class_functions._get_grand_condition_means(X_new_T, cond_order)
        XT_grand_mean = np.mean(XT_cond_means, axis=0)

        # Calculate the number of repeats for each group and condition
        group_sizes = np.array([sum(group) for group in cond_order])
        condition_sizes = np.array(cond_order.flatten())

        # Repeat condition means for each group
        condition_means_for_groups = np.tile(XT_cond_means, (cond_order.shape[0], 1))

        # Expand group means, condition means, and the grand mean to match X_new_T
        group_means_expanded = np.repeat(XT_group_means, group_sizes, axis=0)
        condition_means_expanded = np.repeat(condition_means_for_groups, condition_sizes, axis=0)
        grand_mean_expanded = np.tile(XT_grand_mean, (X_new_T.shape[0], 1))

        # Get smeanmat
        smeanmat = X_new_T - group_means_expanded - condition_means_expanded + grand_mean_expanded

    return smeanmat