import numpy as np


def mean_single_group(x, sg_cond_order):
    """Computes condition-wise mean of a single group and returns it.

    Computes the mean of the subjects/conditions in a single group and
    returns said mean. This function is usually called from pls_classes.py
    for use in the Mean-Center Task PLS algorithm.

    Parameters
    ----------
    x : np_array
        2-dimensional np_array of a single group.
    sg_cond_order : 1-dimensional np_array
        Single-group condition order corresponding to the input `x`. 
        Specifies the number of subjects per condition in a single group.

    Returns
    -------
    meaned : np_array
        Condition-wise mean of a single group.
    """
    # dim of nconds by ncols
    meaned = np.empty((len(sg_cond_order), x.shape[-1]))
    start = 0
    for i in range(len(sg_cond_order)):
        # store in each row of meaned the column-wise mean of each condition
        meaned[i,] = np.mean(x[start : sg_cond_order[i] + start,], axis=0)
        start += sg_cond_order[i]
    return meaned


def get_group_means(X, cond_order):
    """Computes the mean of each group and returns them.

    Computes the group-wise, element-wise mean of an input matrix `X` and
    returns an np_array of the computed means. This function is usually
    called from pls_classes.py for use in the Mean-Center Task PLS algorithm.

    Parameters
    ----------
    X : np_array
        2-dimensional input matrix with conditions and/or groups.
    cond_order : np_array
        Condition order for all groups in input `X`. Specifies the number
        of subjects per condition in each group.

    """

    group_means = np.empty((len(cond_order), X.shape[-1]))
    # sum of subjects across all conditions in each group
    group_sums = np.sum(cond_order, axis=1)
    # index tracking beginning of each group
    start = 0

    for i in range(len(cond_order)):
        group_means[i,] = np.mean(X[start : start + group_sums[i],], axis=0)
        start += group_sums[i]
    return group_means
