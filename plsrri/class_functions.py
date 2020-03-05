import numpy as np
import scipy.stats


def _mean_centre(X, cond_order, mctype=0, return_means=True):
    """Single-group preprocessing for `X`. Generates `X_means` and
    `X_mc` for use with `run_pls`

    Parameters
    ---------
    X : np_array
        Input matrix for use with PLS.
    cond_order: array-like
        List/array where each entry holds the number of subjects per
        condition for each group in the input matrix.
    mctype : int, optional
        Specify which mean-centring method to use. TODO: add other types
    return_means : boolean, optional
        Optionally specify whether or not to return the means along
        with the mean-centred matrix.
        
    Returns
    -------
    X_means: np_array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np_array
        Mean-centred values corresponding to input matrix X.


    """
    X_means = np.empty((np.product(cond_order.shape), X.shape[-1]))
    group_means = _get_group_means(X, cond_order)
    group_sums = np.sum(cond_order, axis=1)
    # index counters for X_means and X, respectively
    mc = 0
    xc = 0

    for i in range(len(cond_order)):
        X_means[mc : mc + len(cond_order[i]),] = _mean_single_group(
            X[xc : xc + group_sums[i],], cond_order[i]
        )
        mc += len(cond_order[i])
        xc += group_sums[i]

    repeats = np.array([len(i) for i in cond_order])
    # subtract condition-wise means from condition grand means
    # (within group)
    X_mc = X_means - np.repeat(group_means, repeats, axis=0)
    # X_mc /= np.linalg.norm(X_mc)
    if return_means:
        return (X_means, X_mc)
    else:
        return X_mc


def _run_pls(M):
    """Runs and returns results of Generalized SVD on `mc`,
    mean-centred input matrix `X`.

    Mostly just a wrapper for gsvd.gsvd right now, but may integrate
    other features in the future.

    Parameters
    ----------
    M: np_array
        Input matrix for use with SVD.

    Returns
    -------
    U: np_array
        Eigenvectors of matrix `M`*`M`^T;
        left singular vectors.
    s: np_array
        vector containing diagonal of the singular values.
    V: np_array
        Eigenvectors of matrix `M`^T*`M`;
        right singular vectors.
    """
    U, s, V = np.linalg.svd(M, full_matrices=False)
    return (U, s, V.T)


def _run_pls_contrast(M, C):
    """Derives U,s,V using input matrix M and contrast matrix C.

    Parameters
    ----------
    M: np.array
        Input matrix whose U,s,V will be derived.
    C: np.array
        Contrast matrix used during derivation.

    Returns
    -------
    U: np_array
        Eigenvectors of matrix `M`*`M`^T;
        left singular vectors.
    s: np_array
        vector containing diagonal of the singular values.
    V: np_array
        Eigenvectors of matrix `M`^T*`M`;
        right singular vectors.

    """
    CB = C.T @ M
    s = np.sqrt(np.sum(np.power(CB, 2), axis=0))
    V = CB.T
    U = C
    # V = VS / s

    # U = (np.linalg.inv(np.diag(s)) @ (V.T @ M.T)).T
    return (U, s, V)


def _compute_X_latents(I, EV, ngroups=1):
    """Computes latent values of original mxn input matrix `I`
    and corresponding nxn eigenvector `EV` by performing a dot-product.

    Parameters
    ----------
    I : np_array
        Input matrix of shape mxn.
    EV : np_array
        Corresponding eigenvector of shape nxn.
    ngroups: int
        Number of groups in input data.
    Returns
    -------
    dotp: np_array
        Computed dot-product of I and EV.
    """
    dotp = np.dot(I, EV)
    return dotp


def _compute_corr(X, Y, cond_order):
    """Compute per-condition correlation matrices (concatenated as R,
    in the case of Behavioural, to pass into GSVD).

    This algorithm uses neural input matrix X and behavioural matrix Y to
    compute per-condition correlation matrices. It then concatenates them
    and returns the correlations.

    Parameters
    ----------
    X : np.array
        Neural matrix passed into PLS class.
    Y : np.array
        Behavioural matrix passed into PLS class.
    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.

    Returns
    -------
    R : np.array
        Concatenated computed correlation matrices for each condition
        in X/Y.
    """
    R = np.empty((np.product(cond_order.shape) * Y.shape[1], X.shape[1]))
    # flatten ordering for easier iteration
    order_all = cond_order.reshape(-1)
    start = 0
    start_R = 0
    for i in range(len(order_all)):
        # X and Y zscored within each condition
        Xc_zsc = scipy.stats.zscore(X[start : order_all[i] + start,])
        Xc_zsc /= np.sqrt(order_all[i])
        # Xc_zsc *= -1
        print(f"Xc_zsc: \n{Xc_zsc.shape}\n")
        Yc_zsc = scipy.stats.zscore(Y[start : order_all[i] + start,])
        Yc_zsc /= np.sqrt(order_all[i])
        # Yc_zsc *= -1
        print(f"Yc_zsc: \n{Yc_zsc.shape}\n")
        np.nan_to_num(Xc_zsc, copy=False)
        np.nan_to_num(Yc_zsc, copy=False)
        R[start_R : Y.shape[1] + start_R,] = np.matmul(Yc_zsc.T, Xc_zsc)
        # print(f"R part: \n{R[start_R : Y.shape[1] + start_R,]}\n")
        start += order_all[i]
        start_R += Y.shape[1]

    return R


def _compute_Y_latents(Y, U, cond_order):
    """Compute latent variables per behavioural condition by breaking
    up Y and U into their corresponding blocks.
    """
    print(f"Y shape: {Y.shape}")
    print(f"co shape: {cond_order.shape}")
    print(f"U shape: {U.shape}")
    Y_latent = np.empty((Y.shape[0], U.shape[1]))
    start = 0
    start_R = 0
    start_U = 0
    order_all = cond_order.reshape(-1)

    for i in range(len(order_all)):
        Y_latent[start : order_all[i] + start,] = np.matmul(
            Y[start : order_all[i] + start,], U[start_U : Y.shape[1] + start_U,]
        )
        start += order_all[i]
        start_R += Y.shape[0]
        start_U += Y.shape[1]

    return Y_latent


def _mean_single_group(x, sg_cond_order):
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


def _get_group_means(X, cond_order):
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


def _create_multiblock(X, Y, cond_order, mctype=0):
    """Creates multiblock matrix from X and Y.

    Combines mean-centred result of X and correlation matrix R from X and Y
    to generate a multi-block matrix for use with SVD.

    Parameters
    ----------
    X : np.array
        Neural matrix passed into PLS class.
    Y : np.array
        Behavioural matrix passed into PLS class.
    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.
    mctype : int
        Specify which mean-centring method to use. TODO: add other types

    Returns
    -------
    mb : np.array
        stacked multi-block matrix containing mean-centring of X and
        correlation matrix computed from X and Y.

    """
    mc = _mean_centre(X, cond_order, mctype, return_means=False)
    R = _compute_corr(X, Y, cond_order)

    # stack mc and R
    mb = np.array([mc, R]).reshape(mc.shape[0] + R.shape[0], -1)
    return mb
