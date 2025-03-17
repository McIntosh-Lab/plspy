import numpy as np
import scipy.stats

from . import exceptions, gsvd


def _mean_centre(X, cond_order, mctype=0, return_means=True):
    """Mean-centring method to use on `X`. Generates `X_means` and
    `X_mc` for use with `run_pls`.

    mctype options:

    0 - within each group remove group means from condition means (default)
    1 - remove grand condition means from each group condition mean
    2 - remove grand mean (over all subjects and conditions)
    3 - remove all main effects - subtract condition and
        group means (group by condition)


    Parameters
    ---------
    X : np.array
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
    X_means: np.array
        Mean-values of X array from specified mean-centring method.
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.


    """

    ngrps = cond_order.shape[0]

    # within each group remove group means from condition means
    if mctype == 0:
        X_means = _get_group_condition_means(X, cond_order)
        group_means = _get_group_means(X, cond_order)

        repeats = np.array([len(i) for i in cond_order])
        # subtract condition-wise means from condition grand means
        # (within group)
        X_mc = X_means - np.repeat(group_means, repeats, axis=0)

    # remove grand condition means from each group condition mean
    elif mctype == 1:
        X_means = _get_group_condition_means(X, cond_order)
        grand_cond_means = _get_grand_condition_means(X, cond_order)

        # tile condition means so it's applied to both groups of X_means
        gcm_tiled = np.tile(A=grand_cond_means, reps=(ngrps, 1))

        X_mc = X_means - gcm_tiled

    # remove grand mean (over all subjects and conditions)
    elif mctype == 2:
        X_means = _get_group_condition_means(X, cond_order)
        X_means_gr = np.mean(X, axis=0)
        X_mc = X_means - X_means_gr

    # remove all main effects
    # subtract condition and group means (group by condition)
    elif mctype == 3:
        X_means = _get_group_condition_means(X, cond_order)
        group_means = _get_group_means(X, cond_order)
        cond_means = _get_grand_condition_means(X, cond_order)
        grand_mean = np.mean(cond_means,axis=0)

        # duplicate X_means so it applies to both groups
        group_repeats = np.array([len(i) for i in cond_order])
        gm_repeats = np.repeat(group_means, group_repeats[0], axis=0)
        cm_repeats = np.tile(A=cond_means, reps=(ngrps, 1))
        gr_repeats = np.tile(A=grand_mean, reps=(X_means.shape[0], 1))

        X_mc = X_means - cm_repeats - gm_repeats + gr_repeats
    else:
        raise exceptions.NotImplementedError(
            "Specified mean-centring method is either not implemented "
            "or is invalid."
        )

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
    M: np.array
        Input matrix for use with SVD.

    Returns
    -------
    U: np.array
        Eigenvectors of matrix `M`*`M`^T;
        left singular vectors.
    s: np.array
        vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `M`^T*`M`;
        right singular vectors.
    """
    # U, s, V = gsvd.gsvd(M, full_matrices=False)
    U, s, V = np.linalg.svd(M, full_matrices=False)
    return (U, s, V.T)


def _run_pls_contrast(M, C, compute_uv=True):
    """Derives U,s,V using input matrix M and contrast matrix C.

    Parameters
    ----------
    M: np.array
        Input matrix whose U,s,V will be derived.
    C: np.array
        Contrast matrix used during derivation.
    compute_uv: boolean, optional
        Specifies whether or not to compute and return Defaults to True.

    Returns
    -------
    U: np.array
        Contrast matrix.
    s: np.array
        vector containing diagonal of the singular values.
    V: np.array
        Result of contrasts applied to input.

    """
    CB = C.T @ M
    s = np.sqrt(np.sum(np.power(CB, 2), axis=1))

    if compute_uv:
        V = CB.T
        U = C
        
        # U = CB.T
        # V = C
        # V = VS / s

        # U = (np.linalg.inv(np.diag(s)) @ (V.T @ M.T)).T
        return (U, s, V)
    else:
        return s


def _compute_X_latents(X, EV):
    """Computes latent values of original mxn input matrix `X`
    and corresponding nxn eigenvector `EV` by performing a dot-product.

    Parameters
    ----------
    X : np.array
        Input matrix of shape mxn.
    EV : np.array
        Corresponding eigenvector of shape nxn.

    Returns
    -------
    dotp: np.array
        Computed dot-product of X and EV.
    """
    dotp = np.dot(X, EV)
    return dotp


def _compute_corr(X, Y, cond_order):  # , n_cond):
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
    # if n_cond == 1:
    #     X_zsc = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    #     Y_zsc = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    #     R = (Y_zsc.T @ X_zsc) / (X_zsc.shape[0] - 1)
    # else:
    for i in range(len(order_all)):
        # X and Y zscored within each condition
        Xc_zsc = scipy.stats.zscore(
            X[
                start : order_all[i] + start,
            ]
        )
        Xc_zsc /= np.sqrt(order_all[i])
        # Xc_zsc *= -1
        # print(f"Xc_zsc: \n{Xc_zsc.shape}\n")
        Yc_zsc = scipy.stats.zscore(
            Y[
                start : order_all[i] + start,
            ]
        )
        Yc_zsc /= np.sqrt(order_all[i])
        # Yc_zsc *= -1
        # print(f"Yc_zsc: \n{Yc_zsc.shape}\n")
        np.nan_to_num(Xc_zsc, copy=False)
        np.nan_to_num(Yc_zsc, copy=False)
        R[
            start_R : Y.shape[1] + start_R,
        ] = np.matmul(Yc_zsc.T, Xc_zsc)
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
            Y[
                start : order_all[i] + start,
            ],
            U[
                start_U : Y.shape[1] + start_U,
            ],
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
    x : np.array
        2-dimensional np.array of a single group.
    sg_cond_order : 1-dimensional np.array
        Single-group condition order corresponding to the input `x`.
        Specifies the number of subjects per condition in a single group.

    Returns
    -------
    meaned : np.array
        Condition-wise mean of a single group.
    """
    # dim of nconds by ncols
    meaned = np.empty((len(sg_cond_order), x.shape[-1]))
    start = 0
    for i in range(len(sg_cond_order)):
        # store in each row of meaned the column-wise mean of each condition
        meaned[i,] = np.mean(
            x[
                start : sg_cond_order[i] + start,
            ],
            axis=0,
        )
        start += sg_cond_order[i]
    return meaned


def _get_group_means(X, cond_order):
    """Computes the mean of each group and returns them.

    Computes the group-wise, element-wise mean of an input matrix `X` and
    returns an np.array of the computed means. This function is usually
    called from pls_classes.py for use in the Mean-Center Task PLS algorithm.

    Parameters
    ----------
    X : np.array
        2-dimensional input matrix with conditions and/or groups.
    cond_order : np.array
        Condition order for all groups in input `X`. Specifies the number
        of subjects per condition in each group.

    Returns
    -------
    meaned : np.array
        Condition-wise mean of a single group.
    """

    group_means = np.empty((len(cond_order), X.shape[-1]))
    # sum of subjects across all conditions in each group
    group_sums = np.sum(cond_order, axis=1)
    # index tracking beginning of each group
    start = 0

    for i in range(len(cond_order)):
        group_means[i,] = np.mean(
            X[
                start : start + group_sums[i],
            ],
            axis=0,
        )
        start += group_sums[i]
    return group_means


def _get_group_condition_means(X, cond_order):
    """Computes per-group condition means.

    Computes the group-wise, condition-wise mean of an input matrix `X` and
    returns an np.array of the computed means. This function is usually
    called from pls_classes.py for use in the Mean-Center Task PLS algorithm.

    Parameters
    ----------
    X : np.array
        2-dimensional input matrix with conditions and/or groups.
    cond_order : np.array
        Condition order for all groups in input `X`. Specifies the number
        of subjects per condition in each group.

    Returns
    -------
    meaned : np.array
        Condition-wise mean for each group.
    """

    grp_cond_means = np.empty((np.product(cond_order.shape), X.shape[-1]))
    # group_means = _get_group_means(X, cond_order)
    group_sums = np.sum(cond_order, axis=1) #number of subs in each group
    # index counters for X_means and X, respectively
    mc = 0
    xc = 0
    
    for i in range(len(cond_order)): #for each group
        grp_cond_means[mc : mc + len(cond_order[i]),] = _mean_single_group(
            X[
                xc : xc + group_sums[i], #grab subjects from current group from X 
            ],
            cond_order[i],
        )
        mc += len(cond_order[i])
        xc += group_sums[i]
    return grp_cond_means


def _get_grand_condition_means(X, cond_order):
    """Computes grand condition means.

    Computes the grand condition-wise mean of an input matrix `X` and
    returns an np.array of the computed means. This function is usually
    called from pls_classes.py for use in the Mean-Center Task PLS algorithm.

    Parameters
    ----------
    X : np.array
        2-dimensional input matrix with conditions and/or groups.
    cond_order : np.array
        Condition order for all groups in input `X`. Specifies the number
        of subjects per condition in each group.

    Returns
    -------
    meaned : np.array
        Condition-wise mean across all groups.
    """

    ngrp = cond_order.shape[0]
    ncond = cond_order.shape[1]

    # get group condition means and use
    # to compute grand condition means
    # (conditions must all have same number of subjects)
    # (change if this needs to be more flexible)
    grp_cond_means = _get_group_condition_means(X, cond_order)

    grand_cond_means = np.empty((cond_order.shape[1], X.shape[-1]))

    for cond in range(ncond):
        # create sets of indices that correspond to all
        # instances of a condition in all groups
        inds = [cond + grp * ncond for grp in range(ngrp)]

        # mean the selected indices and add to result array
        grand_cond_means[cond] = np.mean(grp_cond_means[inds, :], axis=0)

    return grand_cond_means


def _create_multiblock(X, cond_order, pls_alg, bscan, mctype=0, norm_opt = True, Xbscan=None, Ybscan=None):
    """Creates multiblock matrix from X and Y.

    Combines mean-centred result of X and correlation matrix R from X and Y
    to generate a multi-block matrix for use with SVD.

    Parameters
    ----------
    X : np.array
        Neural matrix passed into PLS class.
    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.
    mctype : int
        Specify which mean-centring method to use.
    #TO DO: add in description for norm_opt, Xbscan, Ybscan

    Returns
    -------
    mb : np.array
        stacked multi-block matrix containing mean-centring of X and
        correlation matrix computed from X and Y.

    """

    # Task portion of multi-block - uses full data
    if pls_alg in ["cmb"]:
        # Contrast multi-block
        mc = _get_group_condition_means(X, cond_order)
    else:
        # Regular multi-block
        mc = _mean_centre(X, cond_order, mctype, return_means=False) 

    # Behaviour portion of multi-block - uses bscan data
    bscan_cond_order = cond_order[:,bscan]
    
    R = _compute_corr(Xbscan, Ybscan, bscan_cond_order)
    start_mc = 0
    start_b = 0 
    stacked = []

    # Loop through each group & stack task portion on behaviour portion
    for group_sizes in cond_order:
        num_conditions_b = len(bscan)
        num_conditions_mc = len(group_sizes)
        # Extract corresponding rows for this group
        mc_group = mc[start_mc : start_mc + num_conditions_mc, :]
        R_group = R[start_b : start_b + num_conditions_b, :]

        if norm_opt is True:
            mc_group = mc_group / np.linalg.norm(mc_group, axis=1, keepdims=True)
            R_group = R_group / np.linalg.norm(R_group, axis=1, keepdims=True)

        # Stack mc_group and R_group under each other
        stacked.append(np.vstack((mc_group, R_group)))  

        start_mc += num_conditions_mc  # Update index
        start_b += num_conditions_b  # Update index

    # stack mc and R
    mb = np.vstack(stacked)

    return mb

def _get_Tu_Bu(U, n_cond, n_behav, cond_order, bscan):
    """
    Seperate U (Task/Behaviour LV) into Task & Behaviour for Multi-block PLS

    Parameters:
    -----------
    Tu : np.ndarray
        Task u 

    n_cond : int
        Number of conditions.

    n_behav : int
        Number of behaviours.

    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.

    bscan : array-like
        List/array specifying the subset of conditions to be used.

    Returns:
    --------
    Tu : np.ndarray
        Task portion of U.(equivalent to the task portion of `TBv` in matlab)
    Bu: np.ndarray
        Behaviour portion of U.(equivalent to the behaviour portion of `TBv` in matlab)
    """
    # Split U into Tu and Bu
    for group_num, group_sizes in enumerate(cond_order):
        group_num = group_num + 1       
        num_conditions_b = len(bscan)

        # Get Tu
        start_row = (group_num - 1) * n_cond + (group_num - 1) * num_conditions_b * n_behav
        end_row = start_row + n_cond

        # Extract the rows from u
        extracted_rows = U[start_row:end_row, :]

        # Append the extracted rows to Tu
        if group_num == 1:
            Tu = extracted_rows
        else:
            Tu = np.vstack((Tu, extracted_rows))
        
        # Get Bu
        start_row = (group_num - 1) * n_cond + (group_num - 1) * num_conditions_b * n_behav + n_cond
        end_row = start_row + num_conditions_b * n_behav

        # Extract the rows from u
        extracted_rows = U[start_row:end_row, :]

        # Append the extracted rows to Tu
        if group_num == 1:
            Bu = extracted_rows
        else:
            Bu = np.vstack((Bu, extracted_rows))

    return Tu, Bu
        
def _get_Tusc(Tu, n_cond, cond_order):
    """
    Compute the Tusc matrix - Task/Design scores 

    Parameters:
    -----------
    Tu : np.ndarray
        Task u (store seperately for multi-block)

    n_cond : int
        Number of conditions.

    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.

    Returns:
    --------
    Tusc : np.ndarray
        Matrix of task/design scores.(equivalent to the task portion of `TBvsc` in matlab)

    """
    for group_num, group_sizes in enumerate(cond_order):
        group_num = group_num + 1

        num_subs_in_group = group_sizes[0]
        num_col = Tu.shape[1]
        tmp = np.array([]).reshape(0, num_col)

        # Loop over conditions
        for k1 in range(n_cond):
            # Extract rows for the current condition
            start_row = (group_num - 1) * n_cond + k1
            tmp1 = Tu[start_row:start_row + 1, :]

            # Expand for the number of subjects in this condition
            tmp1 = np.tile(tmp1, (group_sizes[k1], 1))
            tmp = np.vstack((tmp, tmp1))

        # Append to Tusc
        if group_num == 1:
            Tusc = tmp
        else:
            Tusc = np.vstack((Tusc, tmp))
            
    return Tusc


def _get_Busc(Bu, n_cond, Ybscan, cond_order, bscan):
    """
    Compute the Busc matrix - Behaviour scores 

    Parameters:
    -----------
    Bu : np.ndarray
        Behaviour u (store seperately for multi-block)

    n_cond : int
        Number of conditions.

    Ybscan : np.array
        Behavioural matrix for conditions of interest.

    cond_order: np.array
        2-d np array containing number of subjects per condition in
        each group.

    bscan : array-like
        List/array specifying the subset of conditions to be used.

    Returns:
    --------
    Busc : np.ndarray
        Matrix of behaviour scores.(equivalent to the behaviour portion of `TBvsc` in matlab)

    """

    num_conditions_b = len(bscan)
    n_behav = Ybscan.shape[1]
    # Compute Busc
    Busc = []
    for group_num, group_sizes in enumerate(cond_order):
        group_num = group_num + 1

        num_subs_in_group = group_sizes[0]
        num_col = n_behav
        tmp = []

        if group_num == 1:
            span = 0
        else:
            span = sum(cond_order[:group_num-1,0]) * num_conditions_b

        # Loop over conditions
        for k1 in range(1,1+num_conditions_b):
            # Extract rows for the current condition
            bdata_rows = slice(span +  num_subs_in_group * (k1-1), span +  num_subs_in_group * k1)
            lv_rows = slice(num_col * (k1-1) + num_col * num_conditions_b * (group_num-1), 
                            num_col * (k1) + num_col * num_conditions_b * (group_num-1))

            # Perform matrix multiplication
            tmp1 = Ybscan[bdata_rows,:] @ Bu[lv_rows, :]
            tmp.append(tmp1)

        # Append to Busc
        if group_num == 1:
            Busc = np.vstack(tmp)
        else:
            Busc = np.vstack((Busc, np.vstack(tmp)))

    return Busc


def _normalize(variable):
    """
    Normalize Euclidean distance of vectors in original 
 	matrix to unit 1.
    """
    base = np.linalg.norm(variable, axis=0)
    normed_variable = np.divide(variable, base, where=base != 0)
   
    return normed_variable