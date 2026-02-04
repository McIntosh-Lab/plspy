import numpy as np
import scipy.stats

from . import class_functions, exceptions, gsvd, resample
def _get_cond_order(X_shape, groups_tuple, num_conditions):
    """
    Returns a list of lists. Each sub-list contains the number of subjects
    per condition for each group. Length of each sub-list is
    `num_conditions` and length of returned list is `len(groups_tuple)`.
    """

    if sum(groups_tuple) * num_conditions != X_shape[0]:
        raise exceptions.InputMatrixDimensionMismatchError(
            "Derived condition ordering not compatible with input matrix"
            "X's row count. Please specify a custom cond_order field."
        )

    cond_order = np.array(
        [np.array([i] * num_conditions) for i in groups_tuple]
    )
    return cond_order

def split_half_test_train(pls_alg, matrix, Y, cond_order, num_split, mctype=None, contrasts=None, bscan=None, Xbscan=None, Ybscan=None):
    """
    Perform split-half test-train PLS reproducibility test.

    Parameters:
        pls_alg (str): 
            Name of the PLS variant.

        matrix (np.ndarray): 
            Input data matrix with dimensions (subjects x features).

        Y (np.ndarray): 
            Input behavioral matrix.

        cond_order (array-like): 
            List/array where each entry holds the number of subjects per 
            condition for each group in the input matrix.

        num_split (int): 
            Number of split-half samples.

        mctype (int, optional): 
            Method for mean-centering the data

        contrasts (np.ndarray, optional): 
            Contrast matrix for use in Contrast Task PLS. Used to create 
            different methods of comparison.

    Returns:
        pls_repro_tt: Dictionary containing split-half test-train reproducibility results.

            Keys
            ----
            pls_s_train : np.ndarray
                Distribution of singular values from training samples.
            pls_s_test : np.ndarray
                Distribution of singular values from corresponding test samples.
            z : np.ndarray
                Z-values for each test singular value, computed as mean(pls_s_test)/std(pls_s_test).
            pls_s_train_null : np.ndarray
                Distribution singular values from training sample where rows for one data block are permuted.
            pls_s_test_null : np.ndarray
                Distribution singular values from test sample where rows for one data block are permuted.
            z_null : np.ndarray
                Z-values for null test singular values (mean(pls_s_test_null)/std(pls_s_test_null)).

    """

    inds = np.array([i for i in range(len(matrix))])
    num_conditions = np.shape(cond_order)[1]
    num_groups = np.shape(cond_order)[0]
    n, p = matrix.shape
    
    # Initialize empty arrays 
    if pls_alg in ["mct"]:
        d = min(p, num_conditions*num_groups)
    elif pls_alg in ["mb"]:
        d = min(p,num_conditions*num_groups + len(bscan)*num_groups * Ybscan.shape[1])
    elif pls_alg in ["cmb","cst","csb"]:
        d = min(p,contrasts.shape[1])
    else: # rb
        d = min(p, num_conditions*num_groups*Y.shape[1])
    
    pls_s_train = np.zeros((d, d, num_split))
    pls_s_test = np.zeros((d, d, num_split))

    pls_s_train_null = np.zeros((d, d, num_split))
    pls_s_test_null = np.zeros((d, d, num_split))

    start = 0
    allgroup_ids = None
    separate_group_ids = []

    # from scipy.io import loadmat
    # mat_inds = loadmat("C:/Users/lrokos/Documents/plspy_test/plspy/plspy/MATLAB_SPLITHALF_IDX.mat")
    # mat_inds= mat_inds["idx"]-1

    # Loop to get ids for each group
    for g, group_sizes in enumerate(cond_order): # g is group number in this loop
        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start : start + cond_size])  # Slice indices for condition
            start += cond_size
        group_split = np.column_stack(group_split)  # Stack conditions for this group
        separate_group_ids.append(group_split)

        # Concatenate into a mega-array
        if allgroup_ids is None:
            allgroup_ids = group_split  # Initialize with the first group
        else:
            allgroup_ids = np.concatenate((allgroup_ids, group_split))  # Horizontally concatenate groups

    # Loop for each split
    for i in range(num_split):
        start = 0
        idx_1_all = None
        idx_2_all = None
        idx_1_bscan = None
        idx_2_bscan = None
        group_tuple_1 = []
        group_tuple_2 = []       
        # Get splits within each group
        for g, group_sizes in enumerate(cond_order):
            group_split = separate_group_ids[g]
            n_per_g = group_split.shape[0]
            nsplit = int(np.floor(n_per_g / 2))

            # Randomly shuffle subjects
            idx = np.random.permutation(n_per_g)
            tmp_idx_subj = group_split[idx, :]

            # Split indices into training and testing
            idx_1 = tmp_idx_subj[:nsplit, :].flatten()
            idx_2 = tmp_idx_subj[nsplit:, :].flatten()

            # Append group indices
            if idx_1_all is None: # Initialize with the first group
                idx_1_all = idx_1  
                idx_2_all = idx_2
                group_tuple_1.append(len(idx_1)//num_conditions) # Get group 1 size for first split-half
                group_tuple_2.append(len(idx_2)//num_conditions) # Get group 1 size for second split-half
            else: # Horizontally concatenate groups
                idx_1_all = np.concatenate((idx_1_all, idx_1))  
                idx_2_all = np.concatenate((idx_2_all, idx_2))
                group_tuple_1.append(len(idx_1)//num_conditions) # Append subsequent group sizes for first split-half
                group_tuple_2.append(len(idx_2)//num_conditions) # Append subsequent group sizes for second split-half

            # idx_1_all = mat_inds[0:50,i]
            # idx_2_all = mat_inds[50:,i]

            if pls_alg in ["mb", "cmb"]:
                # Split indices into training and testing
                idx_1 = tmp_idx_subj[:nsplit, bscan].flatten()
                idx_2 = tmp_idx_subj[nsplit:, bscan].flatten()

                # Append group indices
                if idx_1_bscan is None: # Initialize with the first group
                    idx_1_bscan = idx_1
                    idx_2_bscan = idx_2
                else: # Horizontally concatenate groups
                    idx_1_bscan = np.concatenate((idx_1_bscan, idx_1))  
                    idx_2_bscan = np.concatenate((idx_2_bscan, idx_2))

        # Extract data for indices for each split-half
        X1 = matrix[idx_1_all, :]
        X2 = matrix[idx_2_all, :]

        # Get new cond_orders
        cond_order_1 = _get_cond_order(
                X1.shape, tuple(group_tuple_1), num_conditions
            )
        cond_order_2 = _get_cond_order(
                X2.shape, tuple(group_tuple_2), num_conditions
            )

        # Run the appropriate PLS
        if pls_alg == "mct":
            # Mean centering
            _, X1_mc = class_functions._mean_centre(
                X1, cond_order_1, mctype=mctype
            )   
            _, X2_mc = class_functions._mean_centre(
                X2, cond_order_2, mctype=mctype
            )

            # Perform mcPLS
            my_U, my_s, my_V = class_functions._run_pls(X1_mc)
            pls_s_train[:, :, i] = my_s
            pls_s_test[:, :, i] = my_V.T @ X2_mc.T @ my_U

        if pls_alg == "rb":
            Y1 = Y[idx_1_all, :]
            Y2 = Y[idx_2_all, :]

            # Compute correlation matrix
            R1= class_functions._compute_R(X1, Y1, cond_order_1)
            R2= class_functions._compute_R(X2, Y2, cond_order_2)

            # Perform bPLS
            my_U, my_s, my_V = class_functions._run_pls(R1)
            pls_s_train[:, :, i] = my_s
            pls_s_test[:, :, i] = my_V.T @ R2.T @ my_U

        if pls_alg == "cst":
            # Mean centering            
            X1_mc = class_functions._mean_centre(
            X1, cond_order_1, return_means=False, mctype=mctype
        )
            X2_mc = class_functions._mean_centre(
            X2, cond_order_2, return_means=False, mctype=mctype
        )
        # Perform Contrast Task PLS    
            my_U, my_s, my_V = class_functions._run_pls_contrast(
            X1_mc, contrasts)

            pls_s_train[:, :, i] = my_s
            pls_s_test[:, :, i] = my_V.T @ X2_mc.T @ my_U

        if pls_alg == "csb":
            
            Y1 = Y[idx_1_all, :]
            Y2 = Y[idx_2_all, :]

            # Compute correlation matrix
            R1= class_functions._compute_R(X1, Y1, cond_order_1)
            R2= class_functions._compute_R(X2, Y2, cond_order_2)

            # Perform contrast bPLS
            my_U, my_s, my_V = class_functions._run_pls_contrast(R1, contrasts)
            pls_s_train[:, :, i] = my_s
            pls_s_test[:, :, i] = my_V.T @ R2.T @ my_U

        if pls_alg in ["mb","cmb"]:
                
            Xbscan1 = matrix[idx_1_bscan, :]
            Ybscan1 = Y[idx_1_bscan, :]

            Xbscan2 = matrix[idx_2_bscan, :]
            Ybscan2 = Y[idx_2_bscan, :]

            # Get multiblock
            multiblock1 = class_functions._create_multiblock(
            X1, cond_order_1, pls_alg, bscan, mctype,
            Xbscan = Xbscan1, Ybscan = Ybscan1)

            multiblock2 = class_functions._create_multiblock(
            X2, cond_order_2, pls_alg, bscan, mctype,
            Xbscan = Xbscan2, Ybscan = Ybscan2)

            if pls_alg =="mb":    
                # Perform Multiblock PLS 
                my_U, my_s, my_V = class_functions._run_pls(multiblock1)

            if pls_alg =="cmb":
                # Perform Contrast Multiblock PLS 
                my_U, my_s, my_V = class_functions._run_pls_contrast(multiblock1, contrasts)

            pls_s_train[:, :, i] = my_s
            pls_s_test[:, :, i] = my_V.T @ multiblock2.T @ my_U

# Generate null distribution
    for i in range(num_split):
        n_per_cond = n // num_conditions
        nsplit = sum(group_tuple_1) # Set split to same size as non-null split
        idx = np.random.permutation(n_per_cond)
        tmp_idx_subj = allgroup_ids[idx, :]

        idx_1_null = tmp_idx_subj[:nsplit, :].flatten()
        idx_2_null = tmp_idx_subj[nsplit:, :].flatten()

        if pls_alg in ["mb","cmb"]:
            idx_1_bscan_null = tmp_idx_subj[:nsplit, bscan].flatten()
            idx_2_bscan_null = tmp_idx_subj[nsplit:, bscan].flatten()

        if pls_alg in ["mct" ,"cst","mb", "cmb"]:
            idx_perm = np.random.permutation(n)
            permx = matrix[idx_perm, :]
        else:
            permx = matrix


        # Extract data for indices for each split-half
        X1_null = permx[idx_1_null, :]
        X2_null = permx[idx_2_null, :]
      
        # Get new cond_orders
        cond_order_1 = _get_cond_order(
                X1_null.shape, tuple(group_tuple_1), num_conditions
            )
        cond_order_2 = _get_cond_order(
                X2_null.shape, tuple(group_tuple_2), num_conditions
            )

        # Run the appropriate PLS               
        if pls_alg == "mct":
            # Mean centering
            _, X1_mc_null = class_functions._mean_centre(
                X1_null, cond_order_1, mctype=mctype
            )
            _, X2_mc_null = class_functions._mean_centre(
                X2_null, cond_order_2, mctype=mctype
            )

            # Perform mcPLS
            my_U, my_s, my_V = class_functions._run_pls(X1_mc_null)
            pls_s_train_null[:, :, i] = my_s
            pls_s_test_null[:, :, i] = my_V.T @ X2_mc_null.T @ my_U

        if pls_alg == "rb":
            permy = Y[np.random.permutation(n), :]
            Y1_null = permy[idx_1_null, :]
            Y2_null = permy[idx_2_null, :]

            # Compute correlation matrix
            R1_null = class_functions._compute_R(X1_null, Y1_null, cond_order_1)
            R2_null = class_functions._compute_R(X2_null, Y2_null, cond_order_2)

            # Perform bPLS
            my_U, my_s, my_V = class_functions._run_pls(R1_null)
            pls_s_train_null[:, :, i] = my_s
            pls_s_test_null[:, :, i] = my_V.T @ R2_null.T @ my_U

        if pls_alg == "cst":
            # Mean centering            
            X1_mc_null = class_functions._mean_centre(
            X1_null, cond_order_1, return_means=False, mctype=mctype
        )
            X2_mc_null = class_functions._mean_centre(
            X2_null, cond_order_2, return_means=False, mctype=mctype
        )
            # Perform Contrast Task PLS    
            my_U, my_s, my_V = class_functions._run_pls_contrast(
            X1_mc_null, contrasts)
            pls_s_train_null[:, :, i] = my_s
            pls_s_test_null[:, :, i] = my_V.T @ X2_mc_null.T @ my_U  

        if pls_alg == "csb":
            permy = Y[np.random.permutation(n), :]
            Y1_null = permy[idx_1_null, :]
            Y2_null = permy[idx_2_null, :]

            # Compute correlation matrix
            R1_null = class_functions._compute_R(X1_null, Y1_null, cond_order_1)
            R2_null = class_functions._compute_R(X2_null, Y2_null, cond_order_2)

            # Perform contrast bPLS
            my_U, my_s, my_V = class_functions._run_pls_contrast(R1_null,contrasts)
            pls_s_train_null[:, :, i] = my_s
            pls_s_test_null[:, :, i] = my_V.T @ R2_null.T @ my_U

        if pls_alg == "mb":

            # Get multiblock
            multiblock1_null = class_functions._create_multiblock(
            X1_null, cond_order_1, pls_alg, bscan, mctype,
            Xbscan = permx[idx_1_bscan_null,:], Ybscan = Y[idx_1_bscan_null,:])

            multiblock2_null = class_functions._create_multiblock(
            X2_null, cond_order_2, pls_alg, bscan, mctype,
            Xbscan = permx[idx_2_bscan_null,:], Ybscan = Y[idx_2_bscan_null,:]) 

            # Perform Multiblock PLS 
            my_U, my_s, my_V = class_functions._run_pls(multiblock1_null)
            pls_s_train_null[:, :, i] = my_s
            pls_s_test_null[:, :, i] = my_V.T @ multiblock2_null.T @ my_U

        if pls_alg =="cmb":
            # Get multiblock
            multiblock1_null = class_functions._create_multiblock(
            X1_null, cond_order_1, pls_alg, bscan, mctype,
            Xbscan = permx[idx_1_bscan_null,:], Ybscan = Y[idx_1_bscan_null,:])

            multiblock2_null = class_functions._create_multiblock(
            X2_null, cond_order_2, pls_alg, bscan, mctype,
            Xbscan = permx[idx_2_bscan_null,:], Ybscan = Y[idx_2_bscan_null,:]) 

            # Perform Contrast Multiblock PLS 
            my_U, my_s, my_V = class_functions._run_pls_contrast(multiblock1_null, contrasts)

            pls_s_train_null[:, :, i] = my_s
            pls_s_test_null[:, :, i] = my_V.T @ multiblock2_null.T @ my_U


    # Compile results
    pls_repro = {
        "pls_s_train": pls_s_train,
        "pls_s_test": pls_s_test,
        "z": [
            np.mean(pls_s_test[i, i, :]) / np.std(pls_s_test[i, i, :],ddof=1)
            for i in range(d)
        ]
    }
    pls_repro["pls_s_train_null"] = pls_s_train_null
    pls_repro["pls_s_test_null"] = pls_s_test_null
    pls_repro["z_null"] = [
        np.mean(pls_s_test_null[i, i, :]) / np.std(pls_s_test_null[i, i, :],ddof=1)
        for i in range(d)
    ]
    return pls_repro


def split_half(pls_alg, matrix, Y, cond_order, num_split, mctype=None, contrasts=None, bscan=None, Xbscan=None, Ybscan=None, lv=1, CI=0.95):
    """
    Perform split-half reproducibility test.

    Parameters:
        pls_alg (str): 
            Name of the PLS variant.

        matrix (np.ndarray): 
            Input data matrix with dimensions (subjects x features).

        Y (np.ndarray): 
            Input behavioral matrix.

        cond_order (array-like): 
            List/array where each entry holds the number of subjects per 
            condition for each group in the input matrix.

        num_split (int): 
            Number of split-half samples.

        mctype (int, optional): 
            Method for mean-centering the data.

        contrasts (np.ndarray, optional): 
            Contrast matrix for use in Contrast Task PLS. Used to create 
            different methods of comparison.

        lv (int): 
            Number of LVs to evaluate.

        CI (float): 
            Confidence interval percentile. Defaults to 0.95.

    Returns:
        pls_repro_sh: Dictionary containing split-half reproducibility results.

            Keys
            ----
            pls_rep_mean_u : list[float]
                Average of cosines for u distribution from split-half.
            pls_rep_mean_v : list[float]
                Average of cosines for v distribution from split-half.
            pls_rep_z_u : list[float]
                Z-value for u distribution (mean_u/std_u)
            pls_rep_z_v : list[float]
                Z-value for v distribution (mean_v/std_v)
            pls_rep_ul_u : list[float]
                Upper bound of the u distribution.
            pls_rep_ll_u : list[float]
                Lower bound of the u distribution.
            pls_rep_ul_v : list[float]
                Upper bound of the v distribution.
            pls_rep_ll_v : list[float]
                Lower bound of the v distribution.

            pls_null_mean_u : list[float]
                Average of null u distribution created by permutation.
            pls_null_mean_v : list[float]
                Average of null v distribution.
            pls_null_z_u : list[float]
                Z-value for null u distribution.
            pls_null_z_v : list[float]
                Z-value for null v distribution.
            pls_null_ul_u : list[float]
                Upper bound of null u distribution.
            pls_null_ll_u : list[float]
                Lower bound of null u distribution.
            pls_null_ul_v : list[float]
                Upper bound of null v distribution.
            pls_null_ll_v : list[float]
                Lower bound of null v distribution.

            pls_dist_u : np.ndarray
                Full distribution of u cosines.
            pls_dist_v : np.ndarray
                Full distribution of v cosines.
            pls_dist_null_u : np.ndarray
                Full distribution of null u cosines.
            pls_dist_null_v : np.ndarray
                Full distribution of null v cosines.

    """

    inds = np.array([i for i in range(len(matrix))])
    num_conditions = np.shape(cond_order)[1]
    num_groups = np.shape(cond_order)[0]
    n, p = matrix.shape

    # Initialize empty arrays 
    if pls_alg in ["mct"]:
        d = min(p, num_conditions*num_groups)
    elif pls_alg in ["mb"]:
        d = min(p,num_conditions*num_groups + len(bscan)*num_groups * Ybscan.shape[1])
    elif pls_alg in ["cmb","cst","csb"]:
        d = min(p,contrasts.shape[1])
    else:
        d = min(p, num_conditions*num_groups*Y.shape[1])

    pls_u_repro = np.zeros((d, d, num_split))
    pls_v_repro = np.zeros((d, d, num_split))

    pls_u_null = np.zeros((d, d, num_split))
    pls_v_null = np.zeros((d, d, num_split))

    start = 0
    allgroup_ids = None
    separate_group_ids = []

    # Loop to get ids for each group
    for g, group_sizes in enumerate(cond_order):

        group_split = []
        for cond_size in group_sizes:
            group_split.append(inds[start : start + cond_size])  # Slice indices for condition
            start += cond_size
        group_split = np.column_stack(group_split)  # Stack conditions for this group
        separate_group_ids.append(group_split)

        # Concatenate into a mega-array
        if allgroup_ids is None:
            allgroup_ids = group_split  # Initialize with the first group
        else:
            allgroup_ids = np.concatenate((allgroup_ids, group_split))  # Horizontally concatenate groups


    # Loop for each split
    for i in range(num_split):
        start = 0
        idx_1_all = None
        idx_2_all = None
        idx_1_bscan = None
        idx_2_bscan = None
        group_tuple_1 = []
        group_tuple_2 = []   
        # Get splits within each group
        for g, group_sizes in enumerate(cond_order):
            group_split = separate_group_ids[g]

            n_per_g = group_split.shape[0]
            nsplit = int(np.floor(n_per_g / 2))

            # Randomly shuffle subjects
            idx = np.random.permutation(n_per_g)
            tmp_idx_subj = group_split[idx, :]

            # Split indices into training and testing
            idx_1 = tmp_idx_subj[:nsplit, :].flatten()
            idx_2 = tmp_idx_subj[nsplit:, :].flatten()

            # Append group indices
            if idx_1_all is None: # Initialize with the first group
                idx_1_all = idx_1  
                idx_2_all = idx_2
                group_tuple_1.append(len(idx_1)//num_conditions) # Get group 1 size for first split-half
                group_tuple_2.append(len(idx_2)//num_conditions) # Get group 1 size for second split-half
            else: # Horizontally concatenate groups
                idx_1_all = np.concatenate((idx_1_all, idx_1))  
                idx_2_all = np.concatenate((idx_2_all, idx_2))
                group_tuple_1.append(len(idx_1)//num_conditions) # Append subsequent group sizes for first split-half
                group_tuple_2.append(len(idx_2)//num_conditions) # Append subsequent group sizes for second split-half

            if pls_alg in ["mb", "cmb"]:

                # Split indices into training and testing
                idx_1 = tmp_idx_subj[:nsplit, bscan].flatten()
                idx_2 = tmp_idx_subj[nsplit:, bscan].flatten()

                # Append group indices
                if idx_1_bscan is None: # Initialize with the first group
                    idx_1_bscan = idx_1  
                    idx_2_bscan = idx_2
                else: # Horizontally concatenate groups
                    idx_1_bscan = np.concatenate((idx_1_bscan, idx_1))  
                    idx_2_bscan = np.concatenate((idx_2_bscan, idx_2))


        # Extract data for indices for each split-half
        X1 = matrix[idx_1_all, :]
        X2 = matrix[idx_2_all, :]

        # Get new cond_orders
        cond_order_1 = _get_cond_order(
                X1.shape, tuple(group_tuple_1), num_conditions
            )
        cond_order_2 = _get_cond_order(
                X2.shape, tuple(group_tuple_2), num_conditions
            )

        # Run the appropriate PLS        
        if pls_alg == "mct":
            # Mean centering
            _, X1_mc = class_functions._mean_centre(
                X1, cond_order_1, mctype=mctype
            )   
            _, X2_mc = class_functions._mean_centre(
                X2, cond_order_2, mctype=mctype
            )

            # Perform mcPLS
            my_U1, _, my_V1 = class_functions._run_pls(X1_mc)
            my_U2, _, my_V2 = class_functions._run_pls(X2_mc)
        
        if pls_alg == "rb":
            # permy = Y[np.random.permutation(n), :]
            # Y1 = permy[idx_1_all, :]
            # Y2 = permy[idx_2_all, :]
            Y1 = Y[idx_1_all, :]
            Y2 = Y[idx_2_all, :]

            # Compute correlation matrix
            R1= class_functions._compute_R(X1, Y1, cond_order_1)
            R2= class_functions._compute_R(X2, Y2, cond_order_2)

            # Perform bPLS
            my_U1, _, my_V1 = class_functions._run_pls(R1)
            my_U2, _, my_V2 = class_functions._run_pls(R2)

        if pls_alg == "cst":
            # Mean centering
            X1_mc = class_functions._mean_centre(
            X1, cond_order_1, return_means=False, mctype=mctype
        )
            X2_mc = class_functions._mean_centre(
            X2, cond_order_2, return_means=False, mctype=mctype
        )
            # Perform Contrast Task PLS 
            my_U1, _, my_V1 = class_functions._run_pls_contrast(
            X1_mc, contrasts
        )
            
            my_U2, _, my_V2 = class_functions._run_pls_contrast(
            X2_mc, contrasts
        )
        if pls_alg == "csb":
            # permy = Y[np.random.permutation(n), :]
            # Y1 = permy[idx_1_all, :]
            # Y2 = permy[idx_2_all, :] 
            Y1 = Y[idx_1_all, :]
            Y2 = Y[idx_2_all, :]
            
            # Compute correlation matrix
            R1= class_functions._compute_R(X1, Y1, cond_order_1)
            R2= class_functions._compute_R(X2, Y2, cond_order_2)

            # Perform contrast bPLS
            my_U1, _, my_V1 = class_functions._run_pls_contrast(R1, contrasts)
            my_U2, _, my_V2 = class_functions._run_pls_contrast(R2, contrasts)

        if pls_alg in ["mb","cmb"]:
            Xbscan1 = matrix[idx_1_bscan, :]
            Ybscan1 = Y[idx_1_bscan, :]
            
            Xbscan2 = matrix[idx_2_bscan, :]
            Ybscan2 = Y[idx_2_bscan, :]

            
            # Get multiblock
            multiblock1 = class_functions._create_multiblock(
            X1, cond_order_1, pls_alg, bscan, mctype,
            Xbscan = Xbscan1, Ybscan = Ybscan1)
            
            multiblock2 = class_functions._create_multiblock(
            X2, cond_order_2, pls_alg, bscan, mctype,
            Xbscan = Xbscan2, Ybscan = Ybscan2)

            if pls_alg =="mb":    
                # Perform Multiblock PLS 
                my_U1, _, my_V1 = class_functions._run_pls(multiblock1)
                my_U2, _, my_V2 = class_functions._run_pls(multiblock2)

            if pls_alg =="cmb":
                # Perform Contrast Multiblock PLS 
                my_U1, _, my_V1 = class_functions._run_pls_contrast(multiblock1, contrasts)
                my_U2, _, my_V2 = class_functions._run_pls_contrast(multiblock2, contrasts)
            
        # Append the outputs
        pls_u_repro[:, :, i] = my_V1.T @ my_V2 # Flip for consistency with matlab
        pls_v_repro[:, :, i] = my_U1.T @ my_U2 # Flip for consistency with matlab

# Generate null distribution
    for i in range(num_split):
        n_per_cond = n // num_conditions # n is the total number of conditions*subjects
        nsplit = sum(group_tuple_1) # Set split to same size as non-null split
        idx = np.random.permutation(n_per_cond)
        tmp_idx_subj = allgroup_ids[idx, :]

        idx_1_null = tmp_idx_subj[:nsplit, :].flatten()
        idx_2_null = tmp_idx_subj[nsplit:, :].flatten()

        if pls_alg in ["mb","cmb"]:
            idx_1_bscan_null = tmp_idx_subj[:nsplit, bscan].flatten()
            idx_2_bscan_null = tmp_idx_subj[nsplit:, bscan].flatten()

        if pls_alg in ["mct" ,"cst", "mb","cmb"]:
            idx_perm = np.random.permutation(n)
            permx = matrix[idx_perm, :]
        else: # "rb" or "csb"
            permx = matrix

        # Extract data for indices for each split-half
        X1_null = permx[idx_1_null, :]
        X2_null = permx[idx_2_null, :]

        # Get new cond_orders
        cond_order_1 = _get_cond_order(
                X1_null.shape, tuple(group_tuple_1), num_conditions
            )
        cond_order_2 = _get_cond_order(
                X2_null.shape, tuple(group_tuple_2), num_conditions
            )
        
        # Run the appropriate PLS
        if pls_alg == "mct":
            # Mean centering
            _, X1_mc_null = class_functions._mean_centre(
                X1_null, cond_order_1, mctype=mctype
            )          
            _, X2_mc_null = class_functions._mean_centre(
                X2_null, cond_order_2, mctype=mctype
            )

            # Perform mcPLS
            my_U1, _, my_V1 = class_functions._run_pls(X1_mc_null)
            my_U2, _, my_V2 = class_functions._run_pls(X2_mc_null)

        if pls_alg == "rb":
            permy = Y[np.random.permutation(n), :]
            Y1_null = permy[idx_1_null,:]
            Y2_null = permy[idx_2_null,:]

            # Compute correlation matrix
            R1_null= class_functions._compute_R(X1_null, Y1_null, cond_order_1)
            R2_null= class_functions._compute_R(X2_null, Y2_null, cond_order_2)

            # Perform bPLS
            my_U1, _, my_V1 = class_functions._run_pls(R1_null)
            my_U2, _, my_V2 = class_functions._run_pls(R2_null)

        if pls_alg == "cst":
            # Mean centering
            X1_mc_null = class_functions._mean_centre(
            X1_null, cond_order_1, return_means=False, mctype=mctype
        )
            X2_mc_null = class_functions._mean_centre(
            X2_null, cond_order_2, return_means=False, mctype=mctype
        )
            # Perform Contrast Task PLS 
            my_U1, _, my_V1 = class_functions._run_pls_contrast(
            X1_mc_null, contrasts
        )
            my_U2, _, my_V2 = class_functions._run_pls_contrast(
            X2_mc_null, contrasts
        )

        if pls_alg == "csb":
            permy = Y[np.random.permutation(n), :]
            Y1_null = permy[idx_1_null,:]
            Y2_null = permy[idx_2_null,:]

            # Compute correlation matrix
            R1_null= class_functions._compute_R(X1_null, Y1_null, cond_order_1)
            R2_null= class_functions._compute_R(X2_null, Y2_null, cond_order_2)

            # Perform contrast bPLS
            my_U1, _, my_V1 = class_functions._run_pls_contrast(R1_null, contrasts)
            my_U2, _, my_V2 = class_functions._run_pls_contrast(R2_null, contrasts)

        if pls_alg == "mb":
            # Get multiblock
            multiblock1_null = class_functions._create_multiblock(
            X1_null, cond_order_1, pls_alg, bscan, mctype,
            Xbscan = permx[idx_1_bscan_null,:], Ybscan = Y[idx_1_bscan_null,:])

            multiblock2_null = class_functions._create_multiblock(
            X2_null, cond_order_2, pls_alg, bscan, mctype,
            Xbscan = permx[idx_2_bscan_null,:], Ybscan = Y[idx_2_bscan_null,:]) 

            # Perform Multiblock PLS 
            my_U1, _, my_V1 = class_functions._run_pls(multiblock1_null)
            my_U2, _, my_V2 = class_functions._run_pls(multiblock2_null)

        if pls_alg =="cmb":
            # Get multiblock
            multiblock1_null = class_functions._create_multiblock(
            X1_null, cond_order_1, pls_alg, bscan, mctype,
            Xbscan = permx[idx_1_bscan_null,:], Ybscan = Y[idx_1_bscan_null,:])

            multiblock2_null = class_functions._create_multiblock(
            X2_null, cond_order_2, pls_alg, bscan, mctype,
            Xbscan = permx[idx_2_bscan_null,:], Ybscan = Y[idx_2_bscan_null,:]) 

            # Perform Contrast Multiblock PLS 
            my_U1, _, my_V1 = class_functions._run_pls_contrast(multiblock1_null, contrasts)
            my_U2, _, my_V2 = class_functions._run_pls_contrast(multiblock2_null, contrasts)

        # Append the outputs
        pls_u_null[:, :, i] = my_V1.T @ my_V2 # Flip for consistency with matlab
        pls_v_null[:, :, i] = my_U1.T @ my_U2 # Flip for consistency with matlab

    # Calculate metrics
    pls_repro = {
        "pls_rep_mean_u": [np.mean(np.abs(pls_u_repro[i, i, :])) for i in range(lv)],
        "pls_rep_mean_v": [np.mean(np.abs(pls_v_repro[i, i, :])) for i in range(lv)],
        "pls_rep_z_u": [
            np.mean(np.abs(pls_u_repro[i, i, :])) / np.std(np.abs(pls_u_repro[i, i, :]),ddof=1)
            for i in range(lv)
        ],
        "pls_rep_z_v": [
            np.mean(np.abs(pls_v_repro[i, i, :])) / np.std(np.abs(pls_v_repro[i, i, :]),ddof=1)
            for i in range(lv)
        ],
        "pls_rep_ul_u": [
            np.percentile(np.abs(pls_u_repro[i, i, :]), CI) for i in range(lv)
        ],
        "pls_rep_ll_u": [
            np.percentile(np.abs(pls_u_repro[i, i, :]), 100 - CI) for i in range(lv)
        ],
        "pls_rep_ul_v": [
            np.percentile(np.abs(pls_v_repro[i, i, :]), CI) for i in range(lv)
        ],
        "pls_rep_ll_v": [
            np.percentile(np.abs(pls_v_repro[i, i, :]), 100 - CI) for i in range(lv)
        ],
    
        "pls_null_mean_u": [np.mean(np.abs(pls_u_null[i, i, :])) for i in range(lv)],
        "pls_null_std_u": [np.std(np.abs(pls_u_null[i, i, :])) for i in range(lv)],
        "pls_null_z_u": [
            np.mean(np.abs(pls_u_null[i, i, :])) / np.std(np.abs(pls_u_null[i, i, :]),ddof=1)
            for i in range(lv)
        ],
        "pls_null_ul_u": [
            np.percentile(np.abs(pls_u_null[i, i, :]), CI) for i in range(lv)
        ],
        "pls_null_ll_u": [
            np.percentile(np.abs(pls_u_null[i, i, :]), 100 - CI) for i in range(lv)
        ],
        "pls_null_mean_v": [np.mean(np.abs(pls_v_null[i, i, :])) for i in range(lv)],
        "pls_null_std_v": [np.std(np.abs(pls_v_null[i, i, :])) for i in range(lv)],
        "pls_null_z_v": [
            np.mean(np.abs(pls_v_null[i, i, :])) / np.std(np.abs(pls_v_null[i, i, :]),ddof=1)
            for i in range(lv)
        ],
        "pls_null_ul_v": [
            np.percentile(np.abs(pls_v_null[i, i, :]), CI) for i in range(lv)
        ],
        "pls_null_ll_v": [
            np.percentile(np.abs(pls_v_null[i, i, :]), 100 - CI) for i in range(lv)
        ],
    }


    pls_repro["pls_dist_u"] = pls_u_repro
    pls_repro["pls_dist_v"] = pls_v_repro
    pls_repro["pls_dist_null_u"] = pls_u_null
    pls_repro["pls_dist_null_v"] = pls_v_null

    return pls_repro
