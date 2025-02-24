import abc

import numpy as np
import scipy
import scipy.stats
from scipy.io import loadmat
# project imports
from . import class_functions, exceptions, gsvd, resample

# import scipy.io as sio

class ResampleTest(abc.ABC):
    """Abstract base class for the ResampleTest class set. Forces existence
    of certain functions.
    """

    _subclasses = {}
    pls_alg = None
    # _algs_larger_lvcorr = {"mb", "cmb"}

    # maps abbreviated user-specified classnames to full PLS variant names
    _pls_types = {
        "mct": "Mean-Centering Task PLS",
        # "mct_mg": "Mean-Centering Task PLS - Multi-Group",
        "cst": "Contrast Task PLS",
        "rb": "Regular Behaviour PLS",
        "mb": "Multiblock PLS",
        "csb": "Contrast Behaviour PLS",
        "cmb": "Contrast Multiblock PLS",
    }

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    # register valid decorated PLS/resample method as a subclass of
    # ResampleTest
    @classmethod
    def _register_subclass(cls, pls_method):
        def decorator(subclass):
            cls._subclasses[pls_method] = subclass
            return subclass

        return decorator

    # instantiate and return valid registered PLS method specified by user
    @classmethod
    def _create(cls, pls_method, *args, **kwargs):
        if pls_method not in cls._subclasses and pls_method in cls._pls_types:
            raise exceptions.NotImplementedError(
                f"Specified PLS/Resample method {cls._pls_types[pls_method]} "
                "has not yet been implemented."
            )
        elif pls_method not in cls._subclasses:
            raise ValueError(f"Invalid PLS/Resample method {pls_method}")
        cls.pls_alg = pls_method
        return cls._subclasses[pls_method](*args, **kwargs)


@ResampleTest._register_subclass("mct")
@ResampleTest._register_subclass("rb")
@ResampleTest._register_subclass("cst")
@ResampleTest._register_subclass("csb")
@ResampleTest._register_subclass("mb")
@ResampleTest._register_subclass("cmb")
class _ResampleTestTaskPLS(ResampleTest):
    """Class that runs permutation and bootstrap tests for Task PLS. When run,
    this class generates fields for permutation test information
    (permutation ratio, etc.) and for bootstrap test informtaion (confidence
    intervals, standard errors, bootstrap ratios, etc.).

    Parameters
    ----------
    X : np.array
        Input neural matrix/matrices for use with PLS. This matrix is passed
        in from the PLS class.
    U : np.array
        Left singular vectors of X.
    s : np.array
        Singular values for X. Used to compute permutation ratio.
    V : np.array
        Right singular vectors of X.
    cond_order : array-like
        Order vector(s) for conditions in X.
    preprocess : function, optional
        Preprocessing function used prior to running GSVD on X in
        PLS class. Used to preprocess resampled matrices in boostrap/
        permutation tests.
    nperm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 1000.
    nboot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 1000.
    nonrotated : boolean, optional
        Not implememted yet.
    dist : 2-tuple of floats, optional
        Distribution values used for calculating the confidence interval in
        the bootstrap test. Defaults to (0.05, 0.95).

    Attributes
    ----------
    dist : 2-tuple of floats, optional
        Distribution values used for calculating the confidence interval in
        the bootstrap test. Defaults to (0.05, 0.95).
    permute_ratio : float
        Ratio of resampled values greater than observed values, divided by
        the number of iterations in the permutation test. A higher ratio
        indicates a higher level of randomness.
    conf_ints : 2-tuple of np.arrays
        Upper and lower element-wise confidence intervals for the resampled
        left singular vectors in a tuple.
    std_errs : np.array
        Element-wise standard errors for the resampled right singular vectors.
    boot_ratios : np.array
        NumPy array containing element-wise ratios of

    """

    def __init__(
        self,
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        mctype,
        contrast=None,
        preprocess=None,
        nperm=1000,
        nboot=1000,
        dist=(0.05, 0.95),
        rotate_method=0,
    ):
        self.dist = dist

        print(f"PLS ALG: {self.pls_alg}")
        if nperm > 0:
            self.permute_ratio, self.perm_debug_dict = self._permutation_test(
                X,
                Y,
                U,
                s,
                V,
                cond_order,
                mctype,
                nperm,
                self.pls_alg,
                preprocess=preprocess,
                rotate_method=rotate_method,
                contrast=contrast,
            )

        if nboot > 0:
            if self.pls_alg in ["rb", "csb"]:
                (
                    self.conf_ints,
                    self.std_errs,
                    self.boot_ratios,
                    self.LVcorr,
                    #self.llcorr,
                    #self.ulcorr,
                    self.boot_debug_dict,
                ) = self._bootstrap_test(
                    X,
                    Y,
                    U,
                    s,
                    V,
                    cond_order,
                    mctype,
                    nboot,
                    self.pls_alg,
                    preprocess=preprocess,
                    rotate_method=rotate_method,
                    dist=self.dist,
                    contrast=contrast,
                )
            elif self.pls_alg in ["mb", "cmb"]:
                (
                    self.conf_ints,
                    self.conf_ints_T,
                    self.std_errs,
                    self.boot_ratios,
                    self.LVcorr,
                    self.boot_debug_dict,
                ) = self._bootstrap_test(
                    X,
                    Y,
                    U,
                    s,
                    V,
                    cond_order,
                    mctype,
                    nboot,
                    self.pls_alg,
                    preprocess=preprocess,
                    rotate_method=rotate_method,
                    dist=self.dist,
                    contrast=contrast,
                )
            else:
                (
                    self.conf_ints,
                    self.std_errs,
                    self.boot_ratios,
                    self.boot_debug_dict,
                ) = self._bootstrap_test(
                    X,
                    Y,
                    U,
                    s,
                    V,
                    cond_order,
                    mctype,
                    nboot,
                    self.pls_alg,
                    preprocess=preprocess,
                    rotate_method=rotate_method,
                    dist=self.dist,
                    contrast=contrast,
                )

    @staticmethod
    def _permutation_test(
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        mctype,
        niter,
        pls_alg,
        preprocess=None,
        contrast=None,
        rotate_method=0,
        threshold=1e-12,
    ):
        """Run permutation test on X. Resamples X (without replacement) based
        on condition order, runs PLS on resampled matrix, and computes the
        element-wise permutation ratio ((number of times permutation > observation)/`niter`.
        """
        # if ngroups > 1:
        #     raise exceptions.NotImplementedError(
        #         "Multi-group MCT-PLS is not yet implemented."
        #     )

        # singvals = np.empty((s.shape[0], niter))
        greatersum = np.zeros(s.shape)
        s[np.abs(s) < threshold] = 0
        debug = True
        debug_dict = {}
        indices = np.empty((niter, X.shape[0]))
        if debug:
            sum_perm = np.empty(niter)
            sum_s = np.empty(niter)
            s_list = np.empty((niter, s.shape[0]))


        if pls_alg in ["mb", "cmb"]:
            mb_datamat_notnormed = preprocess(
                    X, Y, cond_order, mctype, norm_opt = False
                )
            total_s = np.sum(np.power(mb_datamat_notnormed, 2))
            per_orig = np.power(s,2) / np.sum(np.power(s,2))
            org_s = np.sqrt(per_orig * total_s)

        print("----Running Permutation Test----\n")
        for i in range(niter):
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")
            # create resampled X matrix and get resampled indices

            if pls_alg in ["mct", "cst"]:
                X_new, inds = resample.resample_without_replacement(
                    X, cond_order, return_indices=True, pls_alg=pls_alg
                )

            if pls_alg in ["rb", "csb"]:
                Y_new, inds = resample.resample_without_replacement(
                    Y, cond_order, return_indices=True, pls_alg=pls_alg
                )

            if pls_alg in ["mb", "cmb"]:
                X_new_T, inds = resample.resample_without_replacement(
                    X, cond_order, return_indices=True, pls_alg=pls_alg
                )
                # Permute behavioural data (use "rb" option)
                Y_new, inds = resample.resample_without_replacement(Y, cond_order,pls_alg="rb",return_indices=True) # to do: handle bscan
                    
            indices[i] = inds

            # inds = loadmat("BSAMP.mat")
            # inds = inds["BSAMP"][:,i] -1
            # #X_new = X[inds,:]
            # X_new = X
            # Y_new = Y[inds,:]
            # inds = loadmat("TSAMP.mat") 
            # inds = inds["TSAMP"][:,i] -1
            # X_new_T = X[inds,:]

        
            # pass in preprocessing function (i.e. mean-centering) for use
            # after sampling

            if pls_alg in ["mct"]:
                permuted = preprocess(
                    X_new, cond_order, mctype, return_means=False
                )
            if pls_alg in ["cst"]:
                permuted = class_functions._get_group_condition_means(X_new, cond_order)
            
            if pls_alg in ["mb", "cmb"]:
                permuted = preprocess(X, Y_new, cond_order, pls_alg, mctype, XT_provided=X_new_T) 

            if pls_alg in ["rb", "csb"]:
                    permuted = preprocess(X, Y_new, cond_order)
                    #print(f"permuted: {permuted}")
            if debug:
                sum_perm[i] = np.sum(np.power(permuted, 2))


            if rotate_method == 0:
                if contrast is None:
                    VS_hat = permuted.T @ U
                    s_hat = np.sqrt(np.sum(VS_hat**2, axis = 0))
                    #print(s_hat)
                else:
                    inpt = contrast.T @ permuted
                    # s_hat = np.linalg.svd(contrast.T @ permuted, compute_uv=False)
                    s_hat = np.linalg.svd(inpt, compute_uv=False)
                # print(s_hat)
            # elif rotate_method == 1:
            #     # U_hat, s_hat, V_hat = gsvd.gsvd(permuted)
            #     if contrast is not None:
            #         U_hat, s_hat, V_hat = class_functions._run_pls_contrast(
            #             permuted, contrast
            #         )
            #     else:
            #         U_hat, s_hat, V_hat = np.linalg.svd(
            #             permuted, full_matrices=False
            #         )
            #         V_hat = V_hat.T
            #     # procustes
            #     # U_bar, s_bar, V_bar = gsvd.gsvd(V.T @ V_hat)
            #     # U_bar, s_bar, V_bar = np.linalg.svd(V.T @ V_hat, full_matrices=False)
            #     U_bar, s_bar, V_bar = np.linalg.svd(
            #         U.T @ U_hat, full_matrices=False
            #     )
            #     V_bar = V_bar.T
            #     # print(X_new_mc.shape)
            #     rot = V_bar @ U_bar.T
            #     U_rot = (U_hat * s_hat) @ rot
            #     # permuted_rot = permuted @ V_rot
            #     # permuted_rot = U_rot.T @ permuted
            #     # s_rot = np.sqrt(np.sum(np.power(permuted_rot.T, 2), axis=0))
            #     s_rot = np.sqrt(np.sum(np.power(U_rot, 2), axis=0))
            #     s_hat = np.copy(s_rot)
            #     # print(s_hat)
            # elif rotate_method == 2:
            #     # use derivation equations to compute permuted singular values
            #     if pls_alg in ["cst", "csb", "cmb"]:
            #         s_hat = class_functions._run_pls_contrast(
            #             permuted, contrast, compute_uv=False
            #         )
            #     else:
            #         US_hat = permuted.T @ U
            #         s_hat = np.sqrt(np.sum(np.power(US_hat, 2), axis=0))

            #     # U_hat_, s_hat_, V_hat_ = gsvd.gsvd(X_new_mc)

            #     # gd = [float("{:.5f}".format(i)) for i in s_hat_]
            #     # der = [float("{:.5f}".format(i)) for i in s_hat]

            #     # print(f"GSVD: {gd}")
            #     # print(f"Derived: {der}")

            #     # U_hat = US_hat / s_hat
            #     # V_hat = np.linalg.inv(np.diag(s_hat)) @ (U.T @ X_new_mc)
            #     # print(s_hat)
            else:
                raise exceptions.NotImplementedError(
                    f"Specified rotation method ({rotate_method}) "
                    "has not been implemented."
                )

            # insert s_hat into singvals tracking matrix
            # singvals[:, i] = s_hat
            # count number of times sampled singular values are
            # greater than observed singular values, element-wise
            # greatersum += s >= s_hat
            # print(s_hat >= s)

            if pls_alg in ["mb"]:
                mb_permdatamat_notnormed = preprocess(
                    X, Y_new, cond_order, mctype, XT_provided=X_new_T, norm_opt = False
                )
                total_s_hat = np.sum(np.power(mb_permdatamat_notnormed, 2))

                squared_diag = np.diag(s_hat ** 2)
                per_hat_norm = np.diag(np.diag(squared_diag)**2) / np.sum(np.diag(squared_diag)**2)
                per_hat = np.diag(per_hat_norm)
                s_hat = np.sqrt(per_hat * total_s_hat)

                # Compare
                greatersum += s_hat >= org_s

            if pls_alg in ["cst", "csb", "cmb"]:
                contrast_normed = class_functions._normalize(contrast)
                crossblock =  contrast_normed.T @ permuted 
                s_hat = np.sqrt(np.sum(crossblock**2, axis=1)) 
                greatersum += s_hat >= s

            if pls_alg in ["rb", "mct"]:
                s_hat[np.abs(s_hat) < threshold] = 0
                greatersum += s_hat >= s
                if debug:
                    s_list[
                        i:,
                    ] = s_hat
                    sum_s[i] = np.sum(np.power(s_hat, 2))

            permute_ratio = greatersum / (niter + 1)


        print(f"real s: {s}")
        print(f"ratio: {permute_ratio}")
        if debug:
            debug_dict["s_list"] = s_list
            debug_dict["sum_s"] = sum_perm
            debug_dict["sum_perm"] = sum_s
            debug_dict["indices"] = indices
            # debug_dict[""] =
            return (permute_ratio, debug_dict)
        return permute_ratio

    @staticmethod
    def _bootstrap_test(
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        mctype,
        niter,
        pls_alg,
        preprocess=None,
        rotate_method=0,
        dist=(0.05, 0.95),
        contrast=None,
    ):
        """Runs a bootstrap estimation on X matrix. Resamples X with
        replacement according to the condition order, runs PLS on the
        resampled X matrices, and computes `conf_int`, `std_errs`, and
        `boot_ratios`.
        """
        debug = True
        debug_dict = {}

        # allocate memory for sampled values
        left_sv_sampled = np.empty((niter, U.shape[0], U.shape[1]))
        right_sv_sampled = np.empty((niter, V.shape[0], V.shape[1]))
        indices = np.empty((niter, X.shape[0]))

        # m_inds = sio.loadmat("/home/nfrazier-logue/matlab/samps.mat")["x"].T - 1
        # print(f"MATLAB SHAPE: {m_inds.shape}")

        if pls_alg in ["mct","cst"]:
            Tdistrib = np.empty((niter, U.shape[0], U.shape[1]))
        else:

            if pls_alg in ["rb"]:
                ncols = np.product(cond_order.shape) * Y.shape[1]

            if pls_alg in ["csb"]:
                ncols = contrast.shape[1]

            if pls_alg in ["mb", "cmb"]:
                ncols = U.shape[0]
                left_sv_sampled = np.empty((niter,np.product(cond_order.shape) * Y.shape[1],ncols))
                Tdistrib = np.empty((niter, np.product(cond_order.shape) * Y.shape[1], ncols,))

            LVcorr = np.empty((niter, np.product(cond_order.shape) * Y.shape[1], ncols,))

        print("----Running Bootstrap Test----\n")
        for i in range(niter):
            # print out iteration number every 50 iterations
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")


            if pls_alg in ["mb", "cmb"]:
                # X_new_T = Task portion
                X_new_T = resample.resample_with_replacement(
                    X, cond_order, return_indices=False
                )
                # X_new = Behaviour portion
                X_new,inds = resample.resample_with_replacement(
                    X, cond_order, return_indices=True
                ) # to do: handling for bscan
            else:
            # return indices to use with Y_new
                X_new, inds = resample.resample_with_replacement(
                    X, cond_order, return_indices=True
                )

            indices[i] = inds

            if Y is not None:
                Y_new = Y[inds, :]

        #     # TESTING WITH MATLAB
        #     inds = loadmat("TSAMP.mat") 
        #     inds = inds["TSAMP"][:,i] -1
        #     X_new_T = X[inds,:]
        #     inds = loadmat("BSAMP.mat")
        #     inds = inds["BSAMP"][:,i] -1
        #     Y_new = Y[inds,:]
        #     X_new = X[inds,:]
        #     # TESTING WITH MATLAB

            # pass in preprocessing function (e.g. mean-centering) for use
            # after sampling

            if pls_alg in ["mct"]:
                permuted = preprocess(
                    X_new, cond_order, mctype, return_means=False
                )
            elif pls_alg in ["cst"]:
                permuted = class_functions._get_group_condition_means(X_new, cond_order)
            else:
                if pls_alg in ["mb", "cmb"]:
                    permuted = preprocess(X_new, Y_new, cond_order, pls_alg, mctype,XT_provided=X_new_T)
                else:
                    permuted = preprocess(X_new, Y_new, cond_order)

            if rotate_method == 0:
                #Get U
                U_hat = (np.dot(V.T, permuted.T)).T
                
                #Get VS
                VS_hat = permuted.T @ U
               
                # Get V - normalize VS_hat
                V_hat = class_functions._normalize(VS_hat)

            # elif rotate_method == 1:
            #     # U_hat, s_hat, V_hat = gsvd.gsvd(permuted)
            #     U_hat, s_hat, V_hat = np.linalg.svd(
            #         permuted, full_matrices=False
            #     )
            #     V_hat = V_hat.T
            #     # procustes
            #     # U_bar, s_bar, V_bar = gsvd.gsvd(V.T @ V_hat)
            #     # U_bar, s_bar, V_bar = np.linalg.svd(V.T @ V_hat, full_matrices=False)
            #     U_bar, s_bar, V_bar = np.linalg.svd(
            #         U.T @ U_hat, full_matrices=False
            #     )
            #     # s_pro = np.sqrt(np.sum(np.power(V_bar, 2), axis=0))
            #     # print(X_new_mc.shape)
            #     # rot = U_bar @ V.T
            #     # V_rot = V_hat.T @ rot.T

            #     rot = V_bar @ U_bar.T
            #     U_rot = (U_hat * s_hat) @ rot
            #     # permuted_rot = permuted @ V_rot
            #     permuted_rot = U_rot @ permuted
            #     s_rot = np.sqrt(np.sum(np.power(permuted_rot.T, 2), axis=0))
            #     s_hat = np.copy(s_rot)

            # elif rotate_method == 2:
            #     # use derivation equations to compute permuted singular values
            #     # US_hat = X_new_mc @ V
            #     VS_hat = permuted.T @ U
            #     s_hat = np.sqrt(np.sum(np.power(VS_hat, 2), axis=0))
            #     # US_hat = V.T @ permuted.T
            #     # s_hat = np.sqrt(np.sum(np.power(US_hat, 2), axis=0))
            #     V_hat_der = VS_hat / s_hat
            #     U_hat = (
            #         np.linalg.inv(np.diag(s_hat)) @ (V_hat_der.T @ permuted.T)
            #     ).T
            #     # V_hat = (X_new_mc.T @ U_hat_der) / s_hat
            #     # potential fix for sign issues
            #     V_hat = V_hat_der
            #     # U_hat = (X_new_mc @ V_hat) / s_hat
            #     # U_hat_, s_hat_, V_hat_ = gsvd.gsvd(X_new_mc)

            #     # print("DERIVED\n")
            #     # print(U_hat_der)
            #     # print("=====================")
            #     # print("DOUBLE DERIVED\n")
            #     # print(s_hat)
            #     # print("----------------------")
            #     # print(s_hat_)
            #     # print("++++++++++++++++++++++")
            else:
                raise exceptions.NotImplementedError(
                    f"Specified rotation method ({rotate_method}) "
                    "has not been implemented."
                )
            
            # assign right singular vector
            right_sv_sampled[i] = VS_hat

            # insert left singular vector into tracking np.array
            if pls_alg in ["mct"]:
                # Task PLS
                left_sv_sampled[i] = U_hat
                # Compute Tdistrib
                tmp_Tdistrib =  X @ V_hat
                Tdistrib[i] = class_functions._get_group_condition_means(tmp_Tdistrib, cond_order)

            if pls_alg in ["rb"]:
                # Behavioural PLS: compute X latents for use in correlation computation
                X_hat_latent = class_functions._compute_X_latents(X_new, V_hat)
                LVcorr[i] = class_functions._compute_corr(
                    X_hat_latent, Y_new, cond_order
                )
                left_sv_sampled[i] = LVcorr[i]

            if pls_alg in ["mb"]:
            # Regular Multi-block PLS
                    # Behaviour X_latents
                    B_X_hat_latent = class_functions._compute_X_latents(X_new, V_hat)
  
                    # Compute LVcorr (bcorr)
                    LVcorr[i] = class_functions._compute_corr(B_X_hat_latent, Y_new, cond_order)
                    left_sv_sampled[i] = LVcorr[i]

                    # Compute Tdistrib
                    smeanmat = resample._calculate_smeanmat(X_new_T, cond_order, mctype)
                    tmp_Tdistrib =  smeanmat @ V_hat
                    Tdistrib[i] = class_functions._get_group_condition_means(tmp_Tdistrib, cond_order)

            if contrast is None:
                contrast_normed = class_functions._normalize(contrast)
                crossblock =  contrast_normed.T @ permuted
                norm_crossblock = class_functions._normalize(crossblock.T)

                if pls_alg in ["cmb", "cst"]:
                # Contrast Multi-block - task portion & Contrast Task PLS
                        tmp_Tdistrib = X @ norm_crossblock
                        Tdistrib[i] = class_functions._get_group_condition_means(tmp_Tdistrib, cond_order)

                if pls_alg in ["cmb", "csb"]:
                # Contrast Multi-block - behaviour portion & Contrast Behaviour PLS
                        # Behaviour X_latents
                        B_X_hat_latent = class_functions._compute_X_latents(X_new, norm_crossblock)

                        # Compute LVcorr (bcorr)
                        LVcorr[i] = class_functions._compute_corr(B_X_hat_latent, Y_new, cond_order)
                        left_sv_sampled[i] = LVcorr[i]


        # compute confidence intervals
        if pls_alg in ["mct","cst"]:
            # Task PLS (ulusc & llusc in matlab)
            conf_int = resample.confidence_interval(
                Tdistrib, conf=dist)
        else:
            # Behavioural PLS CI (ulcorr & llcorr in matlab)
            conf_int = resample.confidence_interval(
                left_sv_sampled, conf=dist
            )

            if pls_alg in ["mb", "cmb"]:
            # Multi-block Task CI (ulusc & llusc in matlab)
                conf_int_T = resample.confidence_interval(
                Tdistrib, conf=dist
                )      
            
        # compute standard error
        std_errs = np.std(right_sv_sampled, axis=0)
        
        # compute bootstrap ratios
        if contrast is None:
            boot_ratios = np.divide(V * s,std_errs)
        else:
            boot_ratios = np.divide(V, std_errs)

        # TODO: find more elegant solution to returning arbitrary # of vals
        # maybe tokenizing a dictionary?

        if debug:
            debug_dict["left_sv_sampled"] = left_sv_sampled
            debug_dict["right_sv_sampled"] = right_sv_sampled
            #debug_dict["left_grand_mean"] = left_grand_mean
            debug_dict["indices"] = indices


        if pls_alg in ["rb", "csb"]:
            return (
                conf_int,
                std_errs,
                boot_ratios,
                LVcorr,
                #llcorr,
                #ulcorr,
                debug_dict,
            )
        elif pls_alg in ["mb", "cmb"]:
            return (
                conf_int,
                conf_int_T,
                std_errs,
                boot_ratios,
                LVcorr,
                #llcorr,
                #ulcorr,
                debug_dict,
            )
        else:
            return (conf_int, std_errs, boot_ratios, debug_dict)
        # if Y is None:
        #     return (conf_int, std_errs, boot_ratios, debug_dict)
        # else:
        #     llcorr, ulcorr = resample.confidence_interval(LVcorr, conf=dist)
        #     return (
        #         conf_int,
        #         std_errs,
        #         boot_ratios,
        #         LVcorr,
        #         llcorr,
        #         ulcorr,
        #         debug_dict,
        #     )
    
    def __repr__(self):
        stg = ""
        stg += "Permutation Test Results\n"
        stg += "------------------------\n\n"
        stg += f"Ratio: {self.permute_ratio}\n\n"
        stg += "Bootstrap Test Results\n"
        stg += "----------------------\n\n"
        stg += f"Element-wise Confidence Interval: {self.dist}\n"
        stg += "\nLower CI: \n"
        stg += str(self.conf_ints[0])
        stg += "\n\nUpper CI: \n"
        stg += str(self.conf_ints[1])
        # Add conf_ints_T if available
        if self.pls_alg in ["mb", "cmb"]:
            stg += "\n\nLower CI (Task): \n"
            stg += str(self.conf_ints_T[0])
            stg += "\n\nUpper CI (Task): \n"
            stg += str(self.conf_ints_T[1])
        stg += "\n\nStandard Errors:\n"
        stg += str(self.std_errs)
        stg += "\n\nBootstrap Ratios:\n"
        stg += str(self.boot_ratios)

        return stg

    def __str__(self):
        stg = ""
        stg += "Permutation Test Results\n"
        stg += "------------------------\n\n"
        stg += f"Ratio: {self.permute_ratio}\n\n"
        stg += "Bootstrap Test Results\n"
        stg += "----------------------\n\n"
        stg += f"Element-wise Confidence Interval: {self.dist}\n"
        stg += "\nLower CI: \n"
        stg += str(self.conf_ints[0])
        stg += "\n\nUpper CI: \n"
        stg += str(self.conf_ints[1])
        # Add conf_ints_T if available
        if self.pls_alg in ["mb", "cmb"]:
            stg += "\n\nLower CI (Task): \n"
            stg += str(self.conf_ints_T[0])
            stg += "\n\nUpper CI (Task): \n"
            stg += str(self.conf_ints_T[1])
        stg += "\n\nStandard Errors:\n"
        stg += str(self.std_errs)
        stg += "\n\nBootstrap Ratios:\n"
        stg += str(self.boot_ratios)
        return stg
