import abc

import numpy as np
import scipy
import scipy.stats

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
        contrast=None,
        preprocess=None,
        nperm=1000,
        nboot=1000,
        dist=(0.05, 0.95),
        rotate_method=0,
        mctype=0,
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
                nperm,
                self.pls_alg,
                preprocess=preprocess,
                rotate_method=rotate_method,
                mctype=mctype,
                contrast=contrast,
            )

        if nboot > 0:
            if Y is not None:
                (
                    self.conf_ints,
                    self.std_errs,
                    self.boot_ratios,
                    self.LVcorr,
                    self.llcorr,
                    self.ulcorr,
                    self.boot_debug_dict,
                ) = self._bootstrap_test(
                    X,
                    Y,
                    U,
                    s,
                    V,
                    cond_order,
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
                    nboot,
                    self.pls_alg,
                    preprocess=preprocess,
                    rotate_method=rotate_method,
                    mctype=mctype,
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
        niter,
        pls_alg,
        preprocess=None,
        contrast=None,
        rotate_method=0,
        mctype=0,
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

        print("----Running Permutation Test----\n")
        for i in range(niter):
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")
            # create resampled X matrix and get resampled indices

            X_new, inds = resample.resample_without_replacement(
                X, cond_order, return_indices=True
            )
            indices[i] = inds

            if Y is not None:
                Y_new = Y[inds, :]
                # Y_new = resample.resample_without_replacement(Y, cond_order)

            # pass in preprocessing function (i.e. mean-centering) for use
            # after sampling

            if Y is None:
                permuted = preprocess(
                    X_new, cond_order, mctype=mctype, return_means=False
                )

            else:
                permuted = preprocess(X_new, Y_new, cond_order)

            if debug:
                sum_perm[i] = np.sum(np.power(permuted, 2))

            # print(f"permuted shape: {permuted.shape}")

            if rotate_method == 0:
                # run GSVD on mean-centered, resampled matrix
                # U_hat, s_hat, V_hat = gsvd.gsvd(permuted)

                # s_hat = gsvd.gsvd(permuted, compute_uv=False)
                if contrast is None:
                    s_hat = np.linalg.svd(permuted, compute_uv=False)
                    # print(f"s_hat shape: {s_hat.shape}\n")
                else:
                    inpt = contrast.T @ permuted
                    # s_hat = np.linalg.svd(contrast.T @ permuted, compute_uv=False)
                    s_hat = np.linalg.svd(inpt, compute_uv=False)
                # print(s_hat)
            elif rotate_method == 1:
                # U_hat, s_hat, V_hat = gsvd.gsvd(permuted)
                if contrast is not None:
                    U_hat, s_hat, V_hat = class_functions._run_pls_contrast(
                        permuted, contrast
                    )
                else:
                    U_hat, s_hat, V_hat = np.linalg.svd(
                        permuted, full_matrices=False
                    )
                    V_hat = V_hat.T
                # procustes
                # U_bar, s_bar, V_bar = gsvd.gsvd(V.T @ V_hat)
                # U_bar, s_bar, V_bar = np.linalg.svd(V.T @ V_hat, full_matrices=False)
                U_bar, s_bar, V_bar = np.linalg.svd(
                    U.T @ U_hat, full_matrices=False
                )
                V_bar = V_bar.T
                # print(X_new_mc.shape)
                rot = V_bar @ U_bar.T
                U_rot = (U_hat * s_hat) @ rot
                # permuted_rot = permuted @ V_rot
                # permuted_rot = U_rot.T @ permuted
                # s_rot = np.sqrt(np.sum(np.power(permuted_rot.T, 2), axis=0))
                s_rot = np.sqrt(np.sum(np.power(U_rot, 2), axis=0))
                s_hat = np.copy(s_rot)
                # print(s_hat)
            elif rotate_method == 2:
                # use derivation equations to compute permuted singular values
                if pls_alg in ["cst", "csb", "cmb"]:
                    s_hat = class_functions._run_pls_contrast(
                        permuted, contrast, compute_uv=False
                    )
                else:
                    US_hat = permuted.T @ U
                    s_hat = np.sqrt(np.sum(np.power(US_hat, 2), axis=0))

                # U_hat_, s_hat_, V_hat_ = gsvd.gsvd(X_new_mc)

                # gd = [float("{:.5f}".format(i)) for i in s_hat_]
                # der = [float("{:.5f}".format(i)) for i in s_hat]

                # print(f"GSVD: {gd}")
                # print(f"Derived: {der}")

                # U_hat = US_hat / s_hat
                # V_hat = np.linalg.inv(np.diag(s_hat)) @ (U.T @ X_new_mc)
                # print(s_hat)
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
            s_hat[np.abs(s_hat) < threshold] = 0
            # greatersum += s_hat >= np.mean(s)
            greatersum += s_hat >= s
            if debug:
                s_list[
                    i:,
                ] = s_hat
                sum_s[i] = np.sum(np.power(s_hat, 2))

        permute_ratio = greatersum / niter

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
        niter,
        pls_alg,
        preprocess=None,
        rotate_method=0,
        mctype=0,
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
        # left_sv_sampled = np.empty((niter, U.shape[0], U.shape[1]))
        left_sv_sampled = np.empty((niter, X.shape[0], U.shape[1]))
        right_sv_sampled = np.empty((niter, V.shape[0], V.shape[1]))
        indices = np.empty((niter, X.shape[0]))

        # m_inds = sio.loadmat("/home/nfrazier-logue/matlab/samps.mat")["x"].T - 1
        # print(f"MATLAB SHAPE: {m_inds.shape}")

        if Y is not None:
            # LVcorr = np.empty((niter, Y.shape[0], V.shape[1]))
            # change LVcorr column dimension if using multi-block
            ncols = np.product(cond_order.shape) * Y.shape[1]
            if pls_alg in ["mb", "cmb"]:
                ncols = X.shape[1]

            LVcorr = np.empty(
                (
                    niter,
                    np.product(cond_order.shape) * Y.shape[1],
                    # np.product(cond_order.shape) * Y.shape[1],
                    ncols,
                )
            )

        # right_sum = np.zeros(X.shape[1], X.shape[1])
        # right_squares = np.zeros(X.shape[1], X.shape[1])
        print("----Running Bootstrap Test----\n")
        for i in range(niter):
            # print out iteration number every 50 iterations
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")

            # also return indices to use with Y_new
            X_new, inds = resample.resample_with_replacement(
                X, cond_order, return_indices=True
            )

            # X_new = X[m_inds[i], :]

            indices[i] = inds
            # indices[i] = m_inds[i]

            if Y is not None:
                Y_new = Y[inds, :]
                # Y_new = Y[m_inds[i], :]

            # pass in preprocessing function (e.g. mean-centering) for use
            # after sampling

            if Y is None:
                permuted = preprocess(
                    X_new, cond_order, mctype=mctype, return_means=False
                )

            else:
                permuted = preprocess(X_new, Y_new, cond_order)

            # if Y is None:
            #     X_new_means, X_new_mc = preprocess(
            #         X_new, cond_order=cond_order
            #     )  # , ngroups=ngroups)

            if rotate_method == 0:
                # run GSVD on mean-centered, resampled matrix

                # U_hat, s_hat, V_hat = gsvd.gsvd(permuted)
                U_hat, s_hat, V_hat = np.linalg.svd(
                    permuted, full_matrices=False
                )
                V_hat = V_hat.T
            elif rotate_method == 1:
                # U_hat, s_hat, V_hat = gsvd.gsvd(permuted)
                U_hat, s_hat, V_hat = np.linalg.svd(
                    permuted, full_matrices=False
                )
                V_hat = V_hat.T
                # procustes
                # U_bar, s_bar, V_bar = gsvd.gsvd(V.T @ V_hat)
                # U_bar, s_bar, V_bar = np.linalg.svd(V.T @ V_hat, full_matrices=False)
                U_bar, s_bar, V_bar = np.linalg.svd(
                    U.T @ U_hat, full_matrices=False
                )
                # s_pro = np.sqrt(np.sum(np.power(V_bar, 2), axis=0))
                # print(X_new_mc.shape)
                # rot = U_bar @ V.T
                # V_rot = V_hat.T @ rot.T

                rot = V_bar @ U_bar.T
                U_rot = (U_hat * s_hat) @ rot
                # permuted_rot = permuted @ V_rot
                permuted_rot = U_rot @ permuted
                s_rot = np.sqrt(np.sum(np.power(permuted_rot.T, 2), axis=0))
                s_hat = np.copy(s_rot)

            elif rotate_method == 2:
                # use derivation equations to compute permuted singular values
                # US_hat = X_new_mc @ V
                VS_hat = permuted.T @ U
                s_hat = np.sqrt(np.sum(np.power(VS_hat, 2), axis=0))
                # US_hat = V.T @ permuted.T
                # s_hat = np.sqrt(np.sum(np.power(US_hat, 2), axis=0))
                V_hat_der = VS_hat / s_hat
                U_hat = (
                    np.linalg.inv(np.diag(s_hat)) @ (V_hat_der.T @ permuted.T)
                ).T
                # V_hat = (X_new_mc.T @ U_hat_der) / s_hat
                # potential fix for sign issues
                V_hat = V_hat_der
                # U_hat = (X_new_mc @ V_hat) / s_hat
                # U_hat_, s_hat_, V_hat_ = gsvd.gsvd(X_new_mc)

                # print("DERIVED\n")
                # print(U_hat_der)
                # print("=====================")
                # print("DOUBLE DERIVED\n")
                # print(s_hat)
                # print("----------------------")
                # print(s_hat_)
                # print("++++++++++++++++++++++")
            else:
                raise exceptions.NotImplementedError(
                    f"Specified rotation method ({rotate_method}) "
                    "has not been implemented."
                )

            # insert left singular vector into tracking np.array
            # print(f"dst: {right_sv_sampled[i].shape}; src: {V_hat.shape}")
            # left_sv_sampled[i] = U_hat * s_hat
            left_sv_sampled[i] = class_functions._compute_X_latents(
                X_new, V_hat
            )
            right_sv_sampled[i] = V_hat * s_hat
            if Y is not None:
                # compute X latents for use in correlation computation
                X_hat_latent = class_functions._compute_X_latents(X_new, V_hat)
                # print(f"XHL shape: {X_hat_latent.shape}")
                # print(f"U shape: {U.shape}")
                # print(f"V shape: {V.shape}")
                # print(f"U_hat shape: {U_hat.shape}")
                # print(f"V_hat shape: {V_hat.shape}")
                LVcorr[i] = class_functions._compute_corr(
                    X_hat_latent, Y_new, cond_order
                )
                # LVcorr[:, 1:] = LVcorr[:, 1:] * -1  # temp sign change fix
                # LVcorr[:, 0] = np.abs(LVcorr[:, 0])
            # right_sum += V_hat
            # right_squares += np.power(V_hat, 2)

        # compute confidence intervals of U sampled

        # compute iteration-wise, then column-wise means to compute
        # grand mean of
        left_grand_mean = np.mean(np.mean(left_sv_sampled, axis=0), axis=0)

        conf_int = resample.confidence_interval(
            left_sv_sampled - left_grand_mean, conf=dist
        )

        # compute standard error of left singular vector
        std_errs = scipy.stats.sem(right_sv_sampled, axis=0)
        # compute bootstrap ratios
        boot_ratios = np.divide(std_errs, V)
        # TODO: find more elegant solution to returning arbitrary # of vals
        # maybe tokenizing a dictionary?

        if debug:
            debug_dict["left_sv_sampled"] = left_sv_sampled
            debug_dict["right_sv_sampled"] = right_sv_sampled
            debug_dict["left_grand_mean"] = left_grand_mean
            debug_dict["indices"] = indices

        if Y is None:
            return (conf_int, std_errs, boot_ratios, debug_dict)
        else:
            llcorr, ulcorr = resample.confidence_interval(LVcorr, conf=dist)
            return (
                conf_int,
                std_errs,
                boot_ratios,
                LVcorr,
                llcorr,
                ulcorr,
                debug_dict,
            )

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
        stg += "\n\nStandard Errors:\n"
        stg += str(self.std_errs)
        stg += "\n\nBootstrap Ratios:\n"
        stg += str(self.boot_ratios)
        return stg
