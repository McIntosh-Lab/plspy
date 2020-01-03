import abc
import numpy as np
import scipy
import scipy.stats

# project imports
import gsvd
import resample
import exceptions


class ResampleTest(abc.ABC):
    """Abstract base class for the ResampleTest class set. Forces existence
    of certain functions.
    """

    _subclasses = {}

    # maps abbreviated user-specified classnames to full PLS variant names
    _pls_types = {
        "mct": "Mean-Centering Task PLS",
        # "mct_mg": "Mean-Centering Task PLS - Multi-Group",
        "nrt": "Non-Rotated Task PLS",
        "rb": "Regular Behaviour PLS",
        "mb": "Multiblock PLS",
        "nrb": "Non-Rotated Behaviour PLS",
        "nrmb": "Non-Rotated Multiblock PLS",
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
        return cls._subclasses[pls_method](*args, **kwargs)


@ResampleTest._register_subclass("mct")
@ResampleTest._register_subclass("rb")
class _ResampleTestTaskPLS(ResampleTest):
    """Class that runs permutation and bootstrap tests for Task PLS. When run,
    this class generates fields for permutation test information
    (permutation ratio, etc.) and for bootstrap test informtaion (confidence
    intervals, standard errors, bootstrap ratios, etc.).

    Parameters
    ----------
    X : np_array
        Input neural matrix/matrices for use with PLS. This matrix is passed
        in from the PLS class.
    U : np_array
        Left singular vectors of X.
    s : np_array
        Singular values for X. Used to compute permutation ratio.
    V : np_array
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
    ngroups : int, optional
        Value specifying the number of groups used in PLS. Specified by PLS
        class; defaults to 1.
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
    conf_ints : 2-tuple of np_arrays
        Upper and lower element-wise confidence intervals for the resampled
        left singular vectors in a tuple.
    std_errs : np_array
        Element-wise standard errors for the resampled right singular vectors.
    boot_ratios : np_array
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
        preprocess=None,
        nperm=1000,
        nboot=1000,
        ngroups=1,
        nonrotated=None,
        dist=(0.05, 0.95),
    ):
        self.dist = dist

        self.permute_ratio = self._permutation_test(
            X,
            Y,
            s,
            cond_order,
            ngroups,
            nperm,
            preprocess=preprocess,
            nonrotated=nonrotated,
        )
        self.conf_ints, self.std_errs, self.boot_ratios = self._bootstrap_test(
            X,
            Y,
            U,
            s,
            V,
            cond_order,
            ngroups,
            nboot,
            preprocess=preprocess,
            nonrotated=nonrotated,
            dist=self.dist,
        )

    @staticmethod
    def _permutation_test(
        X, Y, s, cond_order, ngroups, niter, preprocess=None, nonrotated=None
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
        print("----Running Permutation Test----\n")
        for i in range(niter):
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")
            # create resampled X matrix and get resampled indices
            X_new = np.empty(X.shape)
            if Y is not None:
                Y_new = np.empty(Y.shape)
            resampled_indices = []
            group_sums = np.array([np.sum(i) for i in cond_order])
            idx = 0
            for i in range(ngroups):
                (
                    X_new[idx : idx + group_sums[i],],
                    res_ind,
                ) = resample.resample_without_replacement(
                    X[idx : idx + group_sums[i],], cond_order, return_indices=True
                )

                if Y is not None:
                    Y_new[idx : idx + group_sums[i],] = Y[idx : idx + group_sums[i],][
                        res_ind,
                    ]
                resampled_indices.append(res_ind)
                idx += group_sums[i]
            resampled_indices = np.array(resampled_indices)

            # X_new, resampled_indices = resample.resample_without_replacement(
            #     X, cond_order, return_indices=True
            # )

            # pass in preprocessing function (i.e. mean-centering) for use
            # after sampling
            if Y is None:
                X_new_means, X_new_mc = preprocess(X_new, cond_order=cond_order)

                # run GSVD on mean-centered, resampled matrix
                U_hat, s_hat, V_hat = gsvd.gsvd(X_new_mc)
            else:
                R_new = preprocess(X_new, Y_new, cond_order)
                U_hat, s_hat, V_hat = gsvd.gsvd(R_new)
            # insert s_hat into singvals tracking matrix
            # singvals[:, i] = s_hat
            # count number of times sampled singular values are
            # greater than observed singular values, element-wise
            greatersum += s_hat > s

        permute_ratio = greatersum / niter
        return permute_ratio

    @staticmethod
    def _bootstrap_test(
        X,
        Y,
        U,
        s,
        V,
        cond_order,
        ngroups,
        niter,
        preprocess=None,
        nonrotated=None,
        dist=(0.05, 0.95),
    ):
        """Runs a bootstrap estimation on X matrix. Resamples X with
        replacement according to the condition order, runs PLS on the
        resampled X matrices, and computes `conf_int`, `std_errs`, and
        `boot_ratios`.
        """
        # if ngroups > 1:
        #     raise exceptions.NotImplementedError(
        #         "Multi-group MCT-PLS is not yet implemented."
        #     )

        # allocate memory for sampled values
        left_sv_sampled = np.empty((niter, U.shape[0], U.shape[0]))
        right_sv_sampled = np.empty((niter, V.shape[0], V.shape[0]))

        # right_sum = np.zeros(X.shape[1], X.shape[1])
        # right_squares = np.zeros(X.shape[1], X.shape[1])
        print("----Running Bootstrap Test----\n")
        for i in range(niter):
            # print out iteration number every 50 iterations
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}")
            # create resampled X matrix and get resampled indices
            # resample within-group using cond_order for group size info
            X_new = np.empty(X.shape)
            if Y is not None:
                Y_new = np.empty(Y.shape)
            resampled_indices = []
            group_sums = np.array([np.sum(i) for i in cond_order])
            idx = 0
            for i in range(ngroups):
                (
                    X_new[idx : idx + group_sums[i],],
                    res_ind,
                ) = resample.resample_with_replacement(
                    X[idx : idx + group_sums[i],], cond_order, return_indices=True
                )
                # use same resampled indices for Y if applicable
                if Y is not None:
                    Y_new[idx : idx + group_sums[i],] = Y[idx : idx + group_sums[i],][
                        res_ind
                    ]
                resampled_indices.append(res_ind)
                idx += group_sums[i]
            resampled_indices = np.array(resampled_indices)

            # pass in preprocessing function (e.g. mean-centering) for use
            # after sampling
            if Y is None:
                X_new_means, X_new_mc = preprocess(
                    X_new, cond_order=cond_order
                )  # , ngroups=ngroups)

                # run GSVD on mean-centered, resampled matrix
                U_hat, s_hat, V_hat = gsvd.gsvd(X_new_mc)
            else:
                # compute condition-wise correlation matrices of resampled
                # input matrices and run GSVD
                R_new = preprocess(X_new, Y_new, cond_order)
                U_hat, s_hat, V_hat = gsvd.gsvd(R_new)

            # insert left singular vector into tracking np_array
            left_sv_sampled[i] = U_hat
            right_sv_sampled[i] = V_hat
            # right_sum += V_hat
            # right_squares += np.power(V_hat, 2)

        # compute confidence intervals of U sampled
        conf_int = resample.confidence_interval(left_sv_sampled)
        # compute standard error of left singular vector
        std_errs = scipy.stats.sem(right_sv_sampled, axis=0)
        # compute bootstrap ratios
        boot_ratios = np.divide(std_errs, V)
        return (conf_int, std_errs, boot_ratios)

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
