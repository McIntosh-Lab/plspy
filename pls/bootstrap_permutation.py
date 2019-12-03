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

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass


class ResampleTestTaskPLS(ResampleTest):
    """
    """

    def __init__(
        self,
        X,
        s,
        cond_order,
        preprocess=None,
        nperm=1000,
        nboot=1000,
        ngroups=1,
        nonrotated=None,
        dist=(0.05, 0.95),
    ):
        # self.X = X
        # self.s = s
        # self.niter = niter

        self.dist = dist

        self.permute_ratio = self._permutation_test(
            X,
            s,
            cond_order,
            ngroups,
            nperm,
            preprocess=preprocess,
            nonrotated=nonrotated,
        )
        self.conf_ints, self.std_errs, self.boot_ratios = self._bootstrap_test(
            X,
            s,
            cond_order,
            ngroups,
            nboot,
            preprocess=preprocess,
            nonrotated=nonrotated,
            dist=self.dist,
        )

    @staticmethod
    def _permutation_test(
        X, s, cond_order, ngroups, niter, preprocess=None, nonrotated=None
    ):
        """
        """
        if ngroups > 1:
            raise exceptions.NotImplementedError(
                "Multi-group MCT-PLS is not yet implemented."
            )

        # singvals = np.empty((s.shape[0], niter))
        print(f"niter={niter}")
        greatersum = 0
        print("----Running Permutation Test----\n")
        for i in range(niter):
            if (i + 1) % 5 == 0:
                print(f"Iteration {i + 1}")
            # create resampled X matrix and get resampled indices
            X_new, resampled_indices = resample.resample_without_replacement(
                X, C=cond_order, return_indices=True
            )

            # pass in preprocessing function (i.e. mean-centering) for use
            # after sampling
            X_new_means, X_new_mc = preprocess(X_new, ngroups=ngroups)

            # run GSVD on mean-centered, resampled matrix
            U_hat, s_hat, V_hat = gsvd.gsvd(X_new_mc)
            # insert s_hat into singvals tracking matrix
            # singvals[:, i] = s_hat
            # count number of times sampled singular values are
            # greater than observed singular values
            greatersum += sum(s_hat > s)

        permute_ratio = greatersum / niter
        return permute_ratio

    @staticmethod
    def _bootstrap_test(
        X,
        s,
        cond_order,
        ngroups,
        niter,
        preprocess=None,
        nonrotated=None,
        dist=(0.05, 0.95),
    ):
        """
        """
        if ngroups > 1:
            raise exceptions.NotImplementedError(
                "Multi-group MCT-PLS is not yet implemented."
            )

        left_sv_sampled = np.empty((niter, X.shape[0], X.shape[0]))
        right_sv_sampled = np.empty((niter, X.shape[1], X.shape[1]))

        # right_sum = np.zeros(X.shape[1], X.shape[1])
        # right_squares = np.zeros(X.shape[1], X.shape[1])
        print("----Running Bootstrap Test----\n")
        for i in range(niter):
            if (i + 1) % 5 == 0:
                print(f"Iteration {i + 1}")
            # create resampled X matrix and get resampled indices
            X_new, resampled_indices = resample.resample_with_replacement(
                X, C=cond_order, return_indices=True
            )

            # pass in preprocessing function (i.e. mean-centering) for use
            # after sampling
            X_new_means, X_new_mc = preprocess(X_new, ngroups=ngroups)

            # run GSVD on mean-centered, resampled matrix
            U_hat, s_hat, V_hat = gsvd.gsvd(X_new_mc)

            # insert left singular vector into tracking np_array
            left_sv_sampled[i] = U_hat
            right_sv_sampled[i] = V_hat
            # right_sum += V_hat
            # right_squares += np.power(V_hat, 2)

        # compute standard error of left singular vector
        std_errs = scipy.stats.sem(right_sv_sampled, axis=0)
        conf_int = resample.confidence_interval(left_sv_sampled)
        boot_ratios = np.divide(std_errs, right_sv_sampled)
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
