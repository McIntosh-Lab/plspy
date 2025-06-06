import abc
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import scipy.stats

# import helpers
# project imports
from . import bootstrap_permutation, split_half_resampling, class_functions, exceptions, gsvd


class PLSBase(abc.ABC):
    """Abstract base class and factory for PLS. Registers and keeps track of
    different defined methods/implementations of PLS, as well as enforces use
    of base functions that all PLS implementations should use.
    """

    # tracks registered PLSBase subclasses
    _subclasses = {}

    # maps abbreviated user-specified classnames to full PLS variant names
    _pls_types = {
        "mct": "Mean-Centring Task PLS",
        "rb": "Regular Behaviour PLS",
        "cst": "Contrast Task PLS",
        "csb": "Contrast Behaviour PLS",
        "mb": "Multiblock PLS",
        "cmb": "Contrast Multiblock PLS",
    }

    # force existence of run function
    # @abc.abstractmethod
    # def _run_pls(self):
    #     pass

    @abc.abstractmethod
    def _get_groups_info(self):
        pass

    @abc.abstractmethod
    def _get_cond_order(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    # register valid decorated PLS method as a subclass of PLSBase
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
            raise ValueError(f"Invalid PLS method {pls_method}")
        return cls._subclasses[pls_method](*args, **kwargs)


@PLSBase._register_subclass("mct")
class _MeanCentreTaskPLS(PLSBase):
    """Driver class for Mean-Centred Task PLS.

    Classed called for Mean-Centred Task PLS. TODO: add more here.

    Parameters
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix.
    Y: None
        Not used in Mean-Centred Task PLS.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    rotate_method : int, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method).
        rotate_method options:

        0 - compute s using SVD/GSVD

        1 - compute s using Procrustes rotation

        2 - compute s by derivation

    mctype : int, optional
        mctype options:

        0 - within each group remove group means from condition means (default)

        1 - remove grand condition means from each group condition mean

        2 - remove grand mean (over all subjects and conditions)

        3 - remove all main effects - subtract condition and group means (group by condition)


    Attributes
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_groups : int
        Value specifying the number of groups in the input data.
    num_conditions : int
        Number of conditions in each matrix.
    cond_order: array-like
        List/array where each entry holds the number of subjects per condition
        for each group in the input matrix.
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    X_means: np.array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.
    U: np.array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np.array
        Vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
    X_latent : np.array
        Latent variables of input X; dot-product of X_mc and V.
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.

    """

    def __init__(
        self,
        X: np.array,
        # Y: None,
        groups_sizes: tuple,
        num_conditions: int,
        Y: list = None,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        mctype: int = 0,
        **kwargs: str,
    ):
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]
        self._user_defined_attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)
        
        if len(X.shape) != 2:  # or len(X.shape) < 2:
            raise exceptions.ImproperShapeError(
                "Input matrix must be 2-dimensional."
            )
        self.X = X

        if Y is not None:
            raise ValueError(
                f"Do not provide a Y/behavioural matrix "
                f"for {self._pls_types[self.pls_alg]}."
            )

        if "contrasts" in kwargs:
            raise ValueError(
                f"Do not provide a contrast matrix "
                f"for {self._pls_types[self.pls_alg]}."
            )
        self.groups_sizes, self.num_groups = self._get_groups_info(
            groups_sizes
        )
        self.num_conditions = num_conditions
        # if no user-specified condition list, generate one
        if cond_order is None:
            self.cond_order = self._get_cond_order(
                self.X.shape, self.groups_sizes, self.num_conditions
            )
        else:
            # TODO: adjust input size and move input error
            # checking to pls.py
            # if len(cond_order.shape) != len(self.X.shape):
            if sum(groups_sizes) * num_conditions != self.X.shape[0]:
                raise exceptions.InputMatrixDimensionMismatchError(
                    "Dimension of condition orders does not match "
                    "dimension of input matrix X. Please make sure "
                    "that the sum of the conditions in all groups adds "
                    "up to the number of rows in the input matrix."
                )
            self.cond_order = cond_order

        self.num_perm = num_perm
        self.num_boot = num_boot
        if num_conditions == 1 and mctype != 1:
            print("Because you are running single condition Task PLS, " 
                "input Mean-Centering Type has to set to 1"
            )
            self.mctype = 1
        else:
            self.mctype = mctype

        # compute X means and X mean-centred values
        self.X_means, self.X_mc = class_functions._mean_centre(
            self.X, self.cond_order, mctype=self.mctype
        )
        self.U, self.s, self.V = class_functions._run_pls(self.X_mc)
        print(f"X_mc shape: {self.X_mc.shape}")
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = class_functions._compute_X_latents(self.X, self.V)
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            None,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            self.mctype,
            preprocess=class_functions._mean_centre,
            nperm=self.num_perm,
            nboot=self.num_boot,
            rotate_method=rotate_method,
        )

        # Split-half resampling
        if "num_split" in self._user_defined_attrs:
            self.num_split = int(self.num_split)
            if self.num_split > 0:
                print("----Running Split-half Reproducibility Tests----\n")
                self.pls_repro_tt = split_half_resampling.split_half_test_train(
                    self.pls_alg,
                    self.X,
                    None,
                    self.cond_order,                    
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts=None
                    )
                self.pls_repro_sh = split_half_resampling.split_half(
                    self.pls_alg,
                    self.X,
                    None,
                    self.cond_order,
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts = None,
                    lv = self.lv-1,
                    CI = self.CI,
                )   
            else:
                print("num_split value was specified as zero. The split-half resampling reproducibility tests were not run.")   
        else:
            print("The split-half resampling reproducibility tests were not run.")

        # swap U & V to be consistent with matlab
        self.U, self.V = self.V, self.U
        print("\nDone.")

    @staticmethod
    def _get_groups_info(groups_tuple):
        """Returns tuple of groups tuple passed into class and
        number of groups.
        """
        # return empty tuple and 0-length if None type
        if groups_tuple is None:
            return ((), 0)
        else:
            return (groups_tuple, len(groups_tuple))

    @staticmethod
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
        # cond_order = []
        # # print(f"GT: {groups_tuple}")
        # for i in range(len(groups_tuple)):
        #     group_list = []
        #     for k in range(num_conditions):
        #         group_list.extend([k] * groups_tuple[i])
        #         cond_order.append(group_list)
        # return np.array(cond_order)

    def __repr__(self):
        stg = ""
        info = f"\nAlgorithm: {self._pls_types[self.pls_alg]}\n\n"
        stg += info
        for k, v in self.__dict__.items():
            if k[0] != "_":
                stg += f"\n{k}:\n\t"
                stg += str(v).replace("\n", "\n\t")
        return stg

    def __str__(self):
        stg = ""
        info = f"\nAlgorithm: {self._pls_types[self.pls_alg]}\n\n"
        stg += info
        for k, v in self.__dict__.items():
            if k[0] != "_":
                stg += f"\n{k}:\n\t"
                stg += str(v).replace("\n", "\n\t")
        return stg


@PLSBase._register_subclass("rb")
class _RegularBehaviourPLS(_MeanCentreTaskPLS):
    """Driver class for Behavioural PLS.

    Class called for Behavioural PLS. TODO: add more here.

    Parameters
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix.
    Y: np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    rotate_method : int, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method).
        rotate_method options:

        0 - compute s using SVD/GSVD

        1 - compute s using Procrustes rotation

        2 - compute s by derivation


    Attributes
    ----------
    X : np.array
        Input neural matrix/matrices for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y : np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_groups : int
        Value specifying the number of groups in the input data.
    num_conditions : int
        Number of conditions in each matrix.
    cond_order : array-like
        List/array where each entry holds the number of subjects per condition
        for each group in the input matrix.
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    X_means: np.array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.
    U: np.array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np.array
        Vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
    Y_latent : np.array
        Latent variables for contrasts.
    X_latent : np.array
        Latent variables of input X; dot-product of X_mc and V.
    lvcorrs : np.array
        Computed latent variable correlations
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.
    """

    def __init__(
        self,
        X: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        Y: list = None,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        **kwargs,
    ):
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]
        self._user_defined_attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)

        if Y is None:
            raise exceptions.MissingParameterError(
                "Please provide a Y/behavioural matrix."
            )
            # raise ValueError(
            #     f"For {self._pls_types[self.pls_alg]}, " f"Y must NOT be of type None."
            # )

        if "contrasts" in kwargs:
            raise ValueError(
                f"Do not provide a contrast matrix "
                f"for {self._pls_types[self.pls_alg]}."
            )

        if len(X.shape) != 2 or len(Y.shape) != 2:  # or len(X.shape) < 2:
            raise exceptions.ImproperShapeError(
                "Input matrices must be 2-dimensional."
            )
        self.X = X
        self.Y = Y

        self.groups_sizes, self.num_groups = self._get_groups_info(
            groups_sizes
        )
        self.num_conditions = num_conditions
        # if no user-specified condition list, generate one
        if cond_order is None:
            self.cond_order = self._get_cond_order(
                self.X.shape, self.groups_sizes, self.num_conditions
            )
        else:
            # TODO: adjust input size and move input error
            # checking to pls.py
            # if len(cond_order.shape) != len(self.X.shape):
            # check calculated size matches input length
            calc_len = sum(groups_sizes) * num_conditions
            if calc_len != self.X.shape[0] or calc_len != self.Y.shape[0]:
                raise exceptions.InputMatrixDimensionMismatchError(
                    "Dimension of condition orders does not match "
                    "dimension of input matrix X and/or Y. Please make sure "
                    "that the sum of the conditions in all groups adds "
                    "up to the number of rows in the input matrices."
                )
            self.cond_order = cond_order

        self.num_perm = num_perm
        self.num_boot = num_boot
        # TODO: catch extraneous keyword args

        # assign functions to class
        # TODO: decide whether or not these should be applied
        # or if users should import from class_functions module
        class_functions._compute_R = class_functions._compute_corr

        # compute R correlation matrix
        self.R = class_functions._compute_R(self.X, self.Y, self.cond_order)
        self.U, self.s, self.V = class_functions._run_pls(self.R)
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = class_functions._compute_X_latents(self.X, self.V)
        self.Y_latent = class_functions._compute_Y_latents(
            self.Y, self.U, self.cond_order
        )
        # compute latent variable correlation matrix for V using compute_R
        self.lvcorrs = class_functions._compute_R(
            self.X_latent, self.Y, self.cond_order
        )
        # self.lvcorrs[:, 1:] = self.lvcorrs[:, 1:] * -1
        # self.lvcorrs[:, 0] = np.abs(self.lvcorrs[:, 0])

        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            self.Y,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            mctype = None,
            preprocess=class_functions._compute_R,
            nperm=self.num_perm,
            nboot=self.num_boot,
            rotate_method=rotate_method,
        )

        # Split-half resampling
        if "num_split" in self._user_defined_attrs:
            self.num_split = int(self.num_split)
            if self.num_split > 0:
                print("----Running Split-half Reproducibility Tests----\n")
                self.pls_repro_tt = split_half_resampling.split_half_test_train(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,                    
                    num_split=self.num_split,
                    mctype=None,
                    contrasts=None
                    )
                self.pls_repro_sh = split_half_resampling.split_half(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,
                    num_split=self.num_split,
                    mctype=None,
                    contrasts = None,
                    lv = self.lv-1,
                    CI = self.CI,
                )   
            else:
                print("num_split value was specified as zero. The split-half resampling reproducibility tests were not run.")   
        else:
            print("The split-half resampling reproducibility tests were not run.")

        # swap U & V to be consistent with matlab
        self.U, self.V = self.V, self.U
        print("\nDone.")


@PLSBase._register_subclass("cst")
class _ContrastTaskPLS(_MeanCentreTaskPLS):
    """Driver class for Contrast Task PLS.

    Class called for Contrast Task PLS. TODO: add more here.

    Parameters
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix.
    Y: None
        Not used in Contrast Task PLS.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    rotate_method : int, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method).
        rotate_method options:

        0 - compute s using SVD/GSVD

        1 - compute s using Procrustes rotation

        2 - compute s by derivation

    mctype : int, optional
        mctype options:

        0 - within each group remove group means from condition means (default)

        1 - remove grand condition means from each group condition mean

        2 - remove grand mean (over all subjects and conditions)

        3 - remove all main effects - subtract condition and group means (group by condition)

    contrasts: np.array
        contrast matrix for use in Contrast Task PLS. Used to create
        different methods of comparison.


    Attributes
    ----------
    X : np.array
        Input neural matrix/matrices for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y : np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_groups : int
        Value specifying the number of groups in the input data.
    num_conditions : int
        Number of conditions in each matrix.
    cond_order : array-like
        List/array where each entry holds the number of subjects per condition
        for each group in the input matrix.
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    X_means: np.array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.
    U: np.array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np.array
        Vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
    Y_latent : np.array
        Latent variables for contrasts.
    X_latent : np.array
        Latent variables of input X; dot-product of X_mc and V.
    lvintercorrs : np.array
        U.T * U. Optionally normed if rotate in [1,2].
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.
    """

    def __init__(
        self,
        X: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        Y: list = None,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        mctype: int = 0,
        contrasts: list = None,
        **kwargs,
    ):
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]
        self._user_defined_attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)

        if Y is not None:
            raise ValueError(
                f"Do not provide a Y/behavioural matrix "
                f"for {self._pls_types[self.pls_alg]}."
            )

        if len(X.shape) != 2:  # or len(X.shape) < 2:
            raise exceptions.ImproperShapeError(
                "Input matrix must be 2-dimensional."
            )
        self.X = X

        self.groups_sizes, self.num_groups = self._get_groups_info(
            groups_sizes
        )
        self.num_conditions = num_conditions
        # if no user-specified condition list, generate one
        if cond_order is None:
            self.cond_order = self._get_cond_order(
                self.X.shape, self.groups_sizes, self.num_conditions
            )
        else:
            # TODO: adjust input size and move input error
            # checking to pls.py
            # if len(cond_order.shape) != len(self.X.shape):
            # check calculated size matches input length
            calc_len = sum(groups_sizes) * num_conditions
            if calc_len != self.X.shape[0]:
                raise exceptions.InputMatrixDimensionMismatchError(
                    "Dimension of condition orders does not match "
                    "dimension of input matrix X and/or Y. Please make sure "
                    "that the sum of the conditions in all groups adds "
                    "up to the number of rows in the input matrices."
                )
            self.cond_order = cond_order

        if contrasts is None:
            raise exceptions.MissingParameterError(
                "Please provide a contrast matrix."
            )
        self.contrasts = class_functions._normalize(contrasts)

        self.num_perm = num_perm
        self.num_boot = num_boot
        if num_conditions == 1 and mctype != 1:
            print("Because you are running single condition Task PLS, " 
                "input Mean-Centering Type has to set to 1"
            )
            self.mctype = 1
        else:
            self.mctype = mctype
        # TODO: catch extraneous keyword args

        # compute mean-centred matrix
        # self.R = class_functions._mean_centre(
        #     self.X, self.cond_order, return_means=False, mctype=self.mctype
        # )
        # get matrix for svd
        self.R = class_functions._get_group_condition_means(self.X, self.cond_order)
        
        self.U, self.s, self.V = class_functions._run_pls_contrast(
            self.R, self.contrasts
        )

        # norm lvintercorrs if rotate method is
        # Procrustes or derived
        if rotate_method in [1, 2]:
            U_normed = self.U / np.linalg.norm(self.U)
            self.lvintercorrs = U_normed.T @ U_normed
        else:
            self.lvintercorrs = self.V.T @ self.V
        # self.X_latent = np.dot(self.X_mc, self.V)
        # get X_latents
        V_normed = class_functions._normalize(self.V)
        self.X_latent = class_functions._compute_X_latents(self.X, V_normed)
        
        # self.Y_latent = class_functions._compute_Y_latents(self.Y, self.U, self.cond_order)
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            None,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            self.mctype,
            preprocess=class_functions._mean_centre,
            nperm=self.num_perm,
            nboot=self.num_boot,
            rotate_method=rotate_method,
            #mctype=self.mctype,
            contrast=self.contrasts,
        )

        # Split-half resampling
        if "num_split" in self._user_defined_attrs:
            self.num_split = int(self.num_split)
            if self.num_split > 0:
                print("----Running Split-half Reproducibility Tests----\n")
                self.pls_repro_tt = split_half_resampling.split_half_test_train(
                    self.pls_alg,
                    self.X,
                    None,
                    self.cond_order,                    
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts=self.contrasts
                    )
                self.pls_repro_sh = split_half_resampling.split_half(
                    self.pls_alg,
                    self.X,
                    None,
                    self.cond_order,
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts = self.contrasts,
                    lv = self.lv-1,
                    CI = self.CI,
                )
            else:
                print("num_split value was specified as zero. The split-half resampling reproducibility tests were not run.")   
        else:
            print("The split-half resampling reproducibility tests were not run.")
                    
        # swap U & V to be consistent with matlab
        self.U, self.V = self.V, self.U
        print("\nDone.")


@PLSBase._register_subclass("csb")
class _ContrastBehaviourPLS(_ContrastTaskPLS):
    """Driver class for Contrast Behaviour PLS.

    Class called for Contrast Behaviour PLS. TODO: add more here.

    Parameters
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix.
    Y: np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    rotate_method : int, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method).
        rotate_method options:

        0 - compute s using SVD/GSVD

        1 - compute s using Procrustes rotation

        2 - compute s by derivation

    contrasts: np.array
        contrast matrix for use in Contrast Task PLS. Used to create
        different methods of comparison.


    Attributes
    ----------
    X : np.array
        Input neural matrix/matrices for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y : np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_groups : int
        Value specifying the number of groups in the input data.
    num_conditions : int
        Number of conditions in each matrix.
    cond_order : array-like
        List/array where each entry holds the number of subjects per condition
        for each group in the input matrix.
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    X_means: np.array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.
    U: np.array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np.array
        Vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
    Y_latent : np.array
        Latent variables for contrasts.
    X_latent : np.array
        Latent variables of input X; dot-product of X_mc and V.
    lvintercorrs : np.array
        U.T * U. Optionally normed if rotate in [1,2].
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.
    """

    def __init__(
        self,
        X: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        Y: list = None,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        contrasts: list = None,
        **kwargs,
    ):
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]
        self._user_defined_attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)

        if Y is None:
            raise exceptions.MissingParameterError(
                "Please provide a Y/behavioural matrix."
            )
            # raise ValueError(
            #     f"For {self._pls_types[self.pls_alg]}, " f"Y must NOT be of type None."
            # )

        if len(X.shape) != 2 or len(Y.shape) != 2:  #  or len(X.shape) < 2:
            raise exceptions.ImproperShapeError(
                "Input matrices must be 2-dimensional."
            )

        self.X = X
        self.Y = Y

        self.groups_sizes, self.num_groups = self._get_groups_info(
            groups_sizes
        )
        self.num_conditions = num_conditions
        # if no user-specified condition list, generate one
        if cond_order is None:
            self.cond_order = self._get_cond_order(
                self.X.shape, self.groups_sizes, self.num_conditions
            )
        else:
            # TODO: adjust input size and move input error
            # checking to pls.py
            # if len(cond_order.shape) != len(self.X.shape):
            # check calculated size matches input length
            calc_len = sum(groups_sizes) * num_conditions
            if calc_len != self.X.shape[0] or calc_len != self.Y.shape[0]:
                raise exceptions.InputMatrixDimensionMismatchError(
                    "Dimension of condition orders does not match "
                    "dimension of input matrix X and/or Y. Please make sure "
                    "that the sum of the conditions in all groups adds "
                    "up to the number of rows in the input matrices."
                )
            self.cond_order = cond_order

        if contrasts is None:
            raise exceptions.MissingParameterError(
                "Please provide a contrast matrix."
            )

        self.contrasts = class_functions._normalize(contrasts)

        self.num_perm = num_perm
        self.num_boot = num_boot
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]
        # TODO: catch extraneous keyword args
        self._user_defined_attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)

        class_functions._compute_R = class_functions._compute_corr

        # compute R correlation matrix
        self.R = class_functions._compute_R(self.X, self.Y, self.cond_order)
        #self.U, _, _= class_functions._run_pls(self.R)
        #self.contrasts = np.sign(contrasts) * np.abs(self.U) # apply contrast signs to U
        self.U, self.s, self.V = class_functions._run_pls_contrast(
            self.R, self.contrasts
        )
        # norm lvintercorrs if rotate method is
        # Procrustes or derived
        if rotate_method in [1, 2]:
            U_normed = self.U / np.linalg.norm(self.U)
            self.lvintercorrs = U_normed.T @ U_normed
        else:
            self.lvintercorrs = self.V.T @ self.V
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = class_functions._compute_X_latents(self.X, self.V)
        self.Y_latent = class_functions._compute_Y_latents(
            self.Y, self.U, self.cond_order
        )
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            self.Y,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            mctype = None,
            preprocess=class_functions._compute_R,
            nperm=self.num_perm,
            nboot=self.num_boot,
            rotate_method=rotate_method,
            contrast=self.contrasts,
        )

        # Split-half resampling
        if "num_split" in self._user_defined_attrs:
            self.num_split = int(self.num_split)
            if self.num_split > 0:
                print("----Running Split-half Reproducibility Tests----\n")
                self.pls_repro_tt = split_half_resampling.split_half_test_train(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,                    
                    num_split=self.num_split,
                    mctype=None,
                    contrasts=self.contrasts
                    )
                self.pls_repro_sh = split_half_resampling.split_half(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,
                    num_split=self.num_split,
                    mctype=None,
                    contrasts = self.contrasts,
                    lv = self.lv-1,
                    CI = self.CI,
                )   
            else:
                print("num_split value was specified as zero. The split-half resampling reproducibility tests were not run.")   
        else:
            print("The split-half resampling reproducibility tests were not run.")

        # swap U & V to be consistent with matlab
        self.U, self.V = self.V, self.U
        print("\nDone.")


@PLSBase._register_subclass("mb")
class _MultiblockPLS(_RegularBehaviourPLS):
    """Driver class for Multiblock PLS.

    Class called for Multiblock PLS. TODO: add more here.

    Parameters
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix.
    Y: np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    rotate_method : int, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method).
        rotate_method options:

        0 - compute s using SVD/GSVD

        1 - compute s using Procrustes rotation

        2 - compute s by derivation


    Attributes
    ----------
    X : np.array
        Input neural matrix/matrices for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y : np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_groups : int
        Value specifying the number of groups in the input data.
    num_conditions : int
        Number of conditions in each matrix.
    cond_order : array-like
        List/array where each entry holds the number of subjects per condition
        for each group in the input matrix.
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    bscan : array-like
        List/array specifying the subset of conditions to be used. E.g., [1 3] for 
        conditions 1 and 3. The conditions should be listed in ascending order. 
        Default bscan is all conditions i.e., [1 2 3 4] for 4 conditions.
    X_means: np.array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.
    U: np.array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np.array
        Vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
    Y_latent : np.array
        Latent variables for contrasts.
    X_latent : np.array
    lvcorrs : np.array
        Computed latent variable correlations
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.
    """

    def __init__(
        self,
        X: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        Y: list = None,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        **kwargs,
    ):
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]
        

        # TODO: catch extraneous keyword args
        self._user_defined_attrs = set()
        #Note: k = number of conditions
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)

        if Y is None:
            raise exceptions.MissingParameterError(
                "Please provide a Y/behavioural matrix."
            )
            # raise ValueError(
            #     f"For {self._pls_types[self.pls_alg]}, " f"Y must NOT be of type None."
            # )

        if "contrasts" in kwargs:
            raise ValueError(
                f"Do not provide a contrast matrix "
                f"for {self._pls_types[self.pls_alg]}."
            )

        if len(X.shape) != 2 or len(Y.shape) != 2:  # or len(X.shape) < 2:
            raise exceptions.ImproperShapeError(
                "Input matrices must be 2-dimensional."
            )

        self.X = X
        self.Y = Y

        self.groups_sizes, self.num_groups = self._get_groups_info(
            groups_sizes
        )
        self.num_conditions = num_conditions
        # if no user-specified condition list, generate one
        if cond_order is None:
            self.cond_order = self._get_cond_order(
                self.X.shape, self.groups_sizes, self.num_conditions
            )
        else:
            # TODO: adjust input size and move input error
            # checking to pls.py
            # if len(cond_order.shape) != len(self.X.shape):
            # check calculated size matches input length
            calc_len = sum(groups_sizes) * num_conditions
            if calc_len != self.X.shape[0] or calc_len != self.Y.shape[0]:
                raise exceptions.InputMatrixDimensionMismatchError(
                    "Dimension of condition orders does not match "
                    "dimension of input matrix X and/or Y. Please make sure "
                    "that the sum of the conditions in all groups adds "
                    "up to the number of rows in the input matrices."
                )
            self.cond_order = cond_order

        # Get bscan
        if "bscan" not in self._user_defined_attrs:
            self.bscan = [i for i in range (self.num_conditions)]
        else: 
            #self.bscan = [i-1 for i in self.bscan]
            if self.bscan != sorted(self.bscan):
                print("provided bscan not in ascending order - conditions in bscan will be correctly reordered")
            invalid_bscan = 0
            for item in self.bscan:
                if item < 0 or item > self.num_conditions-1:
                    invalid_bscan = 1
            if invalid_bscan == 1:
                print(f"bscan should be a subset of: 1 to {self.num_conditions}")

        self.num_perm = num_perm
        self.num_boot = num_boot

        # assign functions to class
        # TODO: decide whether or not these should be applied
        # or if users should import from class_functions module     
        self._create_multiblock = class_functions._create_multiblock
        self._compute_corr = class_functions._compute_corr

        # only keep conditions of interest from bscan for behavioural portion
        mask = np.array([])
        for item in self.cond_order:
            for cond_i, cond in enumerate(item):
                if cond_i in self.bscan:
                    mask = np.concatenate((mask,np.repeat(1,cond)))
                else:
                    mask = np.concatenate((mask,np.repeat(0,cond)))
        
        mask = mask.flatten().astype(bool)
        # Note: If bscan is full (e.g., not specified by user), all instances of Xbscan = X & Ybscan = Y
        # i.e., Xbscan will contain all conditions and will be equivalent to X.
        self.Xbscan = self.X[mask]
        self.Ybscan = self.Y[mask]
        
        # compute R correlation matrix
        self.multiblock = self._create_multiblock(
            self.X, self.cond_order, self.pls_alg, self.bscan, self.mctype,
            Xbscan = self.Xbscan, Ybscan = self.Ybscan) 

        self.U, self.s, self.V = class_functions._run_pls(self.multiblock)
       
        #### COMPUTE USC ####
        # Task X_latent (Tusc in matlab)
        V_normed = class_functions._normalize(self.V)
        T_X_latent = class_functions._compute_X_latents(self.X, V_normed)

        # Behaviour X_latents (Busc in matlab)
        B_X_latent = class_functions._compute_X_latents(self.Xbscan, self.V) 

        #Stack
        self.X_latent = np.vstack((np.array(T_X_latent),  np.array(B_X_latent)))

        # Change names to match matlab
        self.usc = self.X_latent 
        self.Tusc = T_X_latent
        self.Busc = B_X_latent
        # to do: add documentation that explains X and Y latent are equivalent to vsc and usc


       #### COMPUTE VSC ####
        # Split U into Tu and Bu
        Tu, Bu = class_functions._get_Tu_Bu(self.U, num_conditions, self.Y.shape[1], self.cond_order, self.bscan)

        # Compute Tusc (Tvsc in matlab)
        Tusc = class_functions._get_Tusc(Tu, num_conditions, self.cond_order)

        # Compute Busc (Bvsc in matlab)
        Busc = class_functions._get_Busc(Bu, num_conditions, self.Ybscan, self.cond_order, self.bscan)
        
        # Change names to match matlab
        self.Bvsc = Busc
        self.Tvsc = Tusc
        self.Tv = Tu
        self.Bv = Bu
        self.Y_latent = np.vstack([Tusc, Busc])
        self.vsc = self.Y_latent 
        
        # compute latent variable correlation matrix for V using compute_R
        self.lvcorrs = self._compute_corr(
            B_X_latent, self.Ybscan, self.cond_order[:,self.bscan]
        )
        
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            self.Y,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            self.mctype,
            preprocess=self._create_multiblock,
            nperm=self.num_perm,
            nboot=self.num_boot,
            rotate_method=rotate_method,
            bscan = self.bscan,
            Xbscan = self.Xbscan,
            Ybscan = self.Ybscan
        )

        # Split-half resampling
        if "num_split" in self._user_defined_attrs:
            self.num_split = int(self.num_split)
            if self.num_split > 0:
                print("----Running Split-half Reproducibility Tests----\n")
                self.pls_repro_tt = split_half_resampling.split_half_test_train(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,                    
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts=None,
                    bscan = self.bscan,
                    Xbscan = self.Xbscan,
                    Ybscan = self.Ybscan
                    )
                self.pls_repro_sh = split_half_resampling.split_half(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts = None,
                    bscan = self.bscan,
                    Xbscan = self.Xbscan,
                    Ybscan = self.Ybscan,
                    lv = self.lv-1,
                    CI = self.CI,
                )   
            else:
                print("num_split value was specified as zero. The split-half resampling reproducibility tests were not run.")   
        else:
            print("The split-half resampling reproducibility tests were not run.")

        # swap U & V to be consistent with matlab
        self.U, self.V = self.V, self.U
        print("\nDone.")


@PLSBase._register_subclass("cmb")
class _ContrastMultiblockPLS(_MultiblockPLS):
    """Driver class for Multiblock PLS.

    Class called for Multiblock PLS. TODO: add more here.

    Parameters
    ----------
    X : np.array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix.
    Y: np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    rotate_method : int, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method).
        rotate_method options:

        0 - compute s using SVD/GSVD

        1 - compute s using Procrustes rotation

        2 - compute s by derivation


    Attributes
    ----------
    X : np.array
        Input neural matrix/matrices for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y : np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_groups : int
        Value specifying the number of groups in the input data.
    num_conditions : int
        Number of conditions in each matrix.
    cond_order : array-like
        List/array where each entry holds the number of subjects per condition
        for each group in the input matrix.
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    X_means: np.array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np.array
        Mean-centred values corresponding to input matrix X.
    U: np.array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np.array
        Vector containing diagonal of the singular values.
    V: np.array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
    Y_latent : np.array
        Latent variables for contrasts.
    X_latent : np.array
        Latent variables of input X; dot-product of X_mc and V.
    lvcorrs : np.array
        Computed latent variable correlations
    lvintercorrs : np.array
        U.T * U. Optionally normed if rotate in [1,2].
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.
    """

    def __init__(
        self,
        X: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        Y: list = None,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        contrasts: list = None,
        **kwargs,
    ):
        # so pylint will shut up
        self.pls_alg = kwargs["pls_alg"]

        # TODO: catch extraneous keyword args
        self._user_defined_attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self._user_defined_attrs.add(k)

        if Y is None:
            raise exceptions.MissingParameterError(
                "Please provide a Y/behavioural matrix."
            )
            # raise ValueError(
            #     f"For {self._pls_types[self.pls_alg]}, " f"Y must NOT be of type None."
            # )

        if len(X.shape) != 2 or len(Y.shape) != 2:  #  or len(X.shape) < 2:
            raise exceptions.ImproperShapeError(
                "Input matrices must be 2-dimensional."
            )

        self.X = X
        self.Y = Y
        
        self.groups_sizes, self.num_groups = self._get_groups_info(
            groups_sizes
        )
        self.num_conditions = num_conditions
        # if no user-specified condition list, generate one
        if cond_order is None:
            self.cond_order = self._get_cond_order(
                self.X.shape, self.groups_sizes, self.num_conditions
            )
        else:
            # TODO: adjust input size and move input error
            # checking to pls.py
            # if len(cond_order.shape) != len(self.X.shape):
            # check calculated size matches input length
            calc_len = sum(groups_sizes) * num_conditions
            if calc_len != self.X.shape[0] or calc_len != self.Y.shape[0]:
                raise exceptions.InputMatrixDimensionMismatchError(
                    "Dimension of condition orders does not match "
                    "dimension of input matrix X and/or Y. Please make sure "
                    "that the sum of the conditions in all groups adds "
                    "up to the number of rows in the input matrices."
                )
            self.cond_order = cond_order


       # Get bscan
        if "bscan" not in self._user_defined_attrs:
            self.bscan = [i for i in range (self.num_conditions)]
        else: 
            #self.bscan = [i-1 for i in self.bscan]
            if self.bscan != sorted(self.bscan):
                print("provided bscan not in ascending order - conditions in bscan will be correctly reordered")
            invalid_bscan = 0
            for item in self.bscan:
                if item < 0 or item > self.num_conditions-1:
                    invalid_bscan = 1
            if invalid_bscan == 1:
                print(f"bscan should be a subset of: 1 to {self.num_conditions}")

        # only keep conditions of interest from bscan for behavioural portion
        mask = np.array([])
        for item in self.cond_order:
            for cond_i, cond in enumerate(item):
                if cond_i in self.bscan:
                    mask = np.concatenate((mask,np.repeat(1,cond)))
                else:
                    mask = np.concatenate((mask,np.repeat(0,cond)))
        
        mask = mask.flatten().astype(bool)

        # Note: If bscan is full (e.g., not specified by user), all instances of Xbscan = X & Ybscan = Y
        # i.e., Xbscan will contain all conditions and will be equivalent to X.
        self.Xbscan = self.X[mask]
        self.Ybscan = self.Y[mask]

        self.num_perm = num_perm
        self.num_boot = num_boot

        if contrasts is None:
            raise exceptions.MissingParameterError(
                "Please provide a contrast matrix."
            )
        # Create a binary mask for conditions of interest
        TBi = np.ones(self.num_conditions)
        Bi = np.zeros((self.Y.shape[1], self.num_conditions))
        Bi[:, self.bscan] = 1

        # Flatten and repeat for num_groups
        TBi = np.hstack((TBi, Bi.flatten()))
        TBi = np.tile(TBi, self.num_groups)

        # Apply the mask
        contrasts = contrasts[TBi.astype(bool), :]
        self.contrasts = class_functions._normalize(contrasts)
        
        # assign functions to class
        # TODO: decide whether or not these should be applied
        # or if users should import from class_functions module
        self._create_multiblock = class_functions._create_multiblock
        self._compute_corr = class_functions._compute_corr
        class_functions._compute_Y_latents = class_functions._compute_Y_latents

        # compute R correlation matrix
        self.multiblock = self._create_multiblock(
            self.X, self.cond_order, self.pls_alg, self.bscan, self.mctype,
            Xbscan = self.Xbscan, Ybscan = self.Ybscan) 
        self.U, self.s, self.V = class_functions._run_pls_contrast(
            self.multiblock, self.contrasts
        )
        
        # Task X_latent (Tusc in matlab)
        V_normed = class_functions._normalize(self.V)
        T_X_latent = class_functions._compute_X_latents(self.X, V_normed)
        
        # Behaviour X_latents (Busc in matlab)
        B_X_latent = class_functions._compute_X_latents(self.Xbscan, self.V) 

        #Stack
        self.X_latent = np.vstack((np.array(T_X_latent),  np.array(B_X_latent)))
        
        # Split U into Tu and Bu
        Tu, Bu = class_functions._get_Tu_Bu(self.U, num_conditions, self.Y.shape[1], self.cond_order, self.bscan)
        
        # Compute Tusc (Tvsc in matlab)
        Tusc = class_functions._get_Tusc(Tu, num_conditions, self.cond_order)

        # Compute Busc (Bvsc in matlab)
        Busc = class_functions._get_Busc(Bu, num_conditions, self.Ybscan, self.cond_order, self.bscan)

        # Change names to match matlab
        self.Bvsc = Busc
        self.Tvsc = Tusc
        self.Tv = Tu
        self.Bv = Bu
        self.Y_latent = np.vstack([Tusc, Busc])
        self.vsc = self.Y_latent
        self.usc = self.X_latent # to do: add documentation that explains X and Y latent are equivalent to vsc and usc
        self.Tusc = T_X_latent
        self.Busc = B_X_latent

        # compute latent variable correlation matrix for V using compute_R
        self.lvcorrs = self._compute_corr(
            B_X_latent, self.Ybscan, self.cond_order[:,self.bscan]
        )
        
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            self.Y,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            self.mctype,
            preprocess=self._create_multiblock,
            nperm=self.num_perm,
            nboot=self.num_boot,
            rotate_method=rotate_method,
            contrast=self.contrasts,
            bscan = self.bscan,
            Xbscan = self.Xbscan,
            Ybscan = self.Ybscan
        )

        # Split-half resampling
        if "num_split" in self._user_defined_attrs:
            self.num_split = int(self.num_split)
            if self.num_split > 0:
                print("----Running Split-half Reproducibility Tests----\n")
                self.pls_repro_tt = split_half_resampling.split_half_test_train(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,                    
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts=self.contrasts,
                    bscan = self.bscan,
                    Xbscan = self.Xbscan,
                    Ybscan = self.Ybscan,
                    )
                self.pls_repro_sh = split_half_resampling.split_half(
                    self.pls_alg,
                    self.X,
                    self.Y,
                    self.cond_order,
                    num_split=self.num_split,
                    mctype=self.mctype,
                    contrasts = self.contrasts,
                    bscan=self.bscan,
                    Xbscan=self.Xbscan,
                    Ybscan = self.Ybscan,
                    lv = self.lv-1,
                    CI = self.CI,
                )   
            else:
                print("num_split value was specified as zero. The split-half resampling reproducibility tests were not run.")   
        else:
            print("The split-half resampling reproducibility tests were not run.")
            
        # swap U & V to be consistent with matlab
        self.U, self.V = self.V, self.U
        print("\nDone.")
