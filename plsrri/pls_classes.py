import abc
import numpy as np
import scipy.stats

# project imports
import bootstrap_permutation
import gsvd

# import helpers
import exceptions
import class_functions


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
        "nrmb": "Non-Rotated Multiblock PLS",
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
class _MeanCentreTaskSingleGroupPLS(PLSBase):
    """Driver class for Mean-Centred Task PLS.

    Classed called for Mean-Centred Task PLS. TODO: add more here.

    Parameters
    ----------
    X : np_array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y: None
        Not used in MCT-PLS.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix. For example, if input matrix `X`
        contained 7 participants and 3 conditions, it would be of length 21.
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
        If `rotate_method == 2`, singular values will be derived.

    Attributes
    ----------
    X : np_array
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
        Number of conditions in each matrix. For example, if input matrix `X`
        contained 7 participants and 3 conditions, it would be of length 21.
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
    X_means: np_array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np_array
        Mean-centred values corresponding to input matrix X.
    U: np_array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np_array
        Vector containing diagonal of the singular values.
    V: np_array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
     X_latent : np_array
        Latent variables of input X; dot-product of X_mc and V.
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.

    """

    def __init__(
        self,
        X: np.array,
        Y: None,
        groups_sizes: tuple,
        num_conditions: int,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        **kwargs,
    ):

        if len(X.shape) != 2:  #  or len(X.shape) < 2:
            raise exceptions.ImproperShapeError("Input matrix must be 2-dimensional.")
        self.X = X

        if Y is not None:
            raise ValueError(
                "For {self.pls_types[self.pls_alg]},"
                "Y must be of type None. Y = \n{Y}"
            )
        self.groups_sizes, self.num_groups = self._get_groups_info(groups_sizes)
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
        for k, v in kwargs.items():
            setattr(self, k, v)

        # assign functions to class
        self._mean_centre = class_functions._mean_centre
        self._run_pls = class_functions._run_pls
        self._compute_X_latents = class_functions._compute_X_latents

        # compute X means and X mean-centred values
        self.X_means, self.X_mc = self._mean_centre(
            self.X, self.cond_order  # , ngroups=self.num_groups
        )
        self.U, self.s, self.V = self._run_pls(self.X_mc, ngroups=self.num_groups)
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = self._compute_X_latents(self.X_mc, self.V)
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            None,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            preprocess=self._mean_centre,
            nperm=self.num_perm,
            nboot=self.num_boot,
            ngroups=self.num_groups,
            rotate_method=rotate_method,
        )
        print("\nDone.")

    @staticmethod
    def _get_groups_info(groups_tuple):
        """Returns tuple of groups tuple passed into class and
        number of subjects.
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

        cond_order = np.array([np.array([i] * num_conditions) for i in groups_tuple])
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
            stg += f"\n{k}:\n\t"
            stg += str(v).replace("\n", "\n\t")
        return stg

    def __str__(self):
        stg = ""
        info = f"\nAlgorithm: {self._pls_types[self.pls_alg]}\n\n"
        stg += info
        for k, v in self.__dict__.items():
            stg += f"\n{k}:\n\t"
            stg += str(v).replace("\n", "\n\t")
        return stg


@PLSBase._register_subclass("rb")
class _RegularBehaviourPLS(_MeanCentreTaskSingleGroupPLS):
    """Driver class for Behavioural Task PLS.

    Class called for Behavioural Task PLS. TODO: add more here.

    Parameters
    ----------
    X : np_array
        Input neural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y: np.array
        Input behavioural matrix for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    groups_sizes : tuple
        Tuple containing sizes of conditions, where each entry in the tuple
        corresponds to a group and each value in the entry corresponds to
        the number of participants in that group. E.g. in (7,6,5), group 1
        would have 7 participants and group 3 would have 5 participants.
    num_conditions : int
        Number of conditions in each matrix. For example, if input matrix `X`
        contained 7 participants and 3 conditions, it would be of length 21.
    num_perm : int, optional
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int, optional
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.
    nonrotated : boolean, optional
        Optional value specifying whether or not full GSVD should be used
        during bootstrap and permutation tests ("rotated" method). 
        If False, singular values will be derived.

    Attributes
    ----------
    X : np_array
        Input neural matrix/matrices for use with PLS. Each participant's
        data must be flattened and concatenated to form a single 2-dimensional
        matrix, separated by condition, for each group.
    Y : np_array
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
        Number of conditions in each matrix. For example, if input matrix `X`
        contained 7 participants and 3 conditions, it would be of length 21.
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
    X_means: np_array
        Mean-values of X array on axis-0 (column-wise).
    X_mc: np_array
        Mean-centred values corresponding to input matrix X.
    U: np_array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np_array
        Vector containing diagonal of the singular values.
    V: np_array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
     Y_latent : np_array
        Latent variables for contrasts.
     X_latent : np_array
        Latent variables of input X; dot-product of X_mc and V.
    resample_tests : class
        Class containing results for permutation and bootstrap tests. See
        documentation on Resample Tests for more information.
    """

    def __init__(
        self,
        X: np.array,
        Y: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        **kwargs,
    ):

        if len(X.shape) != 2 or len(Y.shape) != 2:  #  or len(X.shape) < 2:
            raise exceptions.ImproperShapeError("Input matrices must be 2-dimensional.")
        self.X = X
        self.Y = Y

        if Y is None:
            raise ValueError(
                "For {self.pls_types[self.pls_alg]},"
                "Y must NOT be of type None. Y = \n{Y}"
            )
        self.groups_sizes, self.num_groups = self._get_groups_info(groups_sizes)
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
        for k, v in kwargs.items():
            setattr(self, k, v)

        # assign functions to class
        # TODO: decide whether or not these should be applied
        # or if users should import from class_functions module
        self._compute_R = class_functions._compute_corr
        self._run_pls = class_functions._run_pls
        self._compute_X_latents = class_functions._compute_X_latents
        self._compute_Y_latents = class_functions._compute_Y_latents

        # compute R correlation matrix
        self.R = self._compute_R(self.X, self.Y, self.cond_order)

        self.U, self.s, self.V = self._run_pls(self.R, ngroups=self.num_groups)
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = self._compute_X_latents(self.X, self.V)
        self.Y_latent = self._compute_Y_latents(self.Y, self.U, self.cond_order)
        # compute latent variable correlation matrix for V using compute_R
        self.lvcorrs = self._compute_R(self.X_latent, self.Y, self.cond_order)

        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            self.Y,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            preprocess=self._compute_R,
            nperm=self.num_perm,
            nboot=self.num_boot,
            ngroups=self.num_groups,
            rotate_method=rotate_method,
        )
        print("\nDone.")


@PLSBase._register_subclass("cst")
class _ContrastTaskPLS(_MeanCentreTaskSingleGroupPLS):
    """
    """

    def __init__(
        self,
        X: np.array,
        Y: None,
        groups_sizes: tuple,
        num_conditions: int,
        cond_order: list = None,
        num_perm: int = 1000,
        num_boot: int = 1000,
        rotate_method: int = 0,
        contrast: list = None,
        **kwargs,
    ):

        if len(X.shape) != 2 or len(Y.shape) != 2:  #  or len(X.shape) < 2:
            raise exceptions.ImproperShapeError("Input matrices must be 2-dimensional.")
        self.X = X
        self.Y = Y

        if Y is not None:
            raise ValueError(
                "For {self.pls_types[self.pls_alg]},"
                "Y must be of type None. Y = \n{Y}"
            )
        self.groups_sizes, self.num_groups = self._get_groups_info(groups_sizes)
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

        if contrast is None:
            raise exceptions.MissingParameterError("Please provide a contrast matrix.")
        self.contast = contrast

        self.num_perm = num_perm
        self.num_boot = num_boot
        # TODO: catch extraneous keyword args
        for k, v in kwargs.items():
            setattr(self, k, v)

        # compute R correlation matrix
        self.R = self._compute_R(self.X, self.Y, self.cond_order)

        self.U, self.s, self.V = self._run_pls(self.R, ngroups=self.num_groups)
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = self._compute_latents(self.X, self.V)
        self.Y_latent = self._compute_Y_latents(self.Y, self.U, self.cond_order)
        self.resample_tests = bootstrap_permutation.ResampleTest._create(
            self.pls_alg,
            self.X,
            self.Y,
            self.U,
            self.s,
            self.V,
            self.cond_order,
            preprocess=self._compute_R,
            nperm=self.num_perm,
            nboot=self.num_boot,
            ngroups=self.num_groups,
            rotate_method=rotate_method,
        )
        print("\nDone.")

    @staticmethod
    def _compute_R(X, Y, cond_order):
        """Compute per-condition correlation matrices (concatenated as R)
        to pass into GSVD.

        This algorithm uses neural input matrix X and behavioural matrix Y to
        compute per-condition correlation matrices. It then concatenates them
        and returns R for use in GSVD.

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
        # print(f"R shape: {R.shape}")
        order_all = cond_order.reshape(-1)
        start = 0
        start_R = 0
        for i in range(len(order_all)):
            # X and Y zscored within each condition
            Xc_zsc = scipy.stats.zscore(X[start : order_all[i] + start,])
            Xc_zsc /= np.sqrt(order_all[i])
            Yc_zsc = scipy.stats.zscore(Y[start : order_all[i] + start,])
            Yc_zsc /= np.sqrt(order_all[i])
            np.nan_to_num(Xc_zsc, copy=False)
            np.nan_to_num(Yc_zsc, copy=False)
            # print(Xc_zsc)
            # print(Yc_zsc)
            # print("----------")
            # print(f"ydim: {Yc_zsc.shape}")
            # R_n = Y^T_n * X_n
            R[start_R : Y.shape[1] + start_R,] = np.matmul(Yc_zsc.T, Xc_zsc)
            start += order_all[i]
            start_R += Y.shape[1]
        # print("+++++++++++++++")

        return R

    @staticmethod
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