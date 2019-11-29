import abc
import numpy as np

# project imports
# import bootstrap_permutation
import gsvd
import exceptions


class PLSBase(abc.ABC):
    """Abstract base class and factory for PLS. Registers and keeps track of
    different defined methods/implementations of PLS, as well as enforces use
    of base functions that all PLS implementations should use.
    """

    # tracks registered PLSBase subclasses
    subclasses = {}

    # maps abbreviated user-specified classnames to full PLS variant names
    pls_types = {
        "mct": "Mean-Centering Task PLS",
        # "mct_mg": "Mean-Centering Task PLS - Multi-Group",
        "nrt": "Non-Rotated Task PLS",
        "rb": "Regular Behaviour PLS",
        "mb": "Multiblock PLS",
        "nrb": "Non-Rotated Behaviour PLS",
        "nrmb": "Non-Rotated Multiblock PLS",
    }

    # force existence of preprocess function
    # @abc.abstractmethod
    # def preprocess(self):
    #     pass

    # force existence of run function
    @abc.abstractmethod
    def run_pls(self):
        pass

    # force existence of bootstrap function
    @abc.abstractmethod
    def bootstrap(self):
        pass

    # force existence of permutation function
    @abc.abstractmethod
    def permutation(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    # register valid decorated PLS method as a subclass of PLSBase
    @classmethod
    def register_subclass(cls, pls_method):
        def decorator(subclass):
            cls.subclasses[pls_method] = subclass
            return subclass

        return decorator

    # instantiate and return valid registered PLS method specified by user
    @classmethod
    def create(cls, pls_method, *args, **kwargs):
        if pls_method not in cls.subclasses:
            raise ValueError(f"Invalid PLS method {pls_method}")
        return cls.subclasses[pls_method](*args, **kwargs)


@PLSBase.register_subclass("mct")
class _MeanCenterTaskSingleGroupPLS(PLSBase):
    """Driver class for Mean-Centered Task PLS. Currently only has single-group
    MCT-PLS implemented.

    Parameters
    ----------
    X : np_array
        Input neural matrix/matrices for use with PLS. Each participant's
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
    num_perm : int
        Optional value specifying the number of iterations for the permutation
        test. Defaults to 0, meaning no permutation test will be run unless
        otherwise specified by the user.
    num_boot : int
        Optional value specifying the number of iterations for the bootstrap
        test. Defaults to 0, meaning no bootstrap test will be run unless
        otherwise specified by the user.

    Attributes
    ----------
    X : np_array
        Input neural matrix/matrices for use with PLS. Each participant's
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
        Mean-centered values corresponding to input matrix X.
    U: np_array
        Eigenvectors of matrix `X_mc`*`X_mc`^T;
        left singular vectors.
    s: np_array
        Vector containing diagonal of the singular values.
    V: np_array
        Eigenvectors of matrix `X_mc`^T*`X_mc`;
        right singular vectors.
     X_latent : np_array
        latent variables of input X; dot-product of X_mc and V.

    """

    def __init__(
        self,
        X: np.array,
        Y: None,
        groups_sizes: tuple,
        num_conditions: int,
        num_perm: int = 0,
        num_boot: int = 0,
        **kwargs,
    ):
        self.X = X
        if Y is not None:
            raise ValueError(
                "For {self.pls_types[self.pls_alg]},"
                "Y must be of type None. Y = \n{Y}"
            )
        self.groups_sizes, self.num_groups = self.get_groups_info(groups_sizes)
        # self.num_groups = len(self.groups_sizes)
        self.num_conditions = num_conditions
        self.num_perm = num_perm
        self.num_boot = num_boot
        for k, v in kwargs.items():
            setattr(self, k, v)

        # compute X means and X mean-centered values
        self.X_means, self.X_mc = self.mean_center(X, ngroups=self.num_groups)
        self.U, self.s, self.V = self.run_pls(
            self.X_mc, ngroups=self.num_groups
        )
        # self.X_latent = np.dot(self.X_mc, self.V)
        self.X_latent = self.compute_latents(self.X_mc, self.V)

    @staticmethod
    def compute_latents(I, EV, ngroups=1):
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
        if ngroups == 1:
            dotp = np.dot(I, EV)
            return dotp
        else:
            raise exceptions.NotImplementedError(
                "Multi-group MCT-PLS " "not yet implemented."
            )

    @staticmethod
    def mean_center(X, ngroups=1):
        """Single-group preprocessing for `X`. Generates `X_means` and
        `X_mc` for use with `run_pls`

        Parameters
        ---------
        X : np_array
            Input matrix for use with PLS.
        ngroups: int
            Number of groups in input data.
        Returns
        -------
        X_means: np_array
            Mean-values of X array on axis-0 (column-wise).
        X_mc: np_array
            Mean-centered values corresponding to input matrix X.


        """
        if ngroups == 1:
            prod = np.dot(
                np.ones((X.shape[0], 1)),
                np.mean(X, axis=0).reshape((1, X.shape[1])),
            )

            X_means = X - prod
            X_mc = X_means / np.linalg.norm(X_means)
            return (X_means, X_mc)
        else:
            raise exceptions.NotImplementedError(
                "Multi-group MCT-PLS " "not yet implemented."
            )

    @staticmethod
    def run_pls(mc, ngroups=1):
        """Runs and returns results of Generalized SVD on `mc`,
        mean-centered input matrix `X`.

        Parameters
        ----------
        mc: np_array
            Mean-centered values corresponding to input matrix X.
        ngroups: int
            Number of groups in input data.

        Returns
        -------
        U: np_array
            Eigenvectors of matrix `mc`*`mc`^T;
            left singular vectors.
        s: np_array
            vector containing diagonal of the singular values.
        V: np_array
            Eigenvectors of matrix `mc`^T*`mc`;
            right singular vectors.
        """
        if ngroups == 1:
            U, s, V = gsvd.gsvd(mc)
            return (U, s, V)
        else:
            raise exceptions.NotImplementedError(
                "Multi-group MCT-PLS not yet implemented."
            )

    # TODO: implement OOP version of boostrap and permutation
    # and wrap them with these
    def bootstrap(self):
        pass

    def permutation(self):
        pass

    @staticmethod
    def get_groups_info(groups_tuple):
        """Returns tuple of groups tuple passed into class and
        number of subjects.
        """
        # return empty tuple and 0-length if None type
        if groups_tuple is None:
            return ((), 0)
        else:
            return (groups_tuple, len(groups_tuple))

    def __repr__(self):
        stg = ""
        info = f"\nAlgorithm: {self.pls_types[self.pls_alg]}\n\n"
        stg += info
        for k, v in self.__dict__.items():
            stg += f"\n{k}:\n\t"
            stg += str(v).replace("\n", "\n\t")
        return stg

    def __str__(self):
        stg = ""
        info = f"\nAlgorithm: {self.pls_types[self.pls_alg]}\n\n"
        stg += info
        for k, v in self.__dict__.items():
            stg += f"\n{k}:\n\t"
            stg += str(v).replace("\n", "\n\t")
        return stg
