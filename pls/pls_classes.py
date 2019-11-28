import abc
import numpy as np

# project imports
import bootstrap_permutation
import gsvd


def PLS(*args, **kwargs):
    """Front-facing wrapper function for PLS that captures user input
    and extracts user-specified PLS method. If no method is specified,
    default PLS is used.

    TODO: implement first version of PLS and document required values here

    """
    try:
        pls_method = kwargs.pop("pls_method")
    except KeyError:
        pls_method = "default"

    # return finished PLS class with user-specified method
    return PLSBase.create(pls_method, *args, **kwargs)


class PLSBase(abc.ABC):
    """Abstract base class and factory for PLS. Registers and keeps track of
    different defined methods/implementations of PLS, as well as enforces use
    of base functions that all PLS implementations should use.
    """

    subclasses = {}

    # force existence of preprocess function
    @abc.abstractmethod
    def preprocess(self):
        pass

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


@PLSBase.register_subclass("default")
class Default(PLSBase):
    """Driver function for
    """

    def __init__(
        self,
        X: np.array,
        groups_sizes: tuple,
        num_conditions: int,
        num_perm: int = 1000,
        num_boot: int = 1000,
        **kwargs,
    ):
        self.X = X
        self.groups_sizes = groups_sizes
        self.num_groups = len(self.groups_sizes)
        self.num_conditions = num_conditions
        self.num_perm = num_perm
        self.num_boot = num_boot
        for k, v in kwargs.items():
            setattr(self, k, v)

    # TODO: implenent
    def preprocess(self):
        pass

    def run_pls(self):
        pass

    def bootstrap(self):
        pass

    def permutation(self):
        pass

    def __str__(self):
        stg = ""
        for k, v in self.__dict__.items():
            stg += f"\n{k}:\n\t"
            stg += str(v).replace("\n", "\n\t")
        return stg
