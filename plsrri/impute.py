import rpy2
import numpy as np

# import rpy2's package module and importer
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


def impute(M):
    """Performs imputation using the R missMDA package.

    Function loads in missMDA R packages and related dependencies,
    estimates the number of components, imputes the matrix
    accordingly, and returns the imputed matrix.

    Parameters
    ----------
    M : array-like
        matrix to be imputed


    Returns
    -------
    M_impute : array-like
        imputed matrix
    """


# import basic R functions
base = rpackages.importr("base")
utils = rpackages.importr("utils")

# Get path to R libraries (YMMV - may need to choose another path)

lib_path = base._libPaths()[0]

# Import missMDA (may need to specify library location with lib_loc)
missMDA = importr("missMDA")
