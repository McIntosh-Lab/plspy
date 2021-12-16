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

    # Estimate number of components for the PCA imputation (with scaled input matrix)
    estim_ncp = missMDA.estim_ncpPCA(base.scale(M),
                                     1, # Min components - must keep at least one component to impute
                                     5, # Max components
                                     scale = True) # Each variable given same weight

    # number of components
    num_comps = estim_ncp[0][0]

    # Impute 
    MIPCA_result = missMDA.MIPCA(M,
                            ncp=num_comps,
                            scale=True) 

    M_impute = MIPCA_result[1]

   return M_impute
