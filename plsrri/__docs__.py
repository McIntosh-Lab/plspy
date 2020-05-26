"""
File containing docstrings for various methods, modules, and files
in plsrri. Can be combined as needed to make full docstrings for 
functions and modules.
"""

plsrri_header = """
plsrri
======

plsrri is a Partial Least Squares package developed at the Rotman                              
Research Institute at Baycrest Health Sciences.

In addition to core PLS functionality, this package also contains the following modules:

class_functions
    core PLS functions, such as mean-centring, calling SVD/GSVD, etc.
gsvd
    Implementation of GSVD that uses NumPy's Fortran SVD function
resample
    Functions used for resampling in permutation and bootstrap
bootstrap_permutation
    Houses the bootstrap and permutation object (called by PLS)
pls_classes
    Source code for each PLS version
exceptions
    Houses custom exceptions used within PLS

"""


plsrri_body = """
Basic usage:

    Mean-Centred Task PLS:
        
        >>> result = plsrri.PLS(X, None, method="mct")

    Behavioural PLS:
    
        >>> result = plsrri.PLS(X, Y, method="rb")

    Contrast Task PLS:
        
        >>> result = plsrri.PLS(X, None, contrasts=C, method="cst")

    Contrast Behavioural PLS:
        
        >>> result = plsrri.PLS(X, Y, contrasts=C, method="cst")

    Multiblock PLS:
        
        >>> result = plsrri.PLS(X, Y, method="mb")

To see documentation on additional arguments and fields available, 
call help on a specific PLS method (see below for details).

Documentation is available both in help() form and will also be available
in website form. More information on how to access online documentation is 
forthcoming. Information on how to use help() is below.

To get help documentation on a particular version of PLS, type the following
in a Python interpreter after loading the module:
    >>> import plsrri
    >>> help(plsrri.methods["<methodname>"])

Where <method> is the string of one of the PLS versions shown below.

Available methods:
    "mct" - Mean-Centred Task PLS
    "rb"  - Regular Behaviour PLS
    "cst" - Contrast Task PLS
    "csb" - Contrast Behaviour PLS
    "mb"  - Multiblock PLS
    "cmb" - Contrast Multiblock PLS (under construction)

Note: calling
    >>> help(plsrri.PLS) 

will show you this same help page.


Author: Noah Frazier-Logue
"""

pls_wrapper_header ="""
Front-facing wrapper function for PLS that captures user input
and extracts user-specified PLS method. If no method is specified,
default PLS is used.

TODO: implement first version of PLS and document required values here


"""
        
