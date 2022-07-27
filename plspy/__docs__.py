"""
File containing docstrings for various methods, modules, and files
in plspy. Can be combined as needed to make full docstrings for
functions and modules.
"""

plspy_header = """
plspy
======

plspy is a Partial Least Squares package developed at the Institute for
Neuroscience and Neurotechnology at Simon Fraser University.

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


plspy_body = """
Basic usage examples:

    Note: There are 3 required arguments, used in the following order:

    1) X - 2-d task matrix

    2) a list containing the number of subjects in each group

    3) argument 3 is an int indicating the number of conditions


    Example arguments are used below.

    Mean-Centred Task PLS:

        >>> result = plspy.PLS(X, [10, 10], 3, num_perm=500, num_boot=500,  pls_method="mct")

    Behavioural PLS:

        >>> result = plspy.PLS(X, [10, 10], 3, Y=Y, pls_method="rb")

    Contrast Task PLS:

        >>> result = plspy.PLS(X, [10, 10], 3, contrasts=C, pls_method="cst")

    Contrast Behavioural PLS:

        >>> result = plspy.PLS(X, [10, 10], 3, Y=Y, contrasts=C, pls_method="csb")

    Multiblock PLS:

        >>> result = plspy.PLS(X, [10, 10], 3, Y=Y, pls_method="mb")

To see documentation on additional arguments and fields available,
call help on a specific PLS method (see below for details).

Documentation is available both in help() form and will also be available
in website form. More information on how to access online documentation is
forthcoming. Information on how to use help() is below.

To get help documentation on a particular version of PLS, type the following
in a Python interpreter after loading the module:
    >>> import plspy
    >>> help(plspy.methods["<methodname>"])

Where <method> is the string of one of the PLS versions shown below.

Available methods:

    "mct" - Mean-Centred Task PLS

    "rb"  - Regular Behaviour PLS

    "cst" - Contrast Task PLS

    "csb" - Contrast Behaviour PLS

    "mb"  - Multiblock PLS

    "cmb" - Contrast Multiblock PLS (under construction)


Note: calling
    >>> help(plspy.PLS)

will show you this same help page.


Author: Noah Frazier-Logue
"""

pls_wrapper_header = """
Front-facing wrapper function for PLS that captures user input
and extracts user-specified PLS method. If no method is specified,
default PLS is used.

TODO: implement first version of PLS and document required values here


"""
