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
pls_classes
    Source code for each PLS version
resample
    Functions used for resampling in permutation and bootstrap
bootstrap_permutation
    Houses the bootstrap and permutation object (called by PLS)
split_half_resampling
    Functions used for split-half and split-half test-train resampling
visualize
    Functions for plotting key outputs from PLS analyses
exceptions
    Houses custom exceptions used within PLS

"""


plspy_body = """

Usage Examples
==============

    All methods require:

    1. `X`: 2D data matrix. Rows should be ordered as subject within condition within group.
    2. `groups`: list with the number of subjects per group (e.g., [10, 10])
    3. `num_conditions`: integer specifying the number of conditions  

    Example arguments are used below.

    :class:`Mean-Centred Task PLS <plspy.pls_classes._MeanCentreTaskPLS>`
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, mctype=0, num_perm=500, num_boot=500, num_split=500, lv=1, CI=0.95, pls_method="mct")

    :class:`Behavioural PLS <plspy.pls_classes._RegularBehaviourPLS>`
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, pls_method="rb")

    :class:`Contrast Task PLS <plspy.pls_classes._ContrastTaskPLS>`
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, contrasts=C, pls_method="cst")

    :class:`Contrast Behavioural PLS <plspy.pls_classes._ContrastBehaviourPLS>`
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, contrasts=C, pls_method="csb")

    :class:`Multiblock PLS <plspy.pls_classes._MultiblockPLS>`
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, mctype=0, bscan=[1,2], pls_method="mb")

    :class:`Contrast Multiblock PLS <plspy.pls_classes._ContrastMultiblockPLS>`
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, mctype=0, bscan=[1,2], pls_method="cmb")

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

    "cmb" - Contrast Multiblock PLS


Note: calling
    >>> help(plspy.PLS)

will show you this same help page.


"""

pls_wrapper_header = """
Front-facing wrapper function for PLS that captures user input
and extracts user-specified PLS method. If no method is specified,
default PLS is used.

TODO: implement first version of PLS and document required values here


"""
