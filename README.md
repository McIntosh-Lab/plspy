[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Rotman Institute Partial Least Squares


plsrri is a Partial Least Squares package developed at the Rotman                              
Research Institute at Baycrest Health Sciences.

Basic usage examples:
    Note: There are 3 required arguments, used in the following order:
    
    1) X - 2-d task matrix
    
    2) a list/tuple object containing the number of subjects in each group
    
    3) argument 3 is an int indicating the number of conditions
    
    Example arguments are used below.
    
    Mean-Centred Task PLS:
        
        >>> result = plsrri.PLS(X, (10, 10), 3, num_perm=500, num_boot=500,  pls_method="mct")
    Behavioural PLS:
    
        >>> result = plsrri.PLS(X, (10, 10), 3, Y=Y, pls_method="rb")
    Contrast Task PLS:
        
        >>> result = plsrri.PLS(X, (10, 10), 3, contrasts=C, pls_method="cst")
    Contrast Behavioural PLS:
        
        >>> result = plsrri.PLS(X, (10, 10), 3, Y=Y, contrasts=C, pls_method="csb")
    Multiblock PLS:
        
        >>> result = plsrri.PLS(X, (10, 10), 3, Y=Y, pls_method="mb")
        
        
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

* "mct" - Mean-Centred Task PLS
  
* "rb"  - Regular Behaviour PLS
  
* "cst" - Contrast Task PLS
  
* "csb" - Contrast Behaviour PLS
  
* "mb"  - Multiblock PLS
  
* "cmb" - Contrast Multiblock PLS (under construction)
