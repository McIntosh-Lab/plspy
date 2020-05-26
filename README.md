[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Rotman Institute Partial Least Squares


plsrri is a Partial Least Squares package developed at the Rotman                              
Research Institute at Baycrest Health Sciences.

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

* "mct" - Mean-Centred Task PLS
  
* "rb"  - Regular Behaviour PLS
  
* "cst" - Contrast Task PLS
  
* "csb" - Contrast Behaviour PLS
  
* "mb"  - Multiblock PLS
  
* "cmb" - Contrast Multiblock PLS (under construction)
