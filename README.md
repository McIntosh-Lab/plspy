[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![CircleCI](https://circleci.com/gh/McIntosh-Lab/plspy/tree/main.svg?style=svg&circle-token=3b9c7e2a597b381d8b388e0fae83552ee89e07d3)](https://circleci.com/gh/McIntosh-Lab/plspy/tree/main) [![Documentation Status](https://readthedocs.org/projects/plspy/badge/?version=latest)](https://plspy.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/plspy.svg)](https://badge.fury.io/py/plspy) ![versions](https://img.shields.io/pypi/pyversions/pybadges.svg) [![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/McIntosh-Lab/plspy/blob/master/LICENSE) 

# Partial Least Squares - McIntosh Lab


plspy is a Partial Least Squares package developed to replicate and extend the PLS MATLAB package created by Randy McIntosh, et al for use in neuroimaging applications.
.

Checkout the documentation for `plspy` at https://plspy.readthedocs.io/en/latest/



## Installation

The following steps will download and install plspy to your computer:

`pip install plspy`

If you prefer to build from source, run these commands:

```
git clone https://github.com/McIntosh-LabI/plspy.git
cd plspy
python setup.py install
```


## Usage

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

* "mct" - Mean-Centred Task PLS
  
* "rb"  - Regular Behaviour PLS
  
* "cst" - Contrast Task PLS
  
* "csb" - Contrast Behaviour PLS
  
* "mb"  - Multiblock PLS
  
* "cmb" - Contrast Multiblock PLS (under construction)
