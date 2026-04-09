[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![CircleCI](https://circleci.com/gh/McIntosh-Lab/plspy/tree/main.svg?style=svg&circle-token=3b9c7e2a597b381d8b388e0fae83552ee89e07d3)](https://circleci.com/gh/McIntosh-Lab/plspy/tree/main) [![Documentation Status](https://readthedocs.org/projects/plspy/badge/?version=latest)](https://plspy.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/plspy.svg)](https://badge.fury.io/py/plspy) ![versions](https://img.shields.io/pypi/pyversions/pybadges.svg) [![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/McIntosh-Lab/plspy/blob/master/LICENSE) 

# Partial Least Squares - McIntosh Lab


`plspy` is a Partial Least Squares package developed to replicate and extend the PLS MATLAB package created by Randy McIntosh, et al for use in neuroimaging applications.

Check out the documentation for `plspy` at https://plspy.readthedocs.io/en/latest/



## Installation

The following steps will download and install plspy to your computer:

`pip install plspy`

> **Note:** Currently, the version available via `pip install` is not the most up to date. For the latest features, bug fixes, and development updates, we recommend installing from source.
To build from source, run these commands:

```
git clone https://github.com/McIntosh-Lab/plspy.git
cd plspy
python setup.py install
```


## Usage

### Required inputs

All methods require:

1. `X`: 2D data matrix. Rows should be ordered as subject within condition within group.
2. `groups`: list with the number of subjects per group (e.g., [10, 10])
3. `num_conditions`: integer specifying the number of conditions  

---

### Basic usage examples:

    Mean-Centred Task PLS:
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, mctype=0, num_perm=500, num_boot=500, num_split=500, lv=1, CI=0.95, pls_method="mct")

    Behavioural PLS:
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, pls_method="rb")

    Contrast Task PLS:
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, contrasts=C, pls_method="cst")

    Contrast Behavioural PLS:
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, contrasts=C, pls_method="csb")

    Multiblock PLS:
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, mctype=0, bscan=[1,2], pls_method="mb")

    Contrast Multiblock PLS:
        >>> result = plspy.PLS(X, [10, 10], num_conditions=3, Y=Y, mctype=0, bscan=[1,2], pls_method="cmb")

To see documentation on additional arguments and fields available, see https://plspy.readthedocs.io/en/latest/index.html. Alternatively,
call `help()` on a specific PLS method by typing the following in a Python interpreter:

    >>> import plspy
    >>> help(plspy.methods["<methodname>"])

Where `<method>` is the string of one of the PLS versions shown below.
  
Available methods:

* "mct" - Mean-Centred Task PLS
  
* "rb"  - Regular Behaviour PLS
  
* "cst" - Contrast Task PLS
  
* "csb" - Contrast Behaviour PLS
  
* "mb"  - Multiblock PLS
  
* "cmb" - Contrast Multiblock PLS
