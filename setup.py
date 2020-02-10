import setuptools

# import os

# fn = os.path.join(os.path.dirname(__file__), "README.md")

# with open(fn, "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="plsrri",  # Replace with your own username
    version="0.1a1",
    author="Noah Frazier-Logue",
    author_email="nfrazier-logue@research.baycrest.org",
    description="Implementation of Partial Least Squares c/o Baycrest's Rotman Research Institute",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/noahfl/PartialLeastSquares",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
)
