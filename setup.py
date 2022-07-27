import setuptools
import versioneer

version = (versioneer.get_version(),)
cmdclass = (versioneer.get_cmdclass(),)

# import os

# fn = os.path.join(os.path.dirname(__file__), "README.md")

# with open(fn, "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="plspy",  
    version="0.3.0",
    author="Noah Frazier-Logue",
    author_email="noah_frazier-logue@sfu.ca",
    description="Implementation of McIntosh Lab's Partial Least Squares "
    "neuroimaging tool",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/McIntosh-Lab/plspy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
