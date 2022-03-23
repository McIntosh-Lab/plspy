from pybuilder.core import use_plugin, init, Author

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.coverage")
use_plugin("python.distutils")

name = "plspy"
default_task = ["install_dependencies", "analyze", "publish"]
version = 0.2
summary = (
    "Partial Least Squares implementation in Python, developed by "
    "Noah Frazier-Logue - Institute for Neuroscience and "
    "Neurotechnology, Simon Fraser University"
)
authors = [Author("Noah Frazier-Logue", "noah_frazier-logue@sfu.ca")]

requires_python = ">=3.8"


@init
def set_properties(project):
    project.depends_on("numpy")
    project.depends_on("scipy")
    project.set_property("flake8_include_scripts", True)
    project.set_property("flake8_include_test_sources", True)
    project.set_property("flake8_break_build", False)
