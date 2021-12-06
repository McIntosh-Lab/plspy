import abc

import plsrri
import numpy as np
import seaborn as sns
import nilearn
import nibabel as nib
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib


class SBPlotBase(abc.ABC):
    """Abstract base class and factory for Seaborn plots. Registers and
    keeps track of different defined methods/implementations of plots
    using Seaborn, and enforces use of base functions that all Seaborn
    plots should use.
    """

    # tracks registered SBPlotBase subclasses
    _subclasses = {}

    # maps abbreviated user-specified classnames to full SBPlotBase variant
    # names
    _sbplot_types = {
        "dlv": "Design LV Plot",
        "dsc": "Design Scores Plot",
        "bsc": "Brain Scores Plot",
    }

    @abc.abstractmethod
    def _construct_plot(self):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    # register valid decorated PLS method as a subclass of SBPlotBase
    @classmethod
    def _register_subclass(cls, sbplot_method):
        def decorator(subclass):
            cls._subclasses[sbplot_method] = subclass
            return subclass

        return decorator

    # instantiate and return valid registered PLS method specified by user
    @classmethod
    def _create(cls, sbplot_method, *args, **kwargs):
        if sbplot_method not in cls._subclasses and sbplot_method in cls._sbplot_types:
            raise exceptions.NotImplementedError(
                f"Specified SBPlotBase method {cls._sbplot_types[sbplot_method]} "
                "has not yet been implemented."
            )
        elif sbplot_method not in cls._subclasses:
            raise ValueError(f"Invalid SBplotBase method {sbplot_method}")
        return cls._subclasses[sbplot_method](*args, **kwargs)


@SBPlotBase._register_subclass("dlv")
class DesignScoresPlot(SBPlotBase):
    """
    """

    def __init__(
        self, dim=(800, 600), **kwargs,
    ):
        self.dim = dim
        self.fig, self.ax = self._construct_plot()
        self.plot()

    def _construct_plot(self, **kwargs):
        pass

    def plot(self, matrix):
        pass

    def __str__(self):
        stg = ""
        info = f"Seaborn plot type: {self._sbplot_types[self.sbplot_method]}"
        stg += info
        return stg

    def __repr__(self):
        return self.__str__()
