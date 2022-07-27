import abc

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns

from ..core import exceptions


class _SBPlotBase(abc.ABC):
    """Abstract base class and factory for PLS plots. Registers and
    keeps track of different defined methods/implementations of plots
    using Seaborn, and enforces use of base functions that all PLS
    plots should use.
    """

    # tracks registered SBPlotBase subclasses
    _subclasses = {}

    # maps abbreviated user-specified classnames to full SBPlotBase variant
    # names
    _sbplot_types = {
        "svs": "Singular Value Plot",
        "psvs": "Permuted Singular Values Probabilities Plot",
        "dlv": "Design LV Plot",
        "dsc": "Design Scores Plot",
        "bsc": "Brain Scores Plot",
        "vir": "Voxel Intensity Response Plot",
        "brlv": "Brain LV Plot",
        "belv": "Behaviour LV Plot",
        "cor": "Correlation Plot",
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
        kwargs["sbplot_method"] = sbplot_method
        return cls._subclasses[sbplot_method](*args, **kwargs)


@_SBPlotBase._register_subclass("svs")
class _SingularValuesPlot(_SBPlotBase):
    """ """

    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        self.dim = dim

        for k, v in kwargs.items():
            setattr(self, k, v)
        # self.pls_result = pls_result
        self.fig, self.ax = self._construct_plot(pls_result)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        f, ax = plt.subplots(figsize=(self.dim[0] * px, self.dim[1] * px))
        sv = pls_result.s
        pal = sns.color_palette("husl", n_colors=sv.shape[0])
        svt = pd.DataFrame(data={"x": list(range(1, len(sv) + 1)), "y": sv.reshape(-1)})
        bp = sns.barplot(data=svt, x="x", y="y", palette=pal)
        Ax = bp.axes
        boxes = [
            item
            for item in Ax.get_children()
            if isinstance(item, matplotlib.patches.Rectangle)
        ]
        ax.set(
            xlabel="Latent Variable",
            ylabel="Observed Singular Values",
            title="Observed Singular Values",
        )
        labels = [f"LV{svt['x'][i]}: {svt['y'][i]:.4f}" for i in range(len(svt["x"]))]
        patches = [
            matplotlib.patches.Patch(color=C, label=L)
            for C, L in zip([item.get_facecolor() for item in boxes], labels)
        ]
        bp.legend(
            handles=patches,
            bbox_to_anchor=(1, 1),
            loc=2,
            title="SVs",
            fontsize=8,
            handlelength=0.0,
        )

        return f, ax

    def plot(self):
        self.fig.show()

    def __str__(self):
        info = f"Plot type: {self._sbplot_types[self.sbplot_method]}"
        return info

    def __repr__(self):
        return self.__str__()


@_SBPlotBase._register_subclass("psvs")
class _PermutedSingularValuesPlot(_SingularValuesPlot):
    """ """

    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        f, ax = plt.subplots(figsize=(self.dim[0] * px, self.dim[1] * px))
        perm_sv = pls_result.resample_tests.permute_ratio
        pal = sns.color_palette("husl", n_colors=perm_sv.shape[0])
        svt = pd.DataFrame(
            data={"x": list(range(1, len(perm_sv) + 1)), "y": perm_sv.reshape(-1),}
        )
        bp = sns.barplot(data=svt, x="x", y="y", palette=pal)
        Ax = bp.axes
        boxes = [
            item
            for item in Ax.get_children()
            if isinstance(item, matplotlib.patches.Rectangle)
        ]
        ax.set(
            xlabel="Latent Variable",
            ylabel="Probability",
            title=f"Permuted values greater than observed, {pls_result.num_perm} permutation tests",
            ylim=[0, 1],
        )
        labels = [f"LV{svt['x'][i]}: {svt['y'][i]:.4f}" for i in range(len(svt["x"]))]
        patches = [
            matplotlib.patches.Patch(color=C, label=L)
            for C, L in zip([item.get_facecolor() for item in boxes], labels)
        ]
        bp.legend(
            handles=patches,
            bbox_to_anchor=(1, 1),
            loc=2,
            title="SVs",
            fontsize=8,
            handlelength=0.0,
        )

        return f, ax

    def plot(self):
        self.fig.show()


@_SBPlotBase._register_subclass("dsc")
class _DesignScoresPlot(_SingularValuesPlot):
    """ """

    def __init__(
        self, pls_result, lv=1, dim=(1000, 650), **kwargs,
    ):
        self.lv = lv
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        axes[0].set_ylabel("Design Scores")
        f.suptitle(f"LV {self.lv}", fontsize=14)
        splt = int(pls_result.U[self.lv - 1].shape[0] / pls_result.num_groups)
        bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            # stays in loop since it has to be reset every iteration
            pal = sns.color_palette("husl", n_colors=splt)
            scores.append(
                pd.DataFrame(
                    data={
                        "x": list(range(1, splt + 1)),
                        "y": pls_result.U[self.lv - 1][
                            i * splt : (i + 1) * splt
                        ].reshape(-1),
                    }
                )
            )
            bar_plots.append(
                sns.barplot(data=scores[i], x="x", y="y", palette=pal, ax=axes[i])
            )
            axes[i].set_xlabel(f"Group {i + 1}")
            axes[i].set_ylabel("")
        # pal = sns.color_palette("husl", n_colors=splt)
        # dv2 = pd.DataFrame(data={"x": list(range(1, splt + 1)), "y": dlv[0][splt:].reshape(-1)})
        # bp1 = sns.barplot(data=dv1, x="x", y="y", palette=pal, ax=ax1)
        # bp2 = sns.barplot(data=dv2, x="x", y="y", palette=pal, ax=ax2)
        # ax2.set_ylabel("")
        # ax1.set_xlabel("Group 1")
        # ax2.set_xlabel("Group 2")
        axes[0].set_ylabel("Design Scores")
        Ax = bar_plots[0].axes
        boxes = [
            item
            for item in Ax.get_children()
            if isinstance(item, matplotlib.patches.Rectangle)
        ]
        # axes[0].set(xlabel="Latent Variable", ylabel="Observed Singular Values", title="Observed Singular Values")
        labels = [f"c{scores[i]['x'][i]}" for i in range(len(scores[i]["x"]))]
        patches = [
            matplotlib.patches.Patch(color=C, label=L)
            for C, L in zip([item.get_facecolor() for item in boxes], labels)
        ]
        bar_plots[2].legend(
            handles=patches,
            bbox_to_anchor=(1, 1),
            loc=2,
            title="DSCs",
            fontsize=8,
            handlelength=0.5,
        )

        return f, axes

    def plot(self):
        self.fig.show()


@_SBPlotBase._register_subclass("cor")
class _CorrelationPlot(_SingularValuesPlot):
    """ """

    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        axes[0].set_ylabel("Correlations")
        f.suptitle("Correlation Plot", fontsize=14)
        splt = int(pls_result.lvcorrs.shape[0] / pls_result.num_groups)
        # bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            # stays in loop since it has to be reset every iteration
            pal = sns.color_palette("husl", n_colors=splt)
            y_mat = pls_result.lvcorrs[i * splt : (i + 1) * splt].reshape(-1)
            scores.append(
                pd.DataFrame(data={"x": list(range(1, y_mat.shape[0] + 1)), "y": y_mat})
            )
            axes[i].set_xlabel(f"Group {i + 1}")
            axes[i].set_ylabel("")
        # pal = sns.color_palette("husl", n_colors=splt)
        # dv2 = pd.DataFrame(data={"x": list(range(1, splt + 1)), "y": dlv[0][splt:].reshape(-1)})
        # bp1 = sns.barplot(data=dv1, x="x", y="y", palette=pal, ax=ax1)
        # bp2 = sns.barplot(data=dv2, x="x", y="y", palette=pal, ax=ax2)
        # ax2.set_ylabel("")
        # ax1.set_xlabel("Group 1")
        # ax2.set_xlabel("Group 2")

        # Ax = bar_plots[0].axes
        # boxes = [
        #     item
        #     for item in Ax.get_children()
        #     if isinstance(item, matplotlib.patches.Rectangle)
        # ]
        # # axes[0].set(xlabel="Latent Variable", ylabel="Observed Singular Values", title="Observed Singular Values")
        # labels = [f"c{scores[i]['x'][i]}" for i in range(len(scores[i]["x"]))]
        # patches = [
        #     matplotlib.patches.Patch(color=C, label=L)
        #     for C, L in zip([item.get_facecolor() for item in boxes], labels)
        # ]
        # bar_plots[2].legend(
        #     handles=patches,
        #     bbox_to_anchor=(1, 1),
        #     loc=2,
        #     title="Corrs",
        #     fontsize=8,
        #     handlelength=0.5,
        # )

        return f, axes

    def plot(self):
        self.fig.show()


@_SBPlotBase._register_subclass("brlv")
class _BrainLVPlot(_SingularValuesPlot):
    """ """

    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        axes[0].set_ylabel("Brain LVs")
        f.suptitle("Brain LV Plot", fontsize=14)
        splt = int(pls_result.lvcorrs.shape[0] / pls_result.num_groups)
        bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            # stays in loop since it has to be reset every iteration
            colours = int(pls_result.lvcorrs.shape[0] / pls_result.num_conditions)
            pal = sns.color_palette("husl", n_colors=colours)
            y_mat = pls_result.X_latent[i * splt : (i + 1) * splt].reshape(-1)
            scores.append(
                pd.DataFrame(
                    data={"x": list(range(1, y_mat.shape[0] + 1)), "y": y_mat,}
                )
            )
            bar_plots.append(
                sns.barplot(data=scores[i], x="x", y="y", palette=pal, ax=axes[i])
            )
            axes[i].set_xlabel(f"Group {i + 1}")
            axes[i].set_ylabel("")
        # pal = sns.color_palette("husl", n_colors=splt)
        # dv2 = pd.DataFrame(data={"x": list(range(1, splt + 1)), "y": dlv[0][splt:].reshape(-1)})
        # bp1 = sns.barplot(data=dv1, x="x", y="y", palette=pal, ax=ax1)
        # bp2 = sns.barplot(data=dv2, x="x", y="y", palette=pal, ax=ax2)
        # ax2.set_ylabel("")
        # ax1.set_xlabel("Group 1")
        # ax2.set_xlabel("Group 2")

        # Ax = bar_plots[0].axes
        # boxes = [
        #     item
        #     for item in Ax.get_children()
        #     if isinstance(item, matplotlib.patches.Rectangle)
        # ]
        # # axes[0].set(xlabel="Latent Variable", ylabel="Observed Singular Values", title="Observed Singular Values")
        # labels = [f"c{scores[i]['x'][i]}" for i in range(len(scores[i]["x"]))]
        # patches = [
        #     matplotlib.patches.Patch(color=C, label=L)
        #     for C, L in zip([item.get_facecolor() for item in boxes], labels)
        # ]
        # bar_plots[2].legend(
        #     handles=patches,
        #     bbox_to_anchor=(1, 1),
        #     loc=2,
        #     title="Corrs",
        #     fontsize=8,
        #     handlelength=0.5,
        # )

        return f, axes

    def plot(self):
        self.fig.show()


@_SBPlotBase._register_subclass("belv")
class _BehavLVPlot(_SingularValuesPlot):
    """ """

    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        axes[0].set_ylabel("Behaviour LVs")
        f.suptitle("Behaviour LV Plot", fontsize=14)
        splt = int(pls_result.Y_latent.shape[0] / pls_result.num_groups)
        bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            # stays in loop since it has to be reset every iteration
            colours = int(pls_result.lvcorrs.shape[0] / pls_result.num_conditions)
            pal = sns.color_palette("husl", n_colors=colours)
            y_mat = pls_result.Y_latent[i * splt : (i + 1) * splt].reshape(-1)
            scores.append(
                pd.DataFrame(
                    data={"x": list(range(1, y_mat.shape[0] + 1)), "y": y_mat,}
                )
            )
            bar_plots.append(
                sns.barplot(data=scores[i], x="x", y="y", palette=pal, ax=axes[i])
            )
            axes[i].set_xlabel(f"Group {i + 1} conditions")
            axes[i].set_ylabel("")
        # pal = sns.color_palette("husl", n_colors=splt)
        # dv2 = pd.DataFrame(data={"x": list(range(1, splt + 1)), "y": dlv[0][splt:].reshape(-1)})
        # bp1 = sns.barplot(data=dv1, x="x", y="y", palette=pal, ax=ax1)
        # bp2 = sns.barplot(data=dv2, x="x", y="y", palette=pal, ax=ax2)
        # ax2.set_ylabel("")
        # ax1.set_xlabel("Group 1")
        # ax2.set_xlabel("Group 2")

        # Ax = bar_plots[0].axes
        # boxes = [
        #     item
        #     for item in Ax.get_children()
        #     if isinstance(item, matplotlib.patches.Rectangle)
        # ]
        # # axes[0].set(xlabel="Latent Variable", ylabel="Observed Singular Values", title="Observed Singular Values")
        # labels = [f"c{scores[i]['x'][i]}" for i in range(len(scores[i]["x"]))]
        # patches = [
        #     matplotlib.patches.Patch(color=C, label=L)
        #     for C, L in zip([item.get_facecolor() for item in boxes], labels)
        # ]
        # bar_plots[2].legend(
        #     handles=patches,
        #     bbox_to_anchor=(1, 1),
        #     loc=2,
        #     title="Corrs",
        #     fontsize=8,
        #     handlelength=0.5,
        # )

        return f, axes

    def plot(self):
        self.fig.show()


@_SBPlotBase._register_subclass("vir")
class _VoxelIntensityPlot(_DesignScoresPlot):
    """ """

    def __init__(
        self, pls_result, coords, dim=(1000, 650), **kwargs,
    ):
        self.coords = coords
        super().__init__(self, pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        pass

    def plot(self):
        self.fig.show()

    def mean_neighbourhood(mat, pos, num):
        """
        TODO: add masking/threshold functionality
        """
        if num == 0:
            return mat[pos[0], pos[1], pos[2]]

        x, y, z = pos[0], pos[1], pos[2]
        # creates 3d cube around central index that is neighbourhood
        nhood = mat[x - num - 1 : x + num, y - num - 1 : y + num, z - num - 1 : z + num]
        nsum = np.sum(nhood)
        avg = nsum / (nhood.shape[0] * nhood.shape[1] * nhood.shape[2])

        return avg


@_SBPlotBase._register_subclass("blv")
class _BLVPlot(_SBPlotBase):
    """ """

    def __init__(
        self, pls_result, threshold, dim=(1000, 650), **kwargs,
    ):
        super().__init__(pls_result, dim, **kwargs)
        self.threshold = threshold

    def _construct_plot(self, pls_result, **kwargs):
        pass

    def plot(self):
        self.fig.show()

    def save_html(self):
        pass

    def __str__(self):
        info = f"Plot type: {self._sbplot_types[self.sbplot_method]}"
        # stg += info
        # return stg
        return info

    def __repr__(self):
        return self.__str__()
