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
        "bscvbe": "Brain Scores vs Behaviour Plot",
        "tbsc": "Task PLS Brain Score Plot",
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
        bp = sns.barplot(data=svt, x="x", y="y", hue = "x",palette=pal)
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
        bp = sns.barplot(data=svt, x="x", y="y", hue = "x",palette=pal)
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


@_SBPlotBase._register_subclass("dlv")
class _DesignLVPlot(_SingularValuesPlot):
    """ """

    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        self.lv = kwargs.pop("lv", 1)
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        if pls_result.num_groups >1:
            axes[0].set_ylabel("Design Scores")
        else:
            axes.set_ylabel("Design Scores")
        f.suptitle(f"LV {self.lv}", fontsize=14)
        splt = int(pls_result.V.T[self.lv - 1].shape[0] / pls_result.num_groups)
        bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            # stays in loop since it has to be reset every iteration
            pal = sns.color_palette("husl", n_colors=splt)
            scores.append(
                pd.DataFrame(
                    data={
                        "x": list(range(1, splt + 1)),
                        "y": pls_result.V.T[self.lv - 1][
                            i * splt : (i + 1) * splt
                        ].reshape(-1),
                    }
                )
            )
            if pls_result.num_groups >1:
                bar_plots.append(
                sns.barplot(data=scores[i], x="x", y="y", hue = "x", palette=pal, ax=axes[i])
                )
                axes[i].set_xlabel(f"Group {i + 1}")
                axes[i].set_ylabel("")
            else:
                bar_plots.append(
                sns.barplot(data=scores[i], x="x", y="y", hue = "x", palette=pal, ax=axes)
                )
                axes.set_xlabel(f"Group {i + 1}")
                axes.set_ylabel("")

        # pal = sns.color_palette("husl", n_colors=splt)
        # dv2 = pd.DataFrame(data={"x": list(range(1, splt + 1)), "y": dlv[0][splt:].reshape(-1)})
        # bp1 = sns.barplot(data=dv1, x="x", y="y", palette=pal, ax=ax1)
        # bp2 = sns.barplot(data=dv2, x="x", y="y", palette=pal, ax=ax2)
        # ax2.set_ylabel("")
        # ax1.set_xlabel("Group 1")
        # ax2.set_xlabel("Group 2")
        if pls_result.num_groups >1:
            axes[0].set_ylabel("Design Scores")
            Ax = bar_plots[0].axes
        else:
            axes.set_ylabel("Design Scores")
            Ax = bar_plots[0].axes    

        boxes = [
            item
            for item in Ax.get_children()
            if isinstance(item, matplotlib.patches.Rectangle)
        ]
        # axes[0].set(xlabel="Latent Variable", ylabel="Observed Singular Values", title="Observed Singular Values")
        if pls_result.num_groups >1:
            labels = [f"c{scores[i]['x'][j]}" for j in range(len(scores[i]["x"]))]
        else:
            labels = [f"c{scores[i]['x'][j]}" for j in range(len(scores[i]["x"]))]
        patches = [
            matplotlib.patches.Patch(color=C, label=L)
            for C, L in zip([item.get_facecolor() for item in boxes], labels)
        ]
        bar_plots[i].legend(
            handles=patches,
            bbox_to_anchor=(1, 1),
            loc=i,
            title="DSCs",
            fontsize=8,
            handlelength=0.5,
        )

        return f, axes

    def plot(self):
        self.fig.show()


@_SBPlotBase._register_subclass("dsc")  # Updated to make a scatter plot
class _DesignScoresPlot(_SingularValuesPlot):
    """Scatter Plot for Design Scores (V) vs X Latents."""

    def __init__(self, pls_result, dim=(1000, 650), **kwargs):
        self.lv = kwargs.pop("lv", 1)
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        f, ax = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px)
        )

        ax.set_xlabel("Design Scores (V)")
        ax.set_ylabel("Brain Scores (X Latents)")
        ax.set_title(f"Scatter Plot for LV {self.lv}")

        # Extract y-axis data
        y = pls_result.X_latent.T[self.lv - 1]

        # Extract x-axis data
        original_x = pls_result.V.T[self.lv - 1]  # Original x values (group-level)
        repeated_x = []
        x_counter = 0  # Counter to track the position in original_x
        
        for group_cond in pls_result.cond_order:
            for num_subs in group_cond:
                # Use the counter to access the correct index in original_x
                repeated_x.extend([original_x[x_counter]] * num_subs)
                x_counter += 1  # Increment the counter

        # Create a DataFrame for easier handling of conditions
        data = pd.DataFrame({
            "x": repeated_x,
            "y": y,
            "condition": [
                f"Group {group_idx + 1} Condition {cond_idx + 1}"
                for group_idx, group_cond in enumerate(pls_result.cond_order)
                for cond_idx, num_subs in enumerate(group_cond)
                for _ in range(num_subs)
            ]
        })

        # Assign unique markers and colors for each condition
        markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "d", "h", "H", "+", "x", "|", "_"]
        palette = sns.color_palette("husl", n_colors=len(data["condition"].unique()))

        # Plot each condition with a unique marker and color
        for (condition, marker, color) in zip(
            data["condition"].unique(), markers, palette
        ):
            subset = data[data["condition"] == condition]
            ax.scatter(
                subset["x"], subset["y"],
                label=condition,
                marker=marker,
                color=color,
                edgecolor="black",
                s=50,  # Marker size
            )

        ax.legend(title="Conditions", fontsize=8, loc="best")
        return f, ax

    def plot(self):
        self.fig.show()

@_SBPlotBase._register_subclass("tbsc")
# Add error handling for plotting behavioural PLS
class _TaskPLSBrainScorePlot(_SingularValuesPlot):
    """ """
    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        self.lv = kwargs.pop("lv", 1)
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        if pls_result.num_groups > 1:
            axes[0].set_ylabel("Brain Scores")
        else:
            axes.set_ylabel("Brain Scores")
        f.suptitle(f"LV {self.lv}", fontsize=14)
        
        bar_plots = []
        scores = []
        x_counter = 0  # Counter to track data point indices in X_latent
        num_conditions= np.shape(pls_result.cond_order)[1]
        for group_idx, group_cond in enumerate(pls_result.cond_order):
            pal = [f"cond{i + 1}" for i in range(num_conditions)]
            # Extract data for the current group
            group_data = pls_result.X_latent.T[self.lv - 1][
                x_counter : x_counter + sum(group_cond)
            ]
            x_counter += sum(group_cond)  # Move the counter forward for the next group

            #Get confidence intervals
            ci_values = []
            condition_means = []
            for cond_idx in range(len(group_cond)):
                # Extract data for the current condition
                condition_data = group_data[
                    sum(group_cond[:cond_idx]) : sum(group_cond[:cond_idx + 1])
                ]

                # Compute the mean and confidence intervals for the mean
                condition_means.append(condition_data.mean())

                # Compute 5th and 95th percentiles
                lower_percentile = np.percentile(condition_data, 5)  # 5th percentile
                upper_percentile = np.percentile(condition_data, 95)  # 95th percentile

                # Store as (lower bound, upper bound)
                ci_values.append((condition_data.mean() - lower_percentile, upper_percentile - condition_data.mean()))

            # Create DataFrame for plotting
            scores.append(
                pd.DataFrame(
                    data={
                        "x": list(range(1, len(group_cond) + 1)),  # Condition numbers
                        "y": condition_means,
                    }
                )
            )

            # Plot for each group
            if pls_result.num_groups > 1:
                legend_flag = False
                if group_idx == pls_result.num_groups-1:
                    legend_flag = True
                bar_plots.append(
                    sns.barplot(
                        data=scores[group_idx],
                        x="x",
                        y="y",
                        hue = pal,
                        #palette=pal,
                        ax=axes[group_idx],
                        errorbar=None,
                        legend = legend_flag
                    )
                )
 
                if group_idx == pls_result.num_groups-1:                
                    sns.move_legend(bar_plots[pls_result.num_groups-1], "upper left", bbox_to_anchor=(1.1, 1))

                for j in range(len(group_cond)):
                    axes[group_idx].errorbar(
                        j,
                        scores[group_idx]["y"][j],
                        yerr=[[ci_values[j][0]], [ci_values[j][1]]],
                        fmt="none",
                        capsize=5,
                        color="black",
                    )

                axes[group_idx].set_xlabel(f"Group {group_idx + 1}")
                axes[group_idx].set_ylabel("")
            else:
                bar_plots.append(
                    sns.barplot(
                        data=scores[group_idx],
                        x="x",
                        y="y",
                        hue = pal,
                        #palette=pal,
                        ax=axes,
                        errorbar=None,
                    )
                )
                sns.move_legend(bar_plots[pls_result.num_groups-1], "upper left", bbox_to_anchor=(1, 1))  

                for j in range(len(group_cond)):
                    axes.errorbar(
                        j,
                        scores[group_idx]["y"][j],
                        yerr=[[ci_values[j][0]],
                            [ci_values[j][1]]],
                        fmt="none",
                        capsize=5,
                        color="black",
                    )
                axes.set_xlabel(f"Group {group_idx + 1}")
                axes.set_ylabel("")

        if pls_result.num_groups > 1:
            axes[0].set_ylabel("Brain Scores")
            Ax = bar_plots[0].axes
        else:
            axes.set_ylabel("Brain Scores")
            Ax = bar_plots[0].axes

        return f, axes

    def plot(self):
        self.fig.show()

@_SBPlotBase._register_subclass("cor")
class _CorrelationPlot(_SingularValuesPlot):
    """ """
    def __init__(
        self, pls_result, dim=(1000, 650), **kwargs,
    ):
        self.lv = kwargs.pop("lv", 1)
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        if pls_result.num_groups > 1:
            axes[0].set_ylabel("Correlation")
        else:
            axes.set_ylabel("Correlation")
        f.suptitle(f"LV {self.lv}", fontsize=14)

        lv_corr = pls_result.lvcorrs.T[self.lv - 1]
        num_behaviours = int(np.size(lv_corr)/(np.size(pls_result.cond_order)))
        num_conditions= np.shape(pls_result.cond_order)[1]
        splt = int(lv_corr.shape[0] / pls_result.num_groups) # number of conditions * behaviours per group, number of bars in each group sub-plot
        bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            pal = [f"cond{i + 1}" for i in range(num_conditions) for _ in range(num_behaviours)][:splt]

            # Generate labels for x-axis
            behaviors = [f"behav{j+1}" for j in range(num_behaviours)]
            x_labels = [f"{behav}" for cond in range(1, num_conditions +1) for behav in behaviors]

            scores.append(
                pd.DataFrame(
                    data={
                        "x": list(range(1, splt + 1)),
                        "y": lv_corr[
                            i * splt : (i + 1) * splt
                        ].reshape(-1),
                    }
                )
            )

            has_conf_ints = (
                hasattr(pls_result, "resample_tests")
                and hasattr(pls_result.resample_tests, "conf_ints")
            )
            if has_conf_ints:
                lower_ci = pls_result.resample_tests.conf_ints[0].T[self.lv - 1][
                    i * splt : (i + 1) * splt
                ]
                upper_ci = pls_result.resample_tests.conf_ints[1].T[self.lv - 1][
                    i * splt : (i + 1) * splt
                ]

                ci_values = [                  
                    (scores[i]["y"][j] - lower_ci[j], upper_ci[j] - scores[i]["y"][j])
                    for j in range(len(lower_ci))
                ]

                # Handling of negative error bar values
                for j in range(len(lower_ci)):
                    if ci_values[j][0] <0 or ci_values[j][1]<0:
                        ci_values[j]=(0,0)
                        scores[i]["y"][j]=0
                        print(f"ERROR: Bar #{j+1} in Group {i+1} has invalid confidence intervals. Bar and errors set to zero. Do not use data for that group and condition.")

            if pls_result.num_groups > 1:
                legend_flag = False
                if i == pls_result.num_groups-1:
                    legend_flag = True
                bar_plots.append(
                    sns.barplot(
                        data=scores[i],
                        x="x",
                        y="y",
                        hue = pal,
                        #palette=pal,
                        ax=axes[i],
                        errorbar=None,
                        legend = legend_flag   
                    )
                )
                if i == pls_result.num_groups-1:                
                    sns.move_legend(bar_plots[pls_result.num_groups-1], "upper left", bbox_to_anchor=(1.1, 1))                
                if has_conf_ints:
                    for j in range(splt):
                        axes[i].errorbar(
                            j,
                            scores[i]["y"][j],
                            yerr=[[ci_values[j][0]], [ci_values[j][1]]],
                            fmt="none",
                            capsize=5,
                            color="black",
                        )
                axes[i].set_xlabel(f"Group {i + 1}")
                axes[i].set_xticks(range(len(x_labels[:splt])))
                axes[i].set_xticklabels(x_labels[:splt], rotation=45, ha="right")
                axes[i].set_ylabel("")
            else:
                bar_plots.append(
                    sns.barplot(
                        data=scores[i],
                        x="x",
                        y="y",
                        hue = pal,
                        #palette=pal,
                        ax=axes,
                        errorbar=None,
                    )
                )
                sns.move_legend(bar_plots[pls_result.num_groups-1], "upper left", bbox_to_anchor=(1, 1))   
                if has_conf_ints:
                    for j in range(splt):
                        axes.errorbar(
                            j,
                            scores[i]["y"][j],
                            yerr=[[ci_values[j][0]], [ci_values[j][1]]],
                            fmt="none",
                            capsize=5,
                            color="black",
                        )
                axes.set_xlabel(f"Group {i + 1}")
                axes.set_xticks(range(len(x_labels[:splt]))) # Set the ticks
                axes.set_xticklabels(x_labels[:splt], rotation=45, ha="right")
                axes.set_ylabel("")

        if pls_result.num_groups > 1:
            axes[0].set_ylabel("Correlation")
            Ax = bar_plots[0].axes
        else:
            axes.set_ylabel("Correlation")
            Ax = bar_plots[0].axes

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
        self.lv = kwargs.pop("lv", 1)
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]
        axes = () * pls_result.num_groups
        f, axes = plt.subplots(
            figsize=(self.dim[0] * px, self.dim[1] * px),
            ncols=pls_result.num_groups,
            sharey=True,
        )
        if pls_result.num_groups >1:
            axes[0].set_ylabel("Behaviour LV")
        else:
            axes.set_ylabel("Behaviour LV")
        f.suptitle(f"LV {self.lv}", fontsize=14)

        num_behaviours = int(np.size(pls_result.V[self.lv - 1])/(np.size(pls_result.cond_order)))
        num_conditions= np.shape(pls_result.cond_order)[1]

        splt = int(pls_result.V.T[self.lv - 1].shape[0] / pls_result.num_groups) # number of conditions * behaviours per group, number of bars in each group sub-plot
        bar_plots = []
        scores = []
        for i in range(pls_result.num_groups):
            pal = [f"cond{i + 1}" for i in range(num_conditions) for _ in range(num_behaviours)][:splt]
            
            # Generate labels for x-axis
            behaviors = [f"behav{j+1}" for j in range(num_behaviours)]
            x_labels = [f"{behav}" for cond in range(1, num_conditions +1) for behav in behaviors]

            scores.append(
                pd.DataFrame(
                    data={
                        "x": list(range(1, splt + 1)),
                        "y": pls_result.V.T[self.lv - 1][
                            i * splt : (i + 1) * splt
                        ].reshape(-1),
                    }
                )
            )
            if pls_result.num_groups >1:
                legend_flag = False
                if i == pls_result.num_groups-1:
                    legend_flag = True
                bar_plots.append(
                sns.barplot(
                    data=scores[i], 
                    x="x", 
                    y="y", 
                    hue = pal, 
                    ax=axes[i],
                    legend=legend_flag)
                )
                if i == pls_result.num_groups-1:                
                    sns.move_legend(bar_plots[pls_result.num_groups-1], "upper left", bbox_to_anchor=(1.1, 1))
                axes[i].set_xlabel(f"Group {i + 1}")
                axes[i].set_xticks(range(len(x_labels[:splt])))
                axes[i].set_xticklabels(x_labels[:splt], rotation=45, ha="right")
                axes[i].set_ylabel("")
            else:
                bar_plots.append(
                sns.barplot(data=scores[i], x="x", y="y", hue = pal, ax=axes)
                )
                sns.move_legend(bar_plots[pls_result.num_groups-1], "upper left", bbox_to_anchor=(1, 1)) 
                axes.set_xlabel(f"Group {i + 1}")
                axes.set_xticks(range(len(x_labels[:splt])))
                axes.set_xticklabels(x_labels[:splt], rotation=45, ha="right")
                axes.set_ylabel("")

        if pls_result.num_groups >1:
            axes[0].set_ylabel("Behaviour LV")
            Ax = bar_plots[0].axes
        else:
            axes.set_ylabel("Behaviour LV")
            Ax = bar_plots[0].axes    

        # boxes = [
        #     item
        #     for item in Ax.get_children()
        #     if isinstance(item, matplotlib.patches.Rectangle)
        # ]
        #axes[0].set(xlabel="Latent Variable", ylabel="Observed Singular Values", title="Observed Singular Values")
        # if pls_result.num_groups >1:
        #     labels = [f"c{scores[i]['x'][j]}" for j in range(len(scores[i]["x"]))]
        # else:
        #     labels = [f"c{scores[i]['x'][j]}" for j in range(len(scores[i]["x"]))]


        # patches = [
        #     matplotlib.patches.Patch(color=C, label=L)
        #     for C, L in zip([item.get_facecolor() for item in boxes], labels)
        # ]

        return f, axes

    def plot(self):
        self.fig.show()

@_SBPlotBase._register_subclass("bscvbe")
class _BrainScorevsBehavPlot(_SingularValuesPlot):
    """Scatter Plot for Brain Scores vs Behaviour Data."""

    def __init__(self, pls_result, dim=(1000, 650), **kwargs):
        self.lv = kwargs.pop("lv", 1)  # Latent variable (default LV1)
        self.groups_of_interest = kwargs.pop("group", [1])  # List of groups
        self.conditions_of_interest = kwargs.pop("condition", [1])  # List of conditions
        self.behaviours_of_interest = kwargs.pop("behaviour", [1])  # List of behaviours
        super().__init__(pls_result, dim, **kwargs)

    def _construct_plot(self, pls_result, **kwargs):
        px = 1 / plt.rcParams["figure.dpi"]

        lv_corr = pls_result.lvcorrs.T[self.lv - 1]
        num_behaviours = int(np.size(lv_corr)/(np.size(pls_result.cond_order)))
        num_conditions= np.shape(pls_result.cond_order)[1]
        num_groups = np.shape(pls_result.cond_order)[0]
        
        num_groups_plot = len(self.groups_of_interest)
        num_conditions_plot = len(self.conditions_of_interest)
        num_behaviours_plot = len(self.behaviours_of_interest)
        total_columns = num_conditions_plot * num_behaviours_plot
        px = px * max(total_columns,num_groups_plot) * 1.2

        f, axes = plt.subplots(
            num_groups_plot, total_columns,
            figsize=(self.dim[0] * px * total_columns / 4, self.dim[1] * px * num_groups_plot / 4),
            squeeze=False
        )

        palette = sns.color_palette(
            "husl", 
            n_colors=num_groups * num_conditions * num_behaviours
        )

        for g_idx, group in enumerate(self.groups_of_interest):
            for c_idx, condition in enumerate(self.conditions_of_interest):
                for b_idx, behaviour in enumerate(self.behaviours_of_interest):
                    ax = axes[g_idx, c_idx * num_behaviours + b_idx]

                    num_behaviour = int(np.size(lv_corr) / (np.size(pls_result.cond_order)))
                    num_condition = np.shape(pls_result.cond_order)[1]

                    corr_of_interest = lv_corr[
                        ((group - 1) * num_behaviour * num_condition) + 
                        ((condition - 1) * num_behaviour) + 
                        (behaviour - 1)
                    ]

                    ax.set_xlabel(f"Behaviour ({behaviour})")
                    ax.set_ylabel("Brain Scores")
                    ax.set_title(f"Group {group}, Condition {condition}\nLV {self.lv} r = {corr_of_interest:.2f}")

                    original_x = pls_result.Y  # Behavioral data
                    num_subjects_of_interest = pls_result.cond_order[group - 1, condition - 1]
                    start_idx = np.sum(pls_result.cond_order[:group - 1, :]) + \
                        np.sum(pls_result.cond_order[group - 1, :condition - 1])

                    selected_x = original_x[start_idx:start_idx + num_subjects_of_interest, behaviour - 1]

                    y_lv = pls_result.X_latent.T[self.lv - 1]
                    y = y_lv[start_idx:start_idx + num_subjects_of_interest]

                    data = pd.DataFrame({
                        "x": selected_x,
                        "y": y,
                        "subject": range(start_idx + 1, start_idx + len(selected_x) + 1),
                    })

                    colour_index = (
                        (group - 1) * num_behaviour * num_condition +
                        (condition - 1) * num_behaviour +
                        (behaviour - 1)
                    )

                    selected_colour = palette[colour_index]

                    ax.scatter(
                        data["x"], data["y"],
                        color=selected_colour,
                        edgecolor="black",
                        s=50,  # Marker size
                    )

                    for i, row in data.iterrows():
                        ax.text(
                            row["x"], row["y"], str(row["subject"]),
                            fontsize=8, color="black", ha="right", va="bottom"
                        )

        plt.tight_layout()
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
