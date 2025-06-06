from . import visualize_classes

methods = {
    "svs": visualize_classes._SingularValuesPlot,
    "psvs": visualize_classes._PermutedSingularValuesPlot,
    "dsc": visualize_classes._DesignLVPlot,
    "dsc": visualize_classes._DesignScoresPlot,
    "vir": visualize_classes._VoxelIntensityPlot,
    "blv": visualize_classes._BrainLVPlot,
    "belv": visualize_classes._BehavLVPlot,
    "bscvbe": visualize_classes._BrainScorevsBehavPlot,
    # "bsc": visualize_classes._Brain,
    "tbsc": visualize_classes._TaskPLSBrainScorePlot,
}


def visualize(*args, **kwargs):

    try:
        plot = kwargs.pop("plot")

    except KeyError:
        # default to singular values
        print("Unrecognized plot type. Defaulting to Singular Values.")
        plot = "svs"
    lv = kwargs.pop("lv", 1)  # Default `lv` to 1 if not provided
    kwargs["lv"] = lv
    # return finished PLS viz class with user-specified method
    return visualize_classes._SBPlotBase._create(plot, *args, **kwargs)
