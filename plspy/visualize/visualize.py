from . import visualize_classes

methods = {
    "svs": visualize_classes._SingularValuesPlot,
    "psvs": visualize_classes._PermutedSingularValuesPlot,
    "dsc": visualize_classes._DesignScoresPlot,
    "vir": visualize_classes._VoxelIntensityPlot,
    "blv": visualize_classes._BrainLVPlot,
    # "bsc": visualize_classes._Brain,
}


def visualize(*args, **kwargs):

    try:
        plot = kwargs.pop("plot")

    except KeyError:
        # default to singular values
        print("Unrecognized plot type. Defaulting to Singular Values.")
        plot = "svs"

    # return finished PLS viz class with user-specified method
    return visualize_classes._SBPlotBase._create(plot, *args, **kwargs)
