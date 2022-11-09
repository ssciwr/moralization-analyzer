import moralization as mn
import moralization.plot as mnplot
import pytest


def test_InteractiveCategoryPlot(data_dict):
    # removes automatic window opening by matplotlib
    import matplotlib

    matplotlib.use("Agg")

    heatmap = mnplot.InteractiveCategoryPlot(data_dict, figsize=(15, 10))
    heatmap.show()

    heatmap = mnplot.InteractiveCategoryPlot(data_dict)
    heatmap.show()

    heatmap = mnplot.InteractiveCategoryPlot(
        data_dict, plot_callback=mnplot.percent_matrix_heatmap
    )
    heatmap.show()
