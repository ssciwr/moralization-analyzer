import moralization
import matplotlib.pyplot
import seaborn
import ipywidgets
import IPython


def percent_matrix_heatmap(ax, data_dict, filename, categories):
    """Plot a heatmap of the overlap percent matrix for given data_dict, filename and categories"""
    df = moralization.analyse.get_percent_matrix(
        data_dict,
        filename,
        categories,
    )
    return seaborn.heatmap(df, cmap="cividis", ax=ax)


class InteractiveCategoryPlot:
    """Interactive plotting class for use in Jupyter notebooks

    User selects the filename and categories to plot using GUI widgets.
    The displayed plot is then automatically updated.
    A custom plotting callback can be provided to customize the plot.

    Attributes:
        data_dict: The data_dict to plot.
        plot_callback: The plotting function to call. Default is `percent_matrix_heatmap`.
        figsize: The figsize tuple to pass to matplotlib
    """

    def __init__(self, data_dict, plot_callback=None, figsize=None):
        if plot_callback is None:
            self.plot_callback = moralization.plot.percent_matrix_heatmap
        else:
            self.plot_callback = plot_callback
        self._output = ipywidgets.Output()
        self.data_dict = data_dict
        self.figsize = figsize
        # get all possible categories for each filename
        self._categories = {}
        filenames = list(data_dict.keys())
        for filename in filenames:
            self._categories[filename] = [
                key
                for span_dict_sub_kat in data_dict[filename]["data"].values()
                for key in span_dict_sub_kat.keys()
            ]
        # filename widget
        self._filename_widget = ipywidgets.Dropdown(
            options=filenames, value=filenames[0]
        )
        self._filename_widget.observe(self._filename_changed, names="value")
        # categories widget
        new_categories = self._categories[self._filename_widget.value]
        self._category_widget = ipywidgets.SelectMultiple(
            options=new_categories,
            rows=len(new_categories) + 1,
            description="",
            disabled=False,
        )
        self._display_container = ipywidgets.HBox(
            [
                ipywidgets.VBox([self._filename_widget, self._category_widget]),
                self._output,
            ]
        )
        self._category_widget.observe(self._categories_changed, names="value")
        self._category_widget.value = new_categories

    def _categories_changed(self, change):
        with self._output:
            if change["new"]:
                IPython.display.clear_output(wait=True)
                fig, ax = matplotlib.pyplot.subplots(figsize=self.figsize)
                self.plot_callback(
                    ax, self.data_dict, self._filename_widget.value, change["new"]
                )
                matplotlib.pyplot.show()

    def _filename_changed(self, change):
        new_categories = list(self._categories[change["new"]])
        self._category_widget.options = new_categories
        self._category_widget.value = new_categories

    def show(self):
        """Display the interactive plot"""
        IPython.display.display(self._display_container)
