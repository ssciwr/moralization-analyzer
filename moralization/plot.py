"""
Contains plotting functionality.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets
import IPython


def report_occurrence_heatmap(
    occurence_df: pd.DataFrame, _filter=None, _type="heatmap"
):
    """Returns the occurrence heatmap for the given dataframe.
    Can also filter based on both main_cat and sub_cat keys.

    Args:
        df_sentence_occurrence(pd.DataFrame): The sentence occurrence dataframe.
        _filter(str, optional): Filter values for the dataframe. (Default value = None)
        _filter(str, optional): Filter values for the dataframe.
    Defaults to None.
        df_paragraph_occurrence: pd.DataFrame:

    Returns:
        plt.figure: The heatmap figure.

    """

    if _type not in ["corr", "heatmap"]:
        raise ValueError(
            f"_type argument can only be `corr` or `heatmap` but is {_type}"
        )

    df_corr = _generate_corr_df(occurence_df, _filter=_filter)

    if _type == "corr":
        return df_corr
    elif _type == "heatmap":
        plt.figure(figsize=(16, 16))
        heatmap = sns.heatmap(df_corr, cmap="cividis")
        return heatmap


def _get_filter_multiindex(occurence_df: pd.DataFrame, filters):
    """Search through the given filters and return all sub_cat_keys
    when a main_cat_key is given.

    Args:
        df(pd.Dataframe): The sentence occurrence dataframe.
        filters(str): Filter values for the dataframe.
        occurence_df: pd.DataFrame:

    Returns:
        list: the filter strings of only the sub_cat_keys

    """
    if not isinstance(filters, list):
        filters = [filters]
    sub_cat_filter = []
    for _filter in filters:

        if _filter in occurence_df.columns.levels[0]:
            [sub_cat_filter.append(key) for key in (occurence_df[_filter].keys())]
        elif _filter in occurence_df.columns.levels[1]:
            sub_cat_filter.append(_filter)
        else:
            raise KeyError(f"Filter key: `{ _filter}` not in dataframe columns.")

    return sub_cat_filter


def _generate_corr_df(occurence_df: pd.DataFrame, _filter=None) -> pd.DataFrame:
    """

    Args:
      df_paragraph_occurrence: pd.DataFrame:
      _filter:  (Default value = None)

    Returns:

    """
    if _filter is None:
        return occurence_df.sort_index(level=0).corr()
    else:
        _filter = _get_filter_multiindex(occurence_df, _filter)
        # Couldn't figure out how to easily select columns based on the
        # second level column name.
        # So the df is transposed, the multiindex can be filterd using
        # loc, and then transposed back to get the correct correlation matrix.
        return (
            occurence_df.T.loc[(slice(None), _filter), :].sort_index(level=0).T.corr()
        )


class InteractiveCategoryPlot:
    """Interactive plotting class for use in Jupyter notebooks

    User selects the filename and categories to plot using GUI widgets.
    The displayed plot is then automatically updated.
    A custom plotting callback can be provided to customize the plot.


    """

    def __init__(self, occurence_df, filenames, plot_callback=None, figsize=None):
        """
        Args:
            data_dict (_type_): _description_
            plot_callback (_type_, optional): _description_. Defaults to None.
            figsize (_type_, optional): _description_. Defaults to None.
        """
        if plot_callback is None:
            self.plot_callback = report_occurrence_heatmap
        else:
            self.plot_callback = plot_callback
        self._output = ipywidgets.Output()
        self.df = occurence_df
        self.figsize = figsize
        # get all possible categories for each filename
        self._categories = {}
        for filename in filenames:
            self._categories[filename] = [
                f"{main_kat_key}:: {sub_kat_key}"
                for main_kat_key, sub_kat_key in self.df.columns
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
        """

        Args:
          change:

        Returns:

        """
        with self._output:
            if change["new"]:
                IPython.display.clear_output(wait=True)
                filters = [category.split(":: ")[1] for category in change["new"]]
                self.plot_callback(self.df, _filter=filters)

    def _filename_changed(self, change):
        """

        Args:
          change:

        Returns:

        """
        new_categories = list(self._categories[change["new"]])
        self._category_widget.options = new_categories
        self._category_widget.value = new_categories

    def show(self):
        """Display the interactive plot"""
        IPython.display.display(self._display_container)
