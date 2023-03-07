"""
Contains plotting functionality.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from moralization import analyse as ae
import ipywidgets
import IPython


class PlotSpans:
    """ """

    @staticmethod
    def _get_filter_multiindex(df_paragraph_occurrence: pd.DataFrame, filters):
        """Search through the given filters and return all sub_cat_keys
        when a main_cat_key is given.

        Args:
          df(pd.Dataframe): The sentence occurrence dataframe.
          filters(str): Filter values for the dataframe.
          df_paragraph_occurrence: pd.DataFrame:

        Returns:
          list: the filter strings of only the sub_cat_keys

        """
        if not isinstance(filters, list):
            filters = [filters]
        sub_cat_filter = []
        for _filter in filters:
            if _filter in ae.map_expressions:
                _filter = ae.map_expressions[_filter]

            if _filter in df_paragraph_occurrence.columns.levels[0]:
                [
                    sub_cat_filter.append(key)
                    for key in (df_paragraph_occurrence[_filter].keys())
                ]
            elif _filter in df_paragraph_occurrence.columns.levels[1]:
                sub_cat_filter.append(_filter)
            else:
                raise Warning(f"Filter key: { _filter} not in dataframe columns.")

        return sub_cat_filter

    @staticmethod
    def _generate_corr_df(
        df_paragraph_occurrence: pd.DataFrame, _filter=None
    ) -> pd.DataFrame:
        """

        Args:
          df_paragraph_occurrence: pd.DataFrame:
          _filter:  (Default value = None)

        Returns:

        """
        if _filter is None:
            return df_paragraph_occurrence.corr().sort_index(level=0)
        else:
            _filter = PlotSpans._get_filter_multiindex(df_paragraph_occurrence, _filter)
            # Couldn't figure out how to easily select columns based on the
            # second level column name.
            # So the df is transposed, the multiindex can be filterd using
            # loc, and then transposed back to get the correct correlation matrix.
            return (
                df_paragraph_occurrence.T.loc[(slice(None), _filter), :]
                .sort_index(level=0)
                .T.corr()
            )

    @staticmethod
    def report_occurrence_heatmap(df_paragraph_occurrence: pd.DataFrame, _filter=None):
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

        plt.figure(figsize=(16, 16))
        df_corr = PlotSpans._generate_corr_df(df_paragraph_occurrence, _filter=_filter)

        heatmap = sns.heatmap(df_corr, cmap="cividis")
        return heatmap

    @staticmethod
    def report_occurrence_matrix(
        df_paragraph_occurrence: pd.DataFrame, _filter=None
    ) -> pd.DataFrame:
        """Returns the correlation matrix in regards to the given filters.

        Args:
          df_paragraph_occurrence(pd.DataFrame): The occurence of labels per paragraph.
          _filter(str, optional): Filter values for the dataframe. (Default value = None)
          df_paragraph_occurrence: pd.DataFrame:

        Returns:
          pd.DataFrame: Correlation matrix.

        """
        return PlotSpans._generate_corr_df(df_paragraph_occurrence, _filter)


class InteractiveCategoryPlot:
    """Interactive plotting class for use in Jupyter notebooks

    User selects the filename and categories to plot using GUI widgets.
    The displayed plot is then automatically updated.
    A custom plotting callback can be provided to customize the plot.


    """

    def __init__(self, data_dict, plot_callback=None, figsize=None):
        """
        Args:
            data_dict (_type_): _description_
            plot_callback (_type_, optional): _description_. Defaults to None.
            figsize (_type_, optional): _description_. Defaults to None.
        """
        if plot_callback is None:
            self.plot_callback = PlotSpans.report_occurrence_heatmap
        else:
            self.plot_callback = plot_callback
        self._output = ipywidgets.Output()
        self.data_dict = data_dict
        self.df = ae.AnalyseSpans.report_occurrence_per_paragraph(self.data_dict)
        self.figsize = figsize
        # get all possible categories for each filename
        self._categories = {}
        filenames = list(data_dict.keys())
        for filename in filenames:
            self._categories[filename] = [
                f"{main_kat_key}: {sub_kat_key}"
                for main_kat_key, span_dict_sub_kat in data_dict[filename][
                    "data"
                ].items()
                for sub_kat_key in span_dict_sub_kat.keys()
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
                self.plot_callback(self.df)

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
