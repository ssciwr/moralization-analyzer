"""
Contains plotting functionality.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from moralization import analyse as ae


class PlotSpans:
    @staticmethod
    def _get_filter_multiindex(df_paragraph_occurrence: pd.DataFrame, filters):
        """Search through the given filters and return all sub_cat_keys
        when a main_cat_key is given.

        Args:
            df (pd.Dataframe): The sentence occurrence dataframe.
            filters (str, list(str)): Filter values for the dataframe.

        Raises:
            Warning: Filter not in dataframe columns
        Returns:
            list: the filter strings of only the sub_cat_keys
        """
        if not isinstance(filters, list):
            filters = [filters]
        sub_cat_filter = []
        for filter_ in filters:
            if filter_ in ae.map_expressions:
                filter_ = ae.map_expressions[filter_]

            if filter_ in df_paragraph_occurrence.columns.levels[0]:
                [
                    sub_cat_filter.append(key)
                    for key in (df_paragraph_occurrence[filter_].keys())
                ]
            elif filter_ in df_paragraph_occurrence.columns.levels[1]:
                sub_cat_filter.append(filter_)
            else:
                raise Warning(f"Filter key: {filter_} not in dataframe columns.")

        return sub_cat_filter

    @staticmethod
    def _generate_corr_df(
        df_paragraph_occurrence: pd.DataFrame, filter_=None
    ) -> pd.DataFrame:
        if filter_ is None:
            return df_paragraph_occurrence.corr().sort_index(level=0)
        else:
            filter_ = PlotSpans._get_filter_multiindex(df_paragraph_occurrence, filter_)
            # Couldn't figure out how to easily select columns based on the
            # second level column name.
            # So the df is transposed, the multiindex can be filterd using
            # loc, and then transposed back to get the correct correlation matrix.
            return (
                df_paragraph_occurrence.T.loc[(slice(None), filter_), :]
                .sort_index(level=0)
                .T.corr()
            )

    @staticmethod
    def report_occurrence_heatmap(df_paragraph_occurrence: pd.DataFrame, filter_=None):
        """Returns the occurrence heatmap for the given dataframe.
        Can also filter based on both main_cat and sub_cat keys.

        Args:
            df_sentence_occurrence (pd.DataFrame): The sentence occurrence dataframe.
            filter_ (str,list(str), optional): Filter values for the dataframe.
            Defaults to None.

        Returns:
            plt.figure : The heatmap figure.
        """

        plt.figure(figsize=(16, 16))
        df_corr = PlotSpans._generate_corr_df(df_paragraph_occurrence, filter_=filter_)

        heatmap = sns.heatmap(df_corr, cmap="cividis")
        return heatmap

    @staticmethod
    def report_occurrence_matrix(
        df_paragraph_occurrence: pd.DataFrame, filter_=None
    ) -> pd.DataFrame:
        """
        Returns the correlation matrix in regards to the given filters.
        Args:
            filter_ (str,list(str), optional): Filter values for the dataframe.
            Defaults to None.
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        return PlotSpans._generate_corr_df(df_paragraph_occurrence, filter_)
