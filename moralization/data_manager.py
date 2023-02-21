from moralization import InputOutput
from moralization.analyse import _loop_over_files, _return_span_analyzer
from moralization.plot import report_occurrence_heatmap, InteractiveCategoryPlot
import pandas as pd


class DataManager:
    def __init__(self, data_dir):
        self.doc_dict = InputOutput.read_data(data_dir)
        self.analyzer = None

    def occurence_analysis(self, _type="table", cat_filter=None, file_filter=None):
        """Returns the occurence df, occurence_corr_table or heatmap of the dataset.
            optionally one can filter by filename(s).


        Args:
            _type (str, optional): Either "table", "corr" or "heatmap", defaults to table.
            filter (str/list(str), optional): Filename filters. Defaults to None.

        Returns:
            pd.DataFrame: occurence dataframe per paragraph.
        """

        if _type not in ["table", "corr", "heatmap"]:
            raise ValueError(
                f"_type argument can only be `table`, `corr` or `heatmap` but is {_type}"
            )

        self.occurence_df = _loop_over_files(self.doc_dict, file_filter=file_filter)
        if _type == "table":
            return self.occurence_df
        else:
            return report_occurrence_heatmap(
                self.occurence_df, _type=_type, _filter=cat_filter
            )

    def interactive_analysis(self):
        self.occurence_df = _loop_over_files(self.doc_dict)

        heatmap = InteractiveCategoryPlot(self.occurence_df, list(self.doc_dict.keys()))
        return heatmap

    def return_analyzer_result(self, result_type="frequency"):
        """Returns the result of the spacy_span-analyzer.


        Args:
            result_type (str, optional): Can be `frequency`, `length`,
              `span_distinctiveness` or `boundary_distinctiveness`. Defaults to "frequency".
        """

        if self.analyzer is None:
            self.analyzer = _return_span_analyzer(self.doc_dict)

        return_dict = {
            "frequency": self.analyzer.frequency,
            "length": self.analyzer.length,
            "span_distinctiveness": self.analyzer.span_distinctiveness,
            "boundary_distinctiveness": self.analyzer.boundary_distinctiveness,
        }
        return pd.DataFrame(return_dict[result_type]).fillna(0)
