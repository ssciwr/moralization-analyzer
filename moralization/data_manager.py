from moralization.input_data import InputOutput
from moralization.analyse import _loop_over_files, _return_span_analyzer
from moralization.plot import (
    report_occurrence_heatmap,
    InteractiveCategoryPlot,
    visualize_data,
    InteractiveAnalyzerResults,
)
import logging

from moralization.spacy_data_handler import SpacyDataHandler
import pandas as pd


class DataManager:
    def __init__(self, data_dir):
        doc_dicts = InputOutput.read_data(data_dir)
        self.doc_dict, self.train_dict, self.test_dict = doc_dicts

        self.analyzer = None
        self.spacy_docbin_files = None

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
        return heatmap.show()

    def return_analyzer_result(self, result_type="frequency"):
        """Returns the result of the spacy_span-analyzer.


        Args:
            result_type (str, optional): Can be `frequency`, `length`,
              `span_distinctiveness`, `boundary_distinctiveness` or "all". Defaults to "frequency".
        """

        if self.analyzer is None:
            self.analyzer = _return_span_analyzer(self.doc_dict)

        return_dict = {
            "frequency": pd.DataFrame(self.analyzer.frequency).fillna(0),
            "length": pd.DataFrame(self.analyzer.length).fillna(0),
            "span_distinctiveness": pd.DataFrame(
                self.analyzer.span_distinctiveness
            ).fillna(0),
            "boundary_distinctiveness": pd.DataFrame(
                self.analyzer.boundary_distinctiveness
            ).fillna(0),
        }

        if result_type not in list(return_dict.keys()) and result_type != "all":
            raise KeyError(
                f"result_type '{result_type}' not in '{list(return_dict.keys())}'."
            )
        if result_type == "all":
            return return_dict
        return pd.DataFrame(return_dict[result_type]).fillna(0)

    def interactive_data_analysis(self):
        all_analysis = self.return_analyzer_result("all")
        interactive_analysis = InteractiveAnalyzerResults(all_analysis)
        return interactive_analysis.show()

    def visualize_data(self, _type: str, spans_key="sc"):

        # type can only be all, train or test
        if _type not in ["all", "train", "test"]:
            raise KeyError(
                f"_type must be either 'all', 'train' or `test` but is `{_type}`."
            )

        return_dict = {
            "all": self.doc_dict,
            "train": self.train_dict,
            "test": self.test_dict,
        }

        return visualize_data(return_dict[_type], spans_key=spans_key)

    def export_data_DocBin(self, output_dir=None, overwrite=False):
        """Export the currently loaded docs as a spacy binary. This is used in spacy training.

        Args:
            output_dir (str/Path, optional): The directory in which to place the output files. Defaults to None.
            overwrite(bool, optional): whether or not the spacy files should be written
            even if files are already present.

        Returns:
            list[Path]: A list of the train and test files path.
        """
        self.spacy_docbin_files = SpacyDataHandler().export_training_testing_data(
            self.train_dict, self.test_dict, output_dir, overwrite=overwrite
        )
        return self.spacy_docbin_files

    def check_data_integrity(self):
        # thresholds:
        NEW_LABEL_THRESHOLD = 50

        logging.debug("Checking data integrity:")

        logging.debug("Check span cat occurences:")
        occurence_df = self.return_analyzer_result("occurence")
        under_threshold = (
            occurence_df[occurence_df < NEW_LABEL_THRESHOLD][occurence_df > 0]["sc"]
            .dropna()
            .to_dict()
        )
        if under_threshold:
            logging.warning(
                f"The following span categories have less then {NEW_LABEL_THRESHOLD} entries. "
                + "Be warned that this might result in poor quality data."
                + +f"{under_threshold}"
            )

    def import_data_DocBin(self, input_dir=None, train_file=None, test_file=None):
        """Load spacy files from a given directory, from absolute path,
            from relative path of given directory or from relative path of current working directory.

        Args:
            input_dir (Path, optional): Lookup directory. Defaults to None.
            train_file (Path, optional): Absolute or relative path. Defaults to None.
            test_file (Path, optional): Absolute or relative path. Defaults to None.

        Returns:
            list[Path]: A list of the train and test files path.
        """
        self.spacy_docbin_files = SpacyDataHandler().import_training_testing_data(
            input_dir, train_file, test_file
        )
        return self.spacy_docbin_files
