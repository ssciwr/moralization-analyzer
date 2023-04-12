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
import numpy as np


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

        # cache return dict as well as the analyzer object
        if self.analyzer is None:
            self.analyzer = _return_span_analyzer(self.doc_dict)
            dict_entries = [
                "frequency",
                "length",
                "span_distinctiveness",
                "boundary_distinctiveness",
            ]
            return_dict = {}
            # this allows us to catch numpy runtime warning.
            with np.errstate(all="raise"):
                for entry in dict_entries:
                    try:
                        return_dict[entry] = pd.DataFrame(
                            self.analyzer.__getattribute__(entry)
                        ).fillna(0)

                    except FloatingPointError:
                        logging.warning(
                            f"Numpy FloatingPointError in {entry}!\n"
                            + "Most likely a category has to few entries to perform mean analysis on."
                            + " Check further output to find the culprit."
                        )
                        # after  raised warning continue without further warning.
                        with np.errstate(all="ignore"):
                            return_dict[entry] = pd.DataFrame(
                                self.analyzer.__getattribute__(entry)
                            ).fillna(0)

                    except RuntimeWarning as e:
                        logging.warning(
                            f"Numpy RuntimeWarning in {entry}!\n {e}\n"
                            + "However for unknown reasons this catch is currently not supported by numpy..."
                        )

            self.analyzer_return_dict = return_dict

        if (
            result_type not in list(self.analyzer_return_dict.keys())
            and result_type != "all"
        ):
            raise KeyError(
                f"result_type '{result_type}' not in '{list(self.analyzer_return_dict.keys())}'."
            )
        if result_type == "all":
            return self.analyzer_return_dict
        return pd.DataFrame(self.analyzer_return_dict[result_type]).fillna(0)

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

    def export_data_DocBin(
        self, output_dir=None, overwrite=False, check_data_integrity=True
    ):
        """Export the currently loaded docs as a spacy binary. This is used in spacy training.

        Args:
            output_dir (str/Path, optional): The directory in which to place the output files. Defaults to None.
            overwrite(bool, optional): whether or not the spacy files should be written
            even if files are already present.
            check_data_integrity (bool): Whether or not to test the data integrity.


        Returns:
            list[Path]: A list of the train and test files path.
        """
        if check_data_integrity:
            data_failed_check = self.check_data_integrity()
            if data_failed_check:
                raise ValueError(
                    "The given data did not pass the integrity check. Please check the provided output.\n"
                    + "if you want to continue with your data set `check_data_integrity=False`"
                )

        self.spacy_docbin_files = SpacyDataHandler().export_training_testing_data(
            self.train_dict, self.test_dict, output_dir, overwrite=overwrite
        )
        return self.spacy_docbin_files

    def check_data_integrity(self):
        """This function checks the data and compares it to the spacy thresholds for label count,
        span distinctiveness and boundary distinctiveness.

        If a value is found to be insufficient a warning will be raised.

        By default this function will be called when training data is exported

        """

        data_integrity_failed = False

        # thresholds:
        NEW_LABEL_THRESHOLD = 50
        SPAN_DISTINCT_THRESHOLD = 1
        BOUNDARY_DISTINCT_THRESHOLD = 1
        RELATIV_THRESHOLD = 0.2
        logging.info("Checking data integrity:")

        thresholds = [
            NEW_LABEL_THRESHOLD,
            RELATIV_THRESHOLD,
            SPAN_DISTINCT_THRESHOLD,
            BOUNDARY_DISTINCT_THRESHOLD,
        ]
        analyzer_result_labels = [
            "frequency",
            "relativ_frequency",
            "span_distinctiveness",
            "boundary_distinctiveness",
        ]

        for threshold, analyzer_result_label in zip(thresholds, analyzer_result_labels):
            logging.info(f"Check analyzer category {analyzer_result_label}:")
            warning_str = (
                f"\nThe following span categories have a {analyzer_result_label}"
                + f" of less then {threshold}. \n"
            )

            if analyzer_result_label == "relativ_frequency":
                # for this we need to iterate over each span cat induvidually.
                analyzer_df = self.return_analyzer_result("frequency")

                for column in analyzer_df.columns:
                    warning_str += "----------------\n"
                    warning_str += f"Checking if any labels are disproportionately rare in span_cat '{column}':\n"

                    max_occurence = analyzer_df[analyzer_df > 0][column].max()
                    max_occurence_label = str(
                        analyzer_df.loc[analyzer_df[column] == max_occurence][
                            column
                        ].index
                    )

                    under_threshold_df = analyzer_df[column][analyzer_df[column] > 0][
                        analyzer_df[column] < max_occurence * RELATIV_THRESHOLD
                    ].dropna()

                    under_threshold_df = under_threshold_df / max_occurence
                    under_threshold_dict = under_threshold_df.to_dict()
                    if under_threshold_dict:
                        warning_str += (
                            f"Compared to the maximal occurence of {max_occurence} in "
                            + f"{max_occurence_label}. \n"
                        )

                        for key, value in under_threshold_dict.items():
                            warning_str += f"\t {key} : {round(value,3)} \n"
                        logging.warning(warning_str)
                        under_threshold_dict = None

                        data_integrity_failed = True
                    else:
                        warning_str += "\t No problem found.\n"
                    # warning_str += "----------------\n"
            else:
                analyzer_df = self.return_analyzer_result(analyzer_result_label)

                under_threshold_dict = (
                    analyzer_df[analyzer_df < threshold][analyzer_df > 0]["sc"]
                    .dropna()
                    .to_dict()
                )

                for key, value in under_threshold_dict.items():
                    warning_str += f"\t {key} : {round(value,3)} \n"
                warning_str += (
                    "Be warned that this might result in poor quality data. \n"
                )
                if under_threshold_dict:
                    logging.warning(warning_str)
                    data_integrity_failed = True

        return data_integrity_failed

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
