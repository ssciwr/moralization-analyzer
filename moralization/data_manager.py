from moralization.input_data import InputOutput
from moralization.analyse import _loop_over_files, _return_span_analyzer
from moralization.plot import (
    report_occurrence_heatmap,
    InteractiveCategoryPlot,
    visualize_data,
)
from moralization.spacy_data_handler import SpacyDataHandler
import pandas as pd
import datasets


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

        if result_type not in list(return_dict.keys()):
            raise KeyError(
                f"result_type '{result_type}' not in '{list(return_dict.keys())}'."
            )

        return pd.DataFrame(return_dict[result_type]).fillna(0)

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

    def lists_to_df(self, sentence_list, label_list):
        """Convert nested lists of tokens and labels into a pandas dataframe.

        Args:
            sentence_list (list): A nested list of the tokens (nested by sentence).
            label_list (list): A nested list of the labels (nested by sentence).

        Returns:
            data_in_frame (dataframe): A list of the train and test files path.
        """
        self.data_in_frame = pd.DataFrame(
            zip(sentence_list, label_list), columns=["Sentences", "Labels"]
        )

    def df_to_dataset(self, split=True):
        self.raw_data_set = datasets.Dataset.from_pandas(self.data_in_frame)
        if split:
            # split in train test
            self.train_test_set = self.raw_data_set.train_test_split(test_size=0.1)

    # here we also need a method to publish the dataset to hugging face
