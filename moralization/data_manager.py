from moralization.input_data import InputOutput
from moralization.analyse import _loop_over_files, _return_span_analyzer
from moralization.plot import (
    report_occurrence_heatmap,
    InteractiveCategoryPlot,
    visualize_data,
)
from moralization.spacy_model import SpacyDataHandler, SpacyTraining
import pandas as pd
import spacy
from pathlib import Path


class DataManager:
    def __init__(self, data_dir):
        doc_dicts = InputOutput.read_data(data_dir)
        self.doc_dict, self.train_dict, self.test_dict = doc_dicts

        self.analyzer = None
        self.spacy_docbin_files = None
        self.spacy_model = None

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

    def export_data_DocBin(self, output_dir=None):
        # TODO add check if files already exist, maybe add custom filenames
        """Export the currently loaded docs as a spacy binary. This is used in spacy training.

        Args:
            output_dir (str/Path, optional): The directory in which to place the output files. Defaults to None.

        Returns:
            list[Path]: A list of the train and test files path.
        """
        self.spacy_docbin_files = SpacyDataHandler().export_training_testing_data(
            self.train_dict, self.test_dict, output_dir
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

    def spacy_train(
        self, working_dir=None, config=None, n_epochs=None, use_gpu=-1, overwrite=None
    ):
        if self.spacy_docbin_files is None:
            raise FileNotFoundError(
                "No spacy docbin files are loaded, please first run either `export_data_DocBin` or"
                + " `import_data_DocBin`."
            )
        if overwrite is None:
            overwrite = {}
        spacy_training = SpacyTraining(
            working_dir,
            training_file=self.spacy_docbin_files[0],
            testing_file=self.spacy_docbin_files[1],
            config_file=config,
        )
        if n_epochs is not None:
            overwrite["training.max_epochs"] = n_epochs

        spacy_model_path = spacy_training.train(use_gpu=use_gpu, overwrite=overwrite)
        self.spacy_model_path = Path(spacy_model_path)
        self.spacy_model = spacy.load(self.spacy_model_path)

    def spacy_import_model(self, model_path):
        self.spacy_model_path = Path(model_path)

        self.spacy_model = spacy.load(self.spacy_model_path)

    def spacy_validation(self, validation_file=None, working_dir=None):
        if self.spacy_model is None:
            raise ValueError(
                "No spacy model is loaded, please run `spacy_train` or `import_spacy_model` first."
            )

        if working_dir is None:
            output_file = Path(self.spacy_model_path).parents[0] / "evaluation.json"
        else:
            output_file = Path(working_dir) / "evaluation.json"

        if validation_file is None:
            if self.spacy_docbin_files is None:
                raise FileNotFoundError(
                    "No validation file was provided and none was found in 'DataManager.spacy_docbin_files'. "
                    + "Either provide a validation file as a function argument or run either "
                    + "`DataManager.export_data_DocBin` or `DataManager.import_data_DocBin`."
                )

            validation_file = self.spacy_docbin_files[1]

        return SpacyTraining.evaluate(
            output_file=output_file,
            validation_file=validation_file,
            model=self.spacy_model_path,
        )

    def spacy_test_string(self, test_string, style="span"):

        if self.spacy_model is None:
            raise ValueError(
                "No spacy model is loaded, please run `spacy_train` or `import_spacy_model` first."
            )

        SpacyTraining.test_model_with_string(self.spacy_model, test_string, style)
