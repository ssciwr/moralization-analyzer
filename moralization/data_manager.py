from moralization.input_data import InputOutput
from moralization.analyse import _loop_over_files, _return_span_analyzer
from moralization.plot import (
    report_occurrence_heatmap,
    InteractiveCategoryPlot,
    return_displacy_visualization,
    InteractiveAnalyzerResults,
    InteractiveVisualization,
)
from collections import defaultdict

import logging
import os
from moralization.spacy_data_handler import SpacyDataHandler
from moralization.transformers_data_handler import TransformersDataHandler
import pandas as pd
import datasets
import numpy as np
from typing import Dict, Optional, Union
import huggingface_hub


class DataManager:
    def __init__(
        self,
        data_dir: str,
        language_model: str = "de_core_news_sm",
        skip_read: bool = False,
        selected_labels: Union[list, str] = None,
        merge_dict=None,
        task: str = "task1",
    ):
        """Initialize the DataManager that handles the data transformations.

        Args:
            data_dir (str): The data directory where the data is located, or where the pulled dataset
                should be stored.
            language_model (str, optional): Language model for sentencizing the corpus that is being read.
                Defaults to "de_core_news_sm" (small German).
            skip_read (bool, optional): If this is set to True, no data reading will be attempted. Use
                this if pulling a dataset from Hugging Face. Defaults to False.
            selected_labels (Union[str, list]): The labels used in the training. Either "all", which will
                return all labels of a given task, or a list of selected labels, such as ["Cheating", "Fairness"].
                If you provide a list, this is independent of the task.
                Defaults to None, in which case all labels for all categories are selected.
            merge_dict_cat(dict, optional): map new category to list of existing_categories.
                Default is:
                merge_dict = {
                    "task1": ["KAT1-Moralisierendes Segment"],
                    "task2": ["KAT2-Moralwerte", "KAT2-Subjektive Ausdrücke"],
                    "task3": ["KAT3-Rolle", "KAT3-Gruppe", "KAT3-own/other"],
                    "task4": ["KAT4-Kommunikative Funktion"],
                    "task5": ["KAT5-Forderung explizit"],
                }
                Defaults to None.

            task (str): The task to train on. The options are
                "task1": ["KAT1-Moralisierendes Segment"]
                "task2": ["KAT2-Moralwerte", "KAT2-Subjektive Ausdrücke"]
                "task3": ["KAT3-Rolle", "KAT3-Gruppe", "KAT3-own/other"]
                "task4": ["KAT4-Kommunikative Funktion"]
                "task5": ["KAT5-Forderung explizit"]
                Defaults to "task1".

        Returns:
            A DataManager object.
        """
        # what are these? why set to None?
        self.data_dir = data_dir
        self.analyzer = None
        self.spacy_docbin_files = None
        # select the labels and task for the dataset
        if not selected_labels:
            # if no labels are selected per task, we just choose all
            selected_labels = "all"
        self.selected_labels = selected_labels
        self.task = task
        if not skip_read:
            self.doc_dict = InputOutput.read_data(
                self.data_dir,
                language_model=language_model,
                merge_dict=merge_dict,
                task=self.task,
            )
            # generate the data lists and data frame
            self._docdict_to_lists()
            self._lists_to_df()
            self.df_to_dataset()
            description = self._set_dataset_description()
            self.set_dataset_info(description=description)

    def occurrence_analysis(self, _type="table", cat_filter=None, file_filter=None):
        """Returns the occurrence df, occurrence_corr_table or heatmap of the dataset.
            optionally one can filter by filename(s).


        Args:
            _type (str, optional): Either "table", "corr" or "heatmap", defaults to table.
            filter (str/list(str), optional): Filename filters. Defaults to None.

        Returns:
            pd.DataFrame: occurrence dataframe per paragraph.
        """
        if not hasattr(self, "doc_dict"):
            raise ValueError(
                "The data analysis can only be carried out for xmi data, not datasets pulled from the Hugging Face Hub."
            )
        if _type not in ["table", "corr", "heatmap"]:
            raise ValueError(
                f"_type argument can only be `table`, `corr` or `heatmap` but is {_type}"
            )

        occurrence_df = _loop_over_files(self.doc_dict, file_filter=file_filter)
        if _type == "table":
            return occurrence_df
        else:
            return report_occurrence_heatmap(
                occurrence_df, _type=_type, _filter=cat_filter
            )

    def return_analyzer_result(self, result_type="frequency"):
        """Returns the result of the spacy_span-analyzer.
            If no analyzer has been created yet, a new one will be generated and stored.

        Args:
            result_type (str, optional): Can be `frequency`, `length`,
              `span_distinctiveness`, `boundary_distinctiveness` or "all". Defaults to "frequency".
        """
        if not hasattr(self, "doc_dict"):
            raise ValueError(
                "The data analysis can only be carried out for xmi data, not datasets pulled from the Hugging Face Hub."
            )
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

    def interactive_correlation_analysis(self, port=8051):
        heatmap = InteractiveCategoryPlot(self)
        return heatmap.run_app(port=port)

    def return_categories(self):
        """Returns a dict of all categories in the dataset.

        Returns:
            dict: A list of all categories in the dataset.
        """
        occurrence_df = _loop_over_files(self.doc_dict)
        multi_column = occurrence_df.columns

        category_dict = defaultdict(list)
        for column in multi_column:
            category_dict[column[0]].append(column[1])

        return category_dict

    def interactive_data_analysis(self, port=8053) -> InteractiveAnalyzerResults:
        all_analysis = self.return_analyzer_result("all")

        categories_dict = self.return_categories()
        interactive_analysis = InteractiveAnalyzerResults(all_analysis, categories_dict)
        return interactive_analysis.run_app(port=port)

    def interactive_data_visualization(
        self,
        port=8052,
    ):
        interactive_visualization = InteractiveVisualization(self)
        return interactive_visualization.run_app(port=port)

    def visualize_data(self, spans_key="sc"):
        if not hasattr(self, "doc_dict"):
            raise ValueError(
                "The data analysis can only be carried out for xmi data, not datasets pulled from the Hugging Face Hub."
            )
        return return_displacy_visualization(self.doc_dict, spans_key=spans_key)

    def export_data_DocBin(
        self, output_dir=None, overwrite=False, check_data_integrity=True
    ):
        """Export the currently loaded dataset as a spacy binary. This is used in spacy training.

        Args:
            output_dir (str/Path, optional): The directory in which to place the output files. Defaults to None.
            overwrite(bool, optional): If True, spacy files are written even if files are already present.
                Defaults to False.
            check_data_integrity (bool): Whether or not to test the data integrity. If the data integrity
                check fails, then no output is written. Skip the test by setting to False. In this case,
                the output is always generated even if the data does not pass the quality check.

        Returns:
            list[Path]: A list of the train and test files path.
        """
        if check_data_integrity:
            data_integrity = self.check_data_integrity()
            if not data_integrity:
                raise ValueError(
                    "The given data did not pass the integrity check. Please check the provided output.\n"
                    + "if you want to continue with your data set `check_data_integrity=False`"
                )
        # generate the DocBin files from the train and test split of the dataset object,
        # and optionally from validate if present
        train_path = SpacyDataHandler.docbin_from_dataset(
            self.train_test_set,
            self.task,
            "train",
            output_dir,
            overwrite=overwrite,
            column_names=self.column_names,
        )
        test_path = SpacyDataHandler.docbin_from_dataset(
            self.train_test_set,
            self.task,
            "test",
            output_dir,
            overwrite=overwrite,
            column_names=self.column_names,
        )
        self.spacy_docbin_files = [train_path, test_path]

    def _check_relativ_frequency(
        self,
        threshold,
    ):
        analyzer_df = self.return_analyzer_result("frequency")
        warning_str = ""
        frequency_integrity = True

        for column in analyzer_df.columns:
            warning_str += "----------------\n"
            warning_str += f"Checking if any labels are disproportionately rare in span_cat '{column}':\n"

            max_occurrence = analyzer_df[analyzer_df > 0][column].max()
            max_occurrence_label = str(
                analyzer_df.loc[analyzer_df[column] == max_occurrence][column].index
            )

            under_threshold_df = analyzer_df[column][analyzer_df[column] > 0][
                analyzer_df[column] < max_occurrence * threshold
            ].dropna()

            under_threshold_df = under_threshold_df / max_occurrence
            under_threshold_dict = under_threshold_df.to_dict()
            if under_threshold_dict:
                warning_str += (
                    f"Compared to the maximal occurrence of {max_occurrence} in "
                    + f"{max_occurrence_label}. \n"
                )

                for key, value in under_threshold_dict.items():
                    warning_str += f"\t {key} : {round(value,3)} \n"
                logging.warning(warning_str)
                under_threshold_dict = None

                frequency_integrity = False
            else:
                warning_str += "\t No problem found.\n"
                logging.warning(warning_str)
        return warning_str, frequency_integrity

    def check_data_integrity(self):
        """This function checks the data and compares it to the spacy thresholds for label count,
        span distinctiveness and boundary distinctiveness.

        If a value is found to be insufficient a warning will be raised.

        By default this function will be called when training data is exported
        """

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

        data_integrity = True

        for threshold, analyzer_result_label in zip(thresholds, analyzer_result_labels):
            logging.info(f"Check analyzer category {analyzer_result_label}:")
            warning_str = (
                f"\nThe following span categories have a {analyzer_result_label}"
                + f" of less then {threshold}. \n"
            )

            if analyzer_result_label == "relativ_frequency":
                # for this we need to iterate over each span cat induvidually.
                _warning_str, frequency_integrity = self._check_relativ_frequency(
                    threshold=RELATIV_THRESHOLD
                )
                warning_str += _warning_str
                # set data_integrity to false if any of the checks fail.
                print(frequency_integrity)
                if frequency_integrity is False:
                    data_integrity = False

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

                    # set data_integrity to false if any of the checks fail.
                    data_integrity = False
        return data_integrity

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
        self.spacy_docbin_files = SpacyDataHandler.import_training_testing_data(
            input_dir, train_file, test_file
        )
        return self.spacy_docbin_files

    def _docdict_to_lists(self):
        """Convert the dictionary of doc objects to nested lists."""

        tdh = TransformersDataHandler()
        tdh.get_data_lists(self.doc_dict)
        tdh.generate_labels(self.doc_dict, self.selected_labels, self.task)
        tdh.generate_spans(self.doc_dict, self.selected_labels, self.task)
        (
            self.sentence_list,
            self.label_list,
            self.span_begin,
            self.span_end,
            self.span_label,
        ) = tdh.structure_labels()

    def _lists_to_df(self):
        """Convert nested lists of tokens and labels into a pandas dataframe.

        Returns:
            data_in_frame (dataframe): A list of the train and test files path.
        """
        self.data_in_frame = pd.DataFrame(
            zip(
                self.sentence_list,
                self.label_list,
                self.span_begin,
                self.span_end,
                self.span_label,
            ),
            columns=["Sentences", "Labels", "Span_begin", "Span_end", "Span_label"],
        )
        self.column_names = [
            "Sentences",
            "Labels",
            "Span_begin",
            "Span_end",
            "Span_label",
        ]

    def df_to_dataset(self, data_in_frame: pd.DataFrame = None, split: bool = True):
        if not data_in_frame:
            data_in_frame = self.data_in_frame
        self.raw_data_set = datasets.Dataset.from_pandas(data_in_frame)
        if split:
            # split in train test
            self.train_test_set = self.raw_data_set.train_test_split(test_size=0.1)

    def publish(
        self,
        repo_id: str,
        data_set: datasets.Dataset = None,
        hugging_face_token: Optional[str] = None,
    ) -> Dict[str, str]:
        """Publish the dataset to Hugging Face.

        This requires a User Access Token from https://huggingface.co/

        The token can either be passed via the `hugging_face_token` argument,
        or it can be set via the `HUGGING_FACE_TOKEN` environment variable.
        If the token is not set, a prompt will pop up where it can be provided.

        Args:
            repo_id (str): The name of the repository that you are pushing to.
            This can either be a new repository or an existing one.
            data_set (Dataset, optional): The Dataset to be published to Hugging Face. Please
            note that this is a Dataset object and not a DatasetDict object, meaning
            that if you have already split your dataset into test and train, you can
            either push test and train separately or need to concatenate them using "+".
            If not set, the raw dataset that is connected to the DataManager instance will
            be used.
            hugging_face_token (str, optional): Hugging Face User Access Token.
        """
        if not data_set:
            data_set = self.raw_data_set
        self.print_dataset_info(data_set)
        if hugging_face_token is None:
            hugging_face_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hugging_face_token is None:
            print("Obtaining token directly from user..")
        huggingface_hub.login(token=hugging_face_token)
        data_set.push_to_hub(repo_id=repo_id)
        print(
            "If you have not yet set up a README (dataset card) for your dataset, please do so on Hugging Face Hub!"
        )

    def print_dataset_info(self, data_set: datasets.Dataset = None) -> None:
        """Print information set in the dataset.

        Args:
            data_set (Dataset, optional): The Dataset object of which the information
            is to be printed. Defaults to the raw dataset associated with the DataManager
            instance.
        """
        if not data_set:
            data_set = self.raw_data_set
        if not hasattr(self, "data_set_info"):
            # check for Dataset or DatasetDict
            if isinstance(data_set, datasets.Dataset):
                self.data_set_info = data_set.info
            elif isinstance(data_set, datasets.DatasetDict):
                # the datasetdict should at the very least contain the training data
                self.data_set_info = data_set["train"].info
        print("The following dataset metadata has been set:")
        print("Description:", self.data_set_info.description)
        print("Version:", self.data_set_info.version)
        print("License:", self.data_set_info.license)
        print("Citation:", self.data_set_info.citation)
        print("homepage:", self.data_set_info.homepage)

    def _set_dataset_description(self):
        description = "The dataset was generated for labels: {} and task: {} .".format(
            self.selected_labels, self.task
        )
        description += "It contains the data from the original files {}.".format(
            list(self.doc_dict.keys())
        )
        return description

    def set_dataset_info(
        self,
        data_set: datasets.Dataset = None,
        description: str = None,
        version: str = None,
        license_: str = None,
        citation: str = None,
        homepage: str = None,
    ) -> datasets.Dataset:
        """Update the information set in the dataset.

        Args:
            data_set (Dataset, optional): The Dataset object of which the information is to be updated.
            Defaults to the raw dataset associated with the DataManager instance.
            description (str, optional): The new description to be updated. Optional, defaults to None.
                The description will contain the task for which the labels were created, and the
                names of the original data files.
            version (str, optional): The new version to be updated. Optional, defaults to None.
            license (str, optional): The new license to be updated. Optional, defaults to None.
            citation (str, optional): The new citation to be updated. Optional, defaults to None.
            homepage (str, optional): The new homepage to be updated. Optional, defaults to None.
        Returns:
            Dataset: The updated Dataset object.
        """
        if not data_set:
            data_set = self.raw_data_set
        if not hasattr(self, "data_set_info"):
            # check for Dataset or DatasetDict
            if isinstance(data_set, datasets.Dataset):
                self.data_set_info = data_set.info
            elif isinstance(data_set, datasets.DatasetDict):
                # the datasetdict should at the very least contain the training data
                self.data_set_info = data_set["train"].info
        print("Updating the following dataset metadata:")
        if description:
            print(
                "Description: old - {} new - {}".format(
                    self.data_set_info.description, description
                )
            )
            self.data_set_info.description = description
        if version:
            print(
                "Version: old - {} new - {}".format(self.data_set_info.version, version)
            )
            self.data_set_info.version = version
        if license_:
            print(
                "License: old - {} new - {}".format(
                    self.data_set_info.license, license_
                )
            )
            self.data_set_info.license = license_
        if citation:
            print(
                "Citation: old - {} new - {}".format(
                    self.data_set_info.citation, citation
                )
            )
            self.data_set_info.citation = citation
        if homepage:
            print(
                "homepage: old - {} new - {}".format(
                    self.data_set_info.homepage, homepage
                )
            )
            self.data_set_info.homepage = homepage
        return data_set

    def pull_dataset(self, dataset_name: str, revision: str = None, split: str = None):
        """Method to pull existing dataset from Hugging Face.

        Args:
            dataset_name (str): Name of the dataset to pull.
            revision (str, optional): The revision number of the dataset
                that should be pulled. If not set, the default version from
                the "main" branch will be pulled.
            split (str, optional): Select a specific split of the dataset to pull.
                Depending on the dataset, this can be "train", "test", "validation",
                to be split into new test and train sets after the pull.
                Can also be set to None , pulling the full dataset with existing splits.
                Defaults to None."""
        # this should check if dataset is already downloaded
        self.raw_data_set = datasets.load_dataset(
            path=dataset_name,
            split=split,
            revision=revision,
            cache_dir=self.data_dir,
        )
        if isinstance(self.raw_data_set, datasets.Dataset):
            print(
                "Your dataset is in Dataset format - will now be split into test and train"
            )
            self.data_in_frame = pd.DataFrame(self.raw_data_set)
            self.column_names = self.raw_data_set.column_names
            self.train_test_set = self.raw_data_set.train_test_split(test_size=0.1)
        if isinstance(self.raw_data_set, datasets.DatasetDict):
            print("Your dataset is in DatasetDict format - will keep the split")
            # check if the split contains train
            if "train" in self.raw_data_set:
                print("Found train split - ")
                self.data_in_frame = self.raw_data_set["train"].to_pandas()
                self.column_names = self.raw_data_set.column_names["train"]
            if "test" in self.raw_data_set:
                print("Found test split - ")
                self.data_in_frame = pd.concat(
                    [self.data_in_frame, self.raw_data_set["test"].to_pandas()]
                )
            if "validation" in self.raw_data_set:
                print("Found validation split - ")
                self.data_in_frame = pd.concat(
                    [self.data_in_frame, self.raw_data_set["validation"].to_pandas()]
                )
