import spacy
from spacy.tokens import DocBin
from spacy import displacy

from moralization.input_data import InputOutput
from moralization.utils import is_interactive

import os
import pathlib
from pathlib import Path

from spacy.cli.init_config import fill_config
from sklearn.model_selection import train_test_split
from tempfile import mkdtemp
from collections import defaultdict
from collections.abc import Iterable
import shutil


class SpacySetup:
    """Helper class to organize and prepare spacy trainings data from xml/xmi files."""

    def __init__(self, data_dir, working_dir=None):
        """Handler for machine learning training and analysis.

        Args:
          data_dir(str/Path): Directory with data files
          working_dir(path, optional): Directory where the training data,
          configs and results are stored., defaults to Debug value
          config_file(Path, optional): Filename or path for the config file,
          defaults to searching in the working directory.

        """
        self.data_dir, self.working_dir = self._setup_working_dir(data_dir, working_dir)
        self.convert_data_to_spacy_doc()

    def _setup_working_dir(self, data_dir, working_dir):
        """Check if given path is possible and initialize directory if not already present.

        Args:
            data_dir (path): path to data directory
            working_dir (path): path to data directory

        Returns:
            path: path to data directory
            path: path to working directory
        """

        # maybe set default working_dir to tmp dir
        data_dir = Path(data_dir)

        if working_dir:
            working_dir = Path(working_dir)
        else:
            working_dir = Path(mkdtemp())

        pathlib.Path(working_dir).mkdir(exist_ok=True)
        return data_dir, working_dir

    def convert_data_to_spacy_doc(self):
        """Convert the given xmi/xml files to a spacy specific binary filesystem."""
        data_dict = InputOutput.read_data(self.data_dir)

        merging_dict = {
            "sc": "all",
            "task1": ["KAT1MoralisierendesSegment"],
            "task2": ["Moralwerte", "KAT2Subjektive_Ausdrcke"],
            "task3": ["Protagonistinnen", "Protagonistinnen2", "Protagonistinnen3"],
            "task4": ["KommunikativeFunktion"],
            "task5": ["Forderung"],
        }

        data_dict = self._add_dict_cat(data_dict, merging_dict)
        spans_dict = self._convert_dict_to_spans_dict(data_dict)

        self.doc_dict = self._convert_spans_dict_to_doc_dict(data_dict, spans_dict)

    def export_training_testing_data(self, output_dir=None):
        """Convert a list of spacy docs to a serialisable DocBin object and save it to disk.
        Automatically processes training and testing files.

        Args:
          output_dir(Path, optional): Path of the output directory where the data is saved, defaults to None.
          If None the working directory is used.

        Return:
            output_dir(Path): location of the stored data.

        """
        if output_dir is None:
            output_dir = self.working_dir
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        db_train = DocBin()
        db_dev = DocBin()

        for file in self.doc_dict.values():
            db_train.add(file["train"])
            db_dev.add(file["test"])

        db_train.to_disk(output_dir / "train.spacy")
        db_dev.to_disk(output_dir / "dev.spacy")
        return output_dir

    def _manage_visualisation_filenames(self, filenames):

        # check through given filenames and convert ints to key string
        if filenames is None:
            filename = list(self.doc_dict.keys())
        elif isinstance(filenames, str):
            filename = [filenames]
        elif isinstance(filenames, int):
            filename = [list(self.doc_dict.keys())[filenames]]
        elif isinstance(filenames, Iterable):
            filename = []
            for file in filenames:
                if isinstance(file, int):
                    filename.append(list(self.doc_dict.keys())[file])
                else:
                    filename.append(file)

        # check if all new filenames are in doc_dict.keys()
        for file in filename:
            if file not in list(self.doc_dict.keys()):
                raise IndexError(
                    f"The filename {file} is not provided in the dataset, which only has {list(self.doc_dict.keys())}."
                )

        return filename

    def visualize_data(self, filenames=None, type="all", style="span", spans_key="sc"):
        """Use the displacy class offered by spacy to visualize the current dataset.
            use SpacySetup.span_keys to show possible keys or use 'sc' for all.


        Args:
          filename(str/list, optional, optional): Specify which of the loaded files should be presented,
          if None all files are shown.
          This can also take a list., defaults to None
          display_type(str, optional, optional): Specify is only the trainings,
          the testing or all datapoints should be shown,options are: "all", "test" and "train". Defaults to "all"
          type: the visualization type given to displacy, available are "dep", "ent" and "span,
          defaults to "span".
          style:  (Default value = "span")

        Returns:
            Displacy.render
        """

        if isinstance(spans_key, list):
            raise NotImplementedError(
                "spacy does no support viewing multiple categories at once."
            )
            # we could manually add multiple categories to one span cat and display this new category.

        if spans_key != "sc" and spans_key not in self.span_keys:
            raise ValueError(
                f"""The provided key: {spans_key} is not valid.
                Please use one of the following {set(self.span_keys.keys())}"""
            )
        if type not in ["all", "test", "train"]:
            raise IndexError(
                f"Type argument must be either 'all', 'test' or 'train', but is {type}"
            )

        filename = self._manage_visualisation_filenames(filenames)

        if not is_interactive():
            raise NotImplementedError(
                "Please only use this function in a jupyter notebook for the time being."
            )
        if isinstance(spans_key, list):
            raise NotImplementedError(
                "spacy does no support viewing multiple categories at once."
            )
            # we could manually add multiple categories to one span cat and display this new category.

        if spans_key != "sc" and spans_key not in self.span_keys:
            raise ValueError(
                f"""The provided key: {spans_key} is not valid.
                Please use one of the following {set(self.span_keys.keys())}"""
            )

        return displacy.render(
            [self.doc_dict[file][type] for file in filename],
            style=style,
            options={"spans_key": spans_key},
        )

    def _add_dict_cat(self, data_dict, new_dict_cat):
        """Take the new_dict_cat dict and add its key as a main_cat to data_dict.
        The values are the total sub_dict_entries of the given list.

        Args:
          data_dict(dict: dict): The datadict generated from xmi files.
          new_dict_cat(dict): map new category to list of existing_categories.

        Return:
            dict: The data_dict with new span categories.
        """

        for file in data_dict.keys():
            for new_main_cat, new_cat_entries in new_dict_cat.items():
                if new_cat_entries == "all":
                    for main_cat in list(data_dict[file]["data"].keys()):
                        data_dict[file]["data"][new_main_cat].update(
                            data_dict[file]["data"][main_cat]
                        )
                else:
                    for old_main_cat in new_cat_entries:
                        data_dict[file]["data"][new_main_cat].update(
                            data_dict[file]["data"][old_main_cat]
                        )

        return data_dict

    def _convert_dict_to_spans_dict(self, data_dict):
        """Custom tranformation steps to convert our data_dict into a usable spacy.doc format.
        For this all annotations will be saved under the span_key 'sc'.

        Args:
          data_dict(dict: dict): The datadict generated from xmi files.

        Returns:
          dict: Dict of spacy.doc objects

        """
        self.span_keys = defaultdict(set)
        # nlp = spacy.blank("de")
        nlp = spacy.load(
            "de_core_news_sm",
            exclude=["lemmatizer", "ner", "morphologizer", "attribute_ruler"],
        )
        # defaultdict with structure:
        # file - main_cat - all/train/test
        spans_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for file in data_dict.keys():
            # sort out unused categories
            data_dict[file]["data"].pop("KAT5Ausformulierung", None)
            for main_cat_value in data_dict[file]["data"].values():
                main_cat_value.pop("Dopplung", None)

            doc = nlp(data_dict[file]["sofa"])

            for main_cat_key, main_cat_value in data_dict[file]["data"].items():
                for sub_cat_label, sub_cat_span_list in main_cat_value.items():
                    self.span_keys[main_cat_key].add(sub_cat_label)
                    for span in sub_cat_span_list:
                        spans_dict[file][main_cat_key]["all"].append(
                            doc.char_span(
                                span["begin"],
                                span["end"],
                                label=sub_cat_label,
                            )
                        )
                if len(spans_dict[file][main_cat_key]["all"]) > 1:
                    spans_train, spans_test = train_test_split(
                        spans_dict[file][main_cat_key]["all"],
                        test_size=0.2,
                        random_state=42,
                    )
                    (
                        spans_dict[file][main_cat_key]["train"],
                        spans_dict[file][main_cat_key]["test"],
                    ) = (spans_train, spans_test)
                else:
                    spans_dict[file][main_cat_key]["train"] = spans_dict[file][
                        main_cat_key
                    ]["all"]
            spans_dict[file]["doc"] = doc

        return spans_dict

    def _convert_spans_dict_to_doc_dict(self, data_dict, spans_dict):

        doc_dict = {}
        for file in data_dict.keys():
            doc = spans_dict[file].pop("doc", None)
            doc_test = doc.copy()
            doc_train = doc.copy()
            doc_dict[file] = {"all": doc, "test": doc_test, "train": doc_train}
            # each value now has the keys "all"/"train"/test, each with a list of spans
            for main_cat_key, main_cat_values in spans_dict[file].items():
                for usecase in ["all", "train", "test"]:
                    doc_dict[file][usecase].spans[main_cat_key] = main_cat_values[
                        usecase
                    ]

        return doc_dict


class SpacyTraining:

    """This class is used to configure and run spacy trainings."""

    def __init__(
        self, working_dir, training_file=None, testing_file=None, config_file=None
    ):
        self.working_dir = Path(working_dir)
        self.file_dict = self._find_files(training_file, testing_file, config_file)

    def _find_file(self, _file):
        """

        Args:
          _file:

        Returns:

        """
        if _file:
            _file = Path(_file)

        if not _file.is_file() and (self.working_dir / _file).is_file():
            _file = self.working_dir / _file

        else:
            raise FileNotFoundError(
                f"""{_file} could not be found as absolute path or in {self.working_dir}.
                Available files are: {list(self.working_dir.glob('*'))}"""
            )
        return _file

    def _find_files(self, training_file, testing_file, config_file):
        """

        Args:
          training_file:
          testing_file:
          config_file:

        Returns:

        """

        # use default names is no name is given.
        if training_file is None:
            training_file = "train.spacy"
        if testing_file is None:
            testing_file = "dev.spacy"

        # find the files either by absolute path or search default/relativ filename.
        training_file = self._find_file(training_file)
        testing_file = self._find_file(testing_file)
        config_file = self._check_config_file(config_file)

        file_dict = {
            "training": training_file,
            "testing": testing_file,
            "config": config_file,
        }

        return file_dict

    def _check_config_file(self, config_file):
        """

        Args:
          config_file:

        Returns:

        """
        # find config file as abs path, or as filename in the working directory.

        if config_file:
            config_file = Path(config_file)
            if config_file.is_file():
                config_file = Path(config_file)
            else:
                if (self.working_dir / config_file).is_file():
                    config_file = self.working_dir / config_file
                else:
                    raise FileNotFoundError(
                        "The given config file could not be found in the working directory:"
                        + f" {self.working_dir} or under {config_file.absolute()}"
                    )

        else:
            # search working dir for config file
            config_files = list(self.working_dir.glob("*.cfg"))
            if len(config_files) == 1:
                config_file = config_files[0]
            elif len(config_files) == 0:
                raise FileNotFoundError(
                    f"A config file was not provided and no config file could be found  in {self.working_dir}."
                )
            else:
                raise ValueError(
                    f"""A config file was not provided and multiple config files were found in {self.working_dir}.
                    Please provide only one or specify the filename."""
                )

        # after finding the config we use the provided spacy function to autofill all missing entries.
        fill_config(
            base_path=config_file,
            output_file=self.working_dir / "config_filled.cfg",
        ),

        return self.working_dir / "config_filled.cfg"

    def train(self, use_gpu=-1, overwrite=None):
        """Use the spacy training method to generate a new model.

        Args:
          use_gpu(int, optional, optional): enter the gpu device you want to use.
        Keep in Mind that cuda must be correctly installed, defaults to -1
          overwrite(_type_, optional, optional): additional config overwrites, defaults to None

        Returns:

        """
        from spacy.cli.train import train

        output = os.path.join(self.working_dir, "output")
        os.makedirs(output, exist_ok=True)
        if overwrite is None:
            overwrite = {}

        train(
            config_path=self.file_dict["config"].absolute(),
            output_path=output,
            use_gpu=use_gpu,
            overrides={
                "paths.train": self.file_dict["training"].absolute().as_posix(),
                "paths.dev": self.file_dict["testing"].absolute().as_posix(),
                **overwrite,
            },
        )

    def evaluate(self, validation_file=None):
        """

        Args:
          validation_file:  (Default value = None)

        Returns:

        """
        from spacy.cli.evaluate import evaluate

        if validation_file is None:
            validation_file = self.file_dict["testing"].absolute()

        evaluation_data = evaluate(
            self._best_model(),
            validation_file,
            output=os.path.join(self.working_dir, "output", "evaluation"),
        )
        return evaluation_data

    def test_model_with_string(self, test_string, options=None):
        """

        Args:
          test_string:

        Returns:

        """
        if not is_interactive():
            raise NotImplementedError(
                "Please only use this function in a jupyter notebook for the time being."
            )
        if options is None:
            options = {"spans_key": "task1"}

        nlp = spacy.load(self._best_model())
        doc = nlp(test_string)

        displacy.render(doc, style="span", options=options)
        displacy.render(doc, style="ent")

        return doc, nlp

        print("ents")
        for ent in doc.ents:
            print(ent, ent.label_)

        return doc, nlp

    def _best_model(self):
        """ """
        if os.path.isdir(os.path.join(self.working_dir, "output", "model-best")):
            return os.path.join(self.working_dir, "output", "model-best")
        else:
            raise FileNotFoundError(
                f"""No best model could be found in{os.path.join(self.working_dir,'output')}.
                Did you train your model before?"""
            )

    def save_best_model(self, output_name):
        output_name = Path(output_name)
        if output_name.exists():
            raise FileExistsError(
                f"The directory {output_name} already exists, please choose a unique name."
            )
        shutil.copytree(self._best_model(), output_name)
