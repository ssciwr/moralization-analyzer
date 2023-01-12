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


class Spacy_Setup:
    """Helper class to organize and prepare spacy trainings data from xml/xmi files."""

    def __init__(self, data_dir, working_dir=None):
        """Handler for machine learning training and analysis
        :param data_dir: Directory with data files
        :type data_dir: str/Path
        :param working_dir: Directory where the training data, configs and results are stored., defaults to Debug value
        :type working_dir: _type_, optional
        :param config_file: Filename or path for the config file, defaults to searching in the working directory.
        :type config_file: _type_, optional
        """

        self.data_dir, self.working_dir = self._setup_working_dir(data_dir, working_dir)

    def _setup_working_dir(self, data_dir, working_dir):

        # maybe set default working_dir to tmp dir
        data_dir = Path(data_dir)

        if working_dir:
            working_dir = Path(working_dir)
        else:
            working_dir = Path(mkdtemp())

        pathlib.Path(working_dir).mkdir(exist_ok=True)
        return data_dir, working_dir

    def convert_data_to_spacy_doc(self):
        """convert the given xmi/xml files to a spacy specific binary filesystem.

        :param output_dir: where to store generated files. If None is given the working dir will be used
        :type output_dir: dir
        """
        data_dict = InputOutput.read_data(self.data_dir)

        self.doc_dict = self._convert_dict_to_doc_dict(data_dict)

        return self.doc_dict

    def export_training_testing_data(self, output_dir=None):
        """_summary_

        :param output_dir: _description_, defaults to None
        :type output_dir: _type_, optional
        """
        if output_dir is None:
            output_dir = self.working_dir

        db_train = DocBin()
        db_dev = DocBin()

        for file in self.doc_dict.values():
            db_train.add(file["train"])
            db_dev.add(file["dev"])

        db_train.to_disk(output_dir / "train.spacy")
        db_dev.to_disk(output_dir / "dev.spacy")

    def visualize_data(self, filename=None, type="all", style="span"):
        """Use the displacy class offered by spacy to visualize the current dataset.

        :param filename:    Specify which of the loaded files should be presented, if None all files are shown.
                            This can also take a list., defaults to None
        :type filename: str/list, optional
        :param type: Specify is only the trainings, the testing or all datapoints should be shown, defaults to "all"
        :type type: str, optional
        :param type: the visualization type given to displacy, available are "dep", "ent" and "span, defaults to "span".

        """
        if not is_interactive():
            raise NotImplementedError(
                "Please only use this function in a jupyter notebook for the time being."
            )

        if filename is None:
            filename = list(self.doc_dict.keys())
        return displacy.render(
            [self.doc_dict[file][type] for file in filename], style=style
        )

    def _convert_dict_to_doc_dict(self, data_dict):

        nlp = spacy.blank("de")
        doc_dict = {}

        for file in data_dict.keys():
            doc_train = nlp(data_dict[file]["sofa"])
            doc_dev = nlp(data_dict[file]["sofa"])
            doc_all = nlp(data_dict[file]["sofa"])

            ents = []

            for main_cat_key, main_cat_value in data_dict[file]["data"].items():
                if main_cat_key != "KAT5Ausformulierung":
                    for sub_cat_label, sub_cat_span_list in main_cat_value.items():
                        if sub_cat_label != "Dopplung":
                            for span in sub_cat_span_list:
                                spacy_span = doc_train.char_span(
                                    span["begin"],
                                    span["end"],
                                    label=sub_cat_label,
                                )
                                ents.append(spacy_span)

            ents_train, ents_test = train_test_split(
                ents, test_size=0.2, random_state=42
            )

            doc_train.spans["sc"] = ents_train
            doc_dev.spans["sc"] = ents_test
            doc_all.spans["sc"] = ents

            doc_dict[file] = {"train": doc_train, "dev": doc_dev, "all": doc_all}

        return doc_dict


class Spacy_Training:

    """This class is used to configure and run spacy trainings."""

    def __init__(
        self, working_dir, training_file=None, testing_file=None, config_file=None
    ):
        self.working_dir = Path(working_dir)
        self.file_dict = self._find_files(training_file, testing_file, config_file)

    def _find_file(self, _file):
        if _file:
            _file = Path(_file)

        if not _file.is_file() and (self.working_dir / _file).is_file():
            file = self.working_dir / _file

        else:
            raise FileNotFoundError(
                f"""{_file} could not be found as absolute path or in {self.working_dir}.
                Available files are: {list(self.working_dir.glob('*'))}"""
            )
        return file

    def _find_files(self, training_file, testing_file, config_file):

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
                        f"The given config file could not be found in the working directory: {self.working_dir}"
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
                raise Exception(
                    f"""A config file was not provided and multiple config files were found in {self.working_dir}.
                    Please provide only one or specify the filename."""
                )

        # after finding the config we use the provided spacy function to autofill all missing entries.
        fill_config(
            base_path=config_file, output_file=self.working_dir / "config_filled.cfg"
        ),

        return self.working_dir / "config_filled.cfg"

    def train(self, use_gpu=-1, overwrite=None):
        """Use the spacy training method to generate a new model.

        :param use_gpu: enter the gpu device you want to use.
            Keep in Mind that cuda must be correctly installed, defaults to -1
        :type use_gpu: int, optional
        :param overwrite: additional config overwrites, defaults to None
        :type overwrite: _type_, optional
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
        from spacy.cli.evaluate import evaluate

        if validation_file is None:
            validation_file = self.file_dict["testing"].absolute()

        evaluation_data = evaluate(
            self._best_model(),
            validation_file,
            output=os.path.join(self.working_dir, "output", "evaluation"),
        )
        return evaluation_data

    def test_model_with_string(self, test_string):
        nlp = spacy.load(self._best_model())
        doc = nlp(test_string)
        for span in doc.spans["sc"]:
            print(span, span.label_)

    def _best_model(self):
        if os.path.isdir(os.path.join(self.working_dir, "output", "model-best")):
            return os.path.join(self.working_dir, "output", "model-best")
        else:
            raise FileNotFoundError(
                f"""No best model could be found in{os.path.join(self.working_dir,'output')}.
                Did you train your model before?"""
            )
