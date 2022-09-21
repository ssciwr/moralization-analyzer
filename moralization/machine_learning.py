from logging import exception
import spacy
from spacy.tokens import DocBin
from moralization import InputOutput
import glob
import os
import random
import subprocess


class Machine_Learning:  # name is subject to change
    """Namespace class for Machine Learning
    The config file can also be located in the working directory.
    """

    def __init__(self, data_dir: str, working_dir=None, config_file=None):
        """Handler for machine learning training and analysis

        :param data_dir: Directory with data files
        :type data_dir: str
        :param working_dir: Directory where the training data, configs and results are stored., defaults to Debug value
        :type working_dir: _type_, optional
        :param config_file: Filename or path for the config file, defaults to searching in the working directory.
        :type config_file: _type_, optional

        """
        # maybe set default working_dir to tmp dir
        data_dir = os.path.abspath(data_dir)

        if working_dir:
            working_dir = os.path.abspath(working_dir)
        else:
            working_dir = os.path.abspath("../data/Training/")

        # find config file as abs path, or as filename in the working directory.
        if config_file:
            if os.path.isfile(config_file):
                config_file = os.path.abspath(config_file)
            else:
                if os.path.isfile(os.path.join(working_dir, config_file)):
                    config_file = os.path.abspath(
                        os.path.join(working_dir, config_file)
                    )
                else:
                    raise Exception(
                        f"The given config file could not be found in the working directory: {working_dir}"
                    )

        else:
            # search working dir for config file
            config_files = glob.glob(os.path.join(working_dir, "*.cfg"))
            if len(config_files) == 1:
                config_file = config_files[0]
            elif len(config_files) == 0:
                raise FileNotFoundError(
                    f"A config file was not provided and no config file could be found  in {data_dir}."
                )
            else:
                raise Exception(
                    f"A config file was not provided and multiple config files were found in {data_dir}. Please provide only one or specify the filename."
                )

        self.working_dir = working_dir
        self.data_dir = data_dir

        self.train_file, self.test_file = self.prepare_spacy_data(
            self.data_dir, self.working_dir
        )
        self.config_file = self.prepare_spacy_config(config_file, self.working_dir)

    def train(self, use_gpu=-1):
        from spacy.cli.train import train

        output = os.path.join(self.working_dir, "output")
        os.makedirs(output, exist_ok=True)

        train(
            config_path=self.config_file,
            output_path=output,
            use_gpu=use_gpu,
            overrides={
                "paths.train": self.train_file,
                "paths.dev": self.test_file,
            },
        )

    def evaluate(self, validation_file=None):
        from spacy.cli.evaluate import evaluate

        if validation_file is None:
            validation_file = self.test_file

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
                f"No best model could be found in{os.path.join(self.working_dir,'output')}. Did you train your model before?"
            )

    @staticmethod
    def prepare_spacy_data(dir_path, working_dir):
        """Prepare data for spacy analysis.

        :param dir_path: Path to data dir, defaults to None
        :type dir_path: _type_, str,path

        :raises Warning: data directory and data dict cant both be given.
        """

        data_dict = InputOutput.get_input_dir(dir_path)

        nlp = spacy.blank("de")
        # nlp.add_pipe("sentencizer",config={"punct_chars":['!','.','?']})
        db_train = DocBin()
        db_dev = DocBin()

        for file in data_dict.keys():
            doc_train = nlp(data_dict[file]["sofa"])
            doc_dev = nlp(data_dict[file]["sofa"])
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

            # split data for each file in test and training
            random.shuffle(ents)
            ents_train = ents[: int(0.95 * len(ents))]
            ents_test = ents[int(0.05 * len(ents)) :]

            # https://explosion.ai/blog/spancat
            # use spancat for multiple labels on the same token

            doc_train.spans["sc"] = ents_train
            db_train.add(doc_train)

            doc_dev.spans["sc"] = ents_test
            db_dev.add(doc_dev)

        db_train.to_disk(os.path.join(working_dir, "train.spacy"))
        db_dev.to_disk(os.path.join(working_dir, "dev.spacy"))
        return os.path.join(working_dir, "train.spacy"), os.path.join(
            working_dir, "dev.spacy"
        )

    @staticmethod
    def prepare_spacy_config(config_file, working_dir):
        from pathlib import Path
        from spacy.cli.init_config import fill_config

        fill_config(
            base_path=config_file,
            output_file=Path(os.path.join(working_dir, "config.cfg")),
        )
        return os.path.join(working_dir, "config.cfg")
