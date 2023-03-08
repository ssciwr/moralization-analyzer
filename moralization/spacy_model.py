import spacy
from spacy.tokens import DocBin

import os
from pathlib import Path

from spacy.cli.init_config import fill_config
from tempfile import mkdtemp
from moralization.plot import visualize_data
import shutil


class SpacyDataHandler:
    """Helper class to organize and prepare spacy trainings data."""

    def export_training_testing_data(self, train_dict, test_dict, output_dir=None):
        """Convert a list of spacy docs to a serialisable DocBin object and save it to disk.
        Automatically processes training and testing files.

        Args:
          output_dir(list[Path], optional): Path of the output directory where the data is saved, defaults to None.
          If None the working directory is used.
        Return:
            db_files(list[Path]) the location of the written files.
        """

        if output_dir is None:
            output_dir = Path(mkdtemp())
        else:
            output_dir = Path(output_dir)

        db_train = DocBin()
        db_test = DocBin()

        for doc_train, doc_test in zip(train_dict.values(), test_dict.values()):
            db_train.add(doc_train)
            db_test.add(doc_test)
        db_train.to_disk(output_dir / "train.spacy")
        db_test.to_disk(output_dir / "dev.spacy")
        self.db_files = [output_dir / "train.spacy", output_dir / "dev.spacy"]
        return self.db_files

    def _check_files(self, input_dir=None, train_file=None, test_file=None):
        if input_dir is None and test_file is None and train_file is None:
            raise FileNotFoundError(
                "Please provide either a directory or the file locations."
            )

        if (train_file is not None and test_file is None) or (
            train_file is None and test_file is not None
        ):
            raise FileNotFoundError(
                "When providing a data file location, please also provide the other one."
                + f"Currently `train_file` is {train_file} and `test_file` is {test_file}"
            )

        if train_file and test_file:
            train_file = Path(train_file)
            test_file = Path(test_file)
            # check if files are spacy
            if train_file.suffix != ".spacy" or test_file.suffix != ".spacy":
                raise TypeError("The provided files are not spacy binaries.")

            # if both files exists we can exit at this point.
            if train_file.exists() and test_file.exists():
                return train_file, test_file

        # if no files are given use the default values
        else:
            train_file = Path("train.spacy")
            test_file = Path("dev.spacy")

        # if not we search in the current or given working directory
        if input_dir is None:
            input_dir = Path.cwd()
        else:
            input_dir = Path(input_dir)

        # search the directory for the files.

        input_dir = Path(input_dir)
        if (input_dir / train_file).exists():
            db_train = input_dir / train_file
        else:
            raise FileNotFoundError(f"No trainings file in {input_dir}.")

        if (input_dir / test_file).exists():
            db_test = input_dir / test_file
        else:
            raise FileNotFoundError(f"No test file in {input_dir}.")

        return db_train, db_test

    def import_training_testing_data(
        self, input_dir=None, train_file=None, test_file=None
    ):
        db_train, db_test = self._check_files(input_dir, train_file, test_file)
        self.db_files = [db_train, db_test]
        return self.db_files


class SpacyTraining:

    """This class is used to configure and run spacy trainings."""

    def __init__(self, working_dir, training_file, testing_file, config_file=None):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        self.config_file = self._check_config_file(config_file)
        self.training_file = Path(training_file)
        self.testing_file = Path(testing_file)

        if not self.training_file.exists():
            raise FileNotFoundError(f"The file {self.training_file} does not exist.")

        if not self.testing_file.exists():
            raise FileNotFoundError(f"The file {self.testing_file} does not exist.")

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
            config_path=self.config_file.absolute(),
            output_path=output,
            use_gpu=use_gpu,
            overrides={
                "paths.train": self.training_file.absolute().as_posix(),
                "paths.dev": self.testing_file.absolute().as_posix(),
                **overwrite,
            },
        )

        return self._best_model()

    @staticmethod
    def evaluate(output_file, validation_file, model):

        from spacy.cli.evaluate import evaluate

        output_file = Path(output_file)
        validation_file = Path(validation_file)

        if output_file.is_dir():
            raise IsADirectoryError(
                "spacy.evaluate needs an output file not a directory."
            )

        if output_file.exists():
            i = 1
            while output_file.exists():
                print(output_file)
                output_file = output_file.parents[0] / f"evaluation_{i}.json"
                i = i + 1

        if validation_file.suffix != ".spacy":
            raise RuntimeError(f"The file '{validation_file}' is not a spacy binary.")

        if not validation_file.exists():
            raise FileNotFoundError(f"The file '{validation_file}' does not exist.")

        evaluation_data = evaluate(
            model,
            validation_file,
            output=output_file,
        )
        return evaluation_data

    @staticmethod
    def test_model_with_string(
        model,
        test_string,
        style="span",
    ):
        if isinstance(model, str) or isinstance(model, Path):
            model = spacy.load(model)

        doc_dict = {"test_doc": model(test_string)}
        # this should only ever have one spans key
        # I hope..
        spans_key = list(doc_dict["test_doc"].spans.keys())[0]
        return visualize_data(doc_dict, style=style, spans_key=spans_key)

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
