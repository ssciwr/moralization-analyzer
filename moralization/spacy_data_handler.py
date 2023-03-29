from spacy.tokens import DocBin
from pathlib import Path
from tempfile import mkdtemp


class SpacyDataHandler:
    """Helper class to organize and prepare spacy trainings data."""

    def export_training_testing_data(
        self, train_dict, test_dict, output_dir=None, overwrite=False
    ):
        """Convert a list of spacy docs to a serialisable DocBin object and save it to disk.
        Automatically processes training and testing files.

        Args:
            train_dict(dict): internally handled data storage.
            test_dict(dict): internally handled data storage.
            output_dir(list[Path], optional): Path of the output directory where the data is saved, defaults to None.
            If None the working directory is used.
            overwrite(bool, optional): wether or not the spacy files should be written
            even if files are already present.
        Return:
            db_files(list[Path]) the location of the written files.
        """

        if output_dir is None:
            output_dir = Path(mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        train_filename = output_dir / "train.spacy"
        dev_filename = output_dir / "dev.spacy"

        # check if files already exists, only if overwrite is False:

        if overwrite is False:
            if train_filename.exists() or dev_filename.exists():
                raise FileExistsError(
                    "The given directory already has a training and testing file."
                    + " Please choose a new directory or set overwrite to True."
                    + f"Given directory is: {output_dir}"
                )

        db_train = DocBin()
        db_test = DocBin()

        for doc_train, doc_test in zip(train_dict.values(), test_dict.values()):
            db_train.add(doc_train)
            db_test.add(doc_test)

        db_train.to_disk(train_filename)
        db_test.to_disk(dev_filename)
        self.db_files = [train_filename, dev_filename]
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
