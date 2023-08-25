from moralization.spacy_data_handler import SpacyDataHandler
from moralization import DataManager
import pytest
import re


CONFIG_CFG = "config.cfg"
EVALUATION_JSON = "evaluation.json"


@pytest.fixture
def get_dataset_task1(data_dir):
    dm = DataManager(data_dir)
    return dm.train_test_set


@pytest.fixture
def get_dataset_task2(data_dir):
    dm = DataManager(data_dir, task="task2")
    return dm.train_test_set


def test_docbin_from_dataset(get_dataset_task1, get_dataset_task2, tmp_path):
    data_path = SpacyDataHandler.docbin_from_dataset(
        get_dataset_task1, task="task1", data_split="test", output_dir=tmp_path
    )
    test_file = tmp_path / "dev.spacy"
    assert data_path == test_file
    assert test_file.exists()
    data_path = SpacyDataHandler.docbin_from_dataset(
        get_dataset_task1, task="task1", data_split="train", output_dir=tmp_path
    )
    train_file = tmp_path / "train.spacy"
    assert data_path == train_file
    assert train_file.exists()
    data_path = SpacyDataHandler.docbin_from_dataset(
        get_dataset_task2,
        task="task2",
        data_split="train",
        output_dir=tmp_path,
        overwrite=True,
    )
    train_file = tmp_path / "train.spacy"
    assert data_path == train_file
    assert train_file.exists()


def test_check_docs(doc_dict):
    SpacyDataHandler._check_docs(next(iter(doc_dict.values())), task="task1")


def test_import_training_testing_data(get_dataset_task1, tmp_path):
    data_path_train = SpacyDataHandler.docbin_from_dataset(
        get_dataset_task1, task="task1", data_split="train", output_dir=tmp_path
    )
    data_path_test = SpacyDataHandler.docbin_from_dataset(
        get_dataset_task1, task="task1", data_split="test", output_dir=tmp_path
    )
    db_files = SpacyDataHandler.import_training_testing_data(input_dir=tmp_path)
    assert db_files == [data_path_train, data_path_test]
    db_files = SpacyDataHandler.import_training_testing_data(
        tmp_path, "train.spacy", "dev.spacy"
    )
    assert db_files == [data_path_train, data_path_test]
    db_files = SpacyDataHandler.import_training_testing_data(
        train_file=data_path_train, test_file=data_path_test
    )
    assert db_files == [data_path_train, data_path_test]
    with pytest.raises(
        FileNotFoundError,
        match="Please provide either a directory or the file locations.",
    ):
        SpacyDataHandler.import_training_testing_data()
    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            "When providing a data file location, please also provide the other one."
        ),
    ):
        SpacyDataHandler.import_training_testing_data(test_file="test.spacy")
    with pytest.raises(TypeError):
        SpacyDataHandler.import_training_testing_data(
            test_file="test.xyz", train_file="test.spacy"
        )
    with pytest.raises(
        FileNotFoundError,
        match=re.escape("No trainings file in"),
    ):
        SpacyDataHandler.import_training_testing_data(
            test_file="test1.spacy", train_file="test2.spacy"
        )
    with pytest.raises(
        FileNotFoundError,
        match=re.escape("No trainings file in"),
    ):
        SpacyDataHandler.import_training_testing_data(
            tmp_path, test_file="test.spacy", train_file="test2.spacy"
        )
    with pytest.raises(
        FileNotFoundError,
        match=re.escape("No test file in"),
    ):
        SpacyDataHandler.import_training_testing_data(
            tmp_path, test_file="test1.spacy", train_file="train.spacy"
        )
