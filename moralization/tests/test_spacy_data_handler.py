from moralization.spacy_data_handler import SpacyDataHandler
from tempfile import mkdtemp
import pathlib
import pytest
import re


CONFIG_CFG = "config.cfg"
EVALUATION_JSON = "evaluation.json"


def test_export_training_testing_data(doc_dicts):
    # test export default filename
    test_handler = SpacyDataHandler()

    test_handler.export_training_testing_data(doc_dicts[1], doc_dicts[2])
    assert len(list(test_handler.db_files)) == 2

    # test export with dir
    tmp_dir = pathlib.Path(mkdtemp())
    test_handler.export_training_testing_data(doc_dicts[1], doc_dicts[2], tmp_dir)
    assert sorted(list(test_handler.db_files)) == sorted(list(tmp_dir.glob("*.spacy")))


def test_import_training_testing_data(doc_dicts):
    tmp_dir = pathlib.Path(mkdtemp())

    test_handler = SpacyDataHandler()
    db_files = test_handler.export_training_testing_data(
        doc_dicts[1], doc_dicts[2], tmp_dir
    )
    test_handler2 = SpacyDataHandler()

    db_files2 = test_handler2.import_training_testing_data(tmp_dir)
    assert db_files == db_files2

    db_files3 = test_handler2.import_training_testing_data(
        tmp_dir, "train.spacy", "dev.spacy"
    )
    assert db_files == db_files3

    db_files4 = test_handler2.import_training_testing_data(
        train_file=db_files[0], test_file=db_files[1]
    )
    assert db_files == db_files4

    with pytest.raises(
        FileNotFoundError,
        match="Please provide either a directory or the file locations.",
    ):
        test_handler2.import_training_testing_data()

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            "When providing a data file location, please also provide the other one."
        ),
    ):
        test_handler2.import_training_testing_data(test_file="test.spacy")

    with pytest.raises(TypeError):
        test_handler2.import_training_testing_data(
            test_file="test.xyz", train_file="test.spacy"
        )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape("No trainings file in"),
    ):
        test_handler2.import_training_testing_data(
            test_file="test1.spacy", train_file="test2.spacy"
        )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape("No trainings file in"),
    ):
        test_handler2.import_training_testing_data(
            tmp_dir, test_file="test.spacy", train_file="test2.spacy"
        )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape("No test file in"),
    ):
        test_handler2.import_training_testing_data(
            tmp_dir, test_file="test1.spacy", train_file="train.spacy"
        )
