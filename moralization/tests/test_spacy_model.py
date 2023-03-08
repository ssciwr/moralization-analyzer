from moralization.spacy_model import SpacyTraining, SpacyDataHandler
from tempfile import mkdtemp
import pathlib
import pytest
from shutil import copy
import re
import spacy


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
    assert list(test_handler.db_files) == list(tmp_dir.glob("*.spacy"))


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


def test_spacy_training(doc_dicts, config_file):
    tmp_dir = pathlib.Path(mkdtemp())

    test_handler = SpacyDataHandler()
    db_files = test_handler.export_training_testing_data(
        doc_dicts[1], doc_dicts[2], tmp_dir
    )

    # test no config found:
    with pytest.raises(FileNotFoundError):
        SpacyTraining(tmp_dir, db_files[0], db_files[1])

    copy(config_file, tmp_dir)
    # test with config
    SpacyTraining(tmp_dir, db_files[0], db_files[1], config_file=CONFIG_CFG)
    SpacyTraining(tmp_dir, db_files[0], db_files[1], config_file=tmp_dir / CONFIG_CFG)

    with pytest.raises(FileNotFoundError):
        SpacyTraining(
            tmp_dir,
            training_file="noshow.spacy",
            testing_file="dev.spacy",
            config_file=CONFIG_CFG,
        )
    with pytest.raises(FileNotFoundError):
        SpacyTraining(
            tmp_dir,
            training_file="train.spacy",
            testing_file="noshow.spacy",
            config_file=CONFIG_CFG,
        )
    with pytest.raises(FileNotFoundError):
        SpacyTraining(
            tmp_dir,
            training_file="train.spacy",
            testing_file="dev.spacy",
            config_file="noshow.cfg",
        )

    copy(config_file, tmp_dir / "test.cfg")

    # test multiple configs found
    with pytest.raises(Exception):
        SpacyTraining(tmp_dir)


def test_training_testing(doc_dicts, config_file):

    tmp_dir = pathlib.Path(mkdtemp())

    test_handler = SpacyDataHandler()
    db_files = test_handler.export_training_testing_data(
        doc_dicts[1], doc_dicts[2], tmp_dir
    )

    training_test = SpacyTraining(
        tmp_dir, db_files[0], db_files[1], config_file=config_file
    )

    best_model_path = pathlib.Path(
        training_test.train(overwrite={"training.max_epochs": 5})
    )
    # test if filename already exists

    training_test.evaluate(tmp_dir / EVALUATION_JSON, db_files[1], best_model_path)
    training_test.evaluate(tmp_dir / EVALUATION_JSON, db_files[1], best_model_path)
    assert len(list(tmp_dir.glob("*.json"))) == 2

    training_test.evaluate(tmp_dir / EVALUATION_JSON, db_files[1], best_model_path)

    # # wrong validation file
    with pytest.raises(RuntimeError):
        training_test.evaluate(
            tmp_dir / EVALUATION_JSON, "nonexistend_file.xyz", best_model_path
        )
    with pytest.raises(FileNotFoundError):
        training_test.evaluate(
            tmp_dir / EVALUATION_JSON, "nonexistend_file.spacy", best_model_path
        )

    with pytest.raises(IsADirectoryError):
        training_test.evaluate(tmp_dir, db_files[1], best_model_path)

    with pytest.raises(NotImplementedError):
        training_test.test_model_with_string(
            best_model_path, "Dies ist ein toller Test!"
        )
    with pytest.raises(NotImplementedError):
        training_test.test_model_with_string(
            spacy.load(best_model_path), "Dies ist ein toller Test!"
        )
