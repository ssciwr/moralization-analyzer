from moralization.spacy_model import Spacy_Setup, Spacy_Training
from tempfile import mkdtemp
import pathlib
import pytest
from shutil import copy


def test_Spacy_Setup(data_dir):
    tmp_dir = mkdtemp()

    # test datadir and specific file path
    Spacy_Setup(data_dir)

    # test data_dir, working_dir and specific file.
    Spacy_Setup(data_dir, working_dir=tmp_dir)

    # test finding file by name in data_dir
    Spacy_Setup(
        pathlib.Path(__file__).parents[2].resolve() / "data" / "Training",
    )
    Spacy_Setup(
        pathlib.Path(__file__).parents[2].resolve() / "data" / "Training",
        working_dir=tmp_dir,
    )


def test_Spacy_Setup_convert_data_to_spacy(data_dir):

    # test datadir and specific file path
    test_setup = Spacy_Setup(data_dir)
    test_setup.convert_data_to_spacy_doc()
    assert list(test_setup.doc_dict.keys()) == [
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
    ]


def test_Spacy_Setup_export_training_testing_data(data_dir):
    tmp_dir = pathlib.Path(mkdtemp())
    test_setup = Spacy_Setup(data_dir)
    test_setup.convert_data_to_spacy_doc()
    test_setup.export_training_testing_data()
    assert len(list(test_setup.working_dir.glob("*.spacy"))) == 2
    test_setup.export_training_testing_data(tmp_dir)
    assert len(list(tmp_dir.glob("*.spacy"))) == 2

    tmp_dir = pathlib.Path(mkdtemp())
    test_setup = Spacy_Setup(data_dir, working_dir=tmp_dir)
    test_setup.convert_data_to_spacy_doc()
    test_setup.export_training_testing_data()

    assert len(list(test_setup.working_dir.glob("*.spacy"))) == 2


def test_Spacy_Setup_visualize_data(data_dir):
    test_setup = Spacy_Setup(data_dir)
    test_setup.convert_data_to_spacy_doc()
    with pytest.raises(NotImplementedError):
        test_setup.visualize_data()


def test_Spacy_Training(data_dir, config_file):
    tmp_dir = pathlib.Path(mkdtemp())

    test_setup = Spacy_Setup(data_dir, working_dir=tmp_dir)
    test_setup.convert_data_to_spacy_doc()
    test_setup.export_training_testing_data()
    # test no config found:
    with pytest.raises(FileNotFoundError):
        Spacy_Training(tmp_dir)

    copy(config_file, tmp_dir)
    # test with config
    Spacy_Training(tmp_dir, config_file="config.cfg")
    Spacy_Training(tmp_dir, config_file=tmp_dir / "config.cfg")

    Spacy_Training(
        tmp_dir,
        training_file="train.spacy",
        testing_file="dev.spacy",
        config_file="config.cfg",
    )
    with pytest.raises(FileNotFoundError):
        Spacy_Training(
            tmp_dir,
            training_file="noshow.spacy",
            testing_file="dev.spacy",
            config_file="config.cfg",
        )
    with pytest.raises(FileNotFoundError):
        Spacy_Training(
            tmp_dir,
            training_file="train.spacy",
            testing_file="noshow.spacy",
            config_file="config.cfg",
        )
    with pytest.raises(FileNotFoundError):
        Spacy_Training(
            tmp_dir,
            training_file="train.spacy",
            testing_file="dev.spacy",
            config_file="noshow.cfg",
        )

    copy(config_file, tmp_dir / "test.cfg")

    # test multiple configs found
    with pytest.raises(Exception):
        Spacy_Training(tmp_dir)

    tmp_dir = mkdtemp()


def test_Spacy_Training_training_testing(data_dir, config_file):
    tmp_dir = pathlib.Path(mkdtemp())

    test_setup = Spacy_Setup(data_dir, working_dir=tmp_dir)
    test_setup.convert_data_to_spacy_doc()
    test_setup.export_training_testing_data()
    copy(config_file, tmp_dir)
    training_test = Spacy_Training(tmp_dir, config_file="config.cfg")

    training_test.train(overwrite={"training.max_epochs": 5})
    training_test.evaluate()
