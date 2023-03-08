from moralization.spacy_model import SpacySetup, SpacyTraining
from tempfile import mkdtemp
import pathlib
import pytest
from shutil import copy


def test_SpacySetup(data_dir):
    tmp_dir = mkdtemp()

    # test datadir and specific file path
    SpacySetup(data_dir)

    # test data_dir, working_dir and specific file.
    SpacySetup(data_dir, working_dir=tmp_dir)

    # test finding file by name in data_dir
    SpacySetup(
        pathlib.Path(__file__).parents[1].resolve() / "data",
    )
    SpacySetup(
        pathlib.Path(__file__).parents[1].resolve() / "data",
        working_dir=tmp_dir,
    )


def test_SpacySetup_convert_data_to_spacy(data_dir):

    # test datadir and specific file path
    test_setup = SpacySetup(data_dir)
    assert sorted(list(test_setup.doc_dict.keys())) == sorted(
        [
            "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
            "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
        ]
    )


def test_SpacySetup_export_training_testing_data(data_dir):
    tmp_dir = pathlib.Path(mkdtemp())
    test_setup = SpacySetup(data_dir)
    test_setup.export_training_testing_data()
    assert len(list(test_setup.working_dir.glob("*.spacy"))) == 2
    test_setup.export_training_testing_data(tmp_dir)
    assert len(list(tmp_dir.glob("*.spacy"))) == 2

    tmp_dir = pathlib.Path(mkdtemp())
    test_setup = SpacySetup(data_dir, working_dir=tmp_dir)
    test_setup.export_training_testing_data()

    assert len(list(test_setup.working_dir.glob("*.spacy"))) == 2


@pytest.mark.skip
def test_SpacySetup_manage_visualisation_filenames(data_dir):

    test_setup = SpacySetup(data_dir)

    # test filename logic
    test_setup._manage_visualisation_filenames([0])
    test_setup._manage_visualisation_filenames(0)
    test_setup._manage_visualisation_filenames(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB"
    )
    test_setup._manage_visualisation_filenames(
        ["test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB"]
    )
    assert (
        sorted(test_setup._manage_visualisation_filenames([0, 1]))
        == sorted(
            test_setup._manage_visualisation_filenames(
                [0, "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"]
            )
        )
        == sorted(
            test_setup._manage_visualisation_filenames(
                [
                    "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
                    "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
                ]
            )
        )
        == sorted(
            [
                "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
                "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
            ]
        )
    )

    with pytest.raises(IndexError):
        test_setup._manage_visualisation_filenames(4)
    with pytest.raises(IndexError):
        test_setup._manage_visualisation_filenames("test_not_found")


def test_SpacySetup_visualize_data(data_dir):
    test_setup = SpacySetup(data_dir)

    with pytest.raises(IndexError):
        test_setup.visualize_data(type="false_type")

    with pytest.raises(ValueError):
        test_setup.visualize_data(spans_key="false_key")

    with pytest.raises(NotImplementedError):
        test_setup.visualize_data(spans_key=["task1", "sc"])

    # test NotImplementedException when not in Jupyter Notebook
    with pytest.raises(NotImplementedError):
        test_setup.visualize_data()


def test_SpacyTraining(data_dir, config_file):
    tmp_dir = pathlib.Path(mkdtemp())

    test_setup = SpacySetup(data_dir, working_dir=tmp_dir)
    test_setup.export_training_testing_data()
    # test no config found:
    with pytest.raises(FileNotFoundError):
        SpacyTraining(tmp_dir)

    copy(config_file, tmp_dir)
    # test with config
    SpacyTraining(tmp_dir, config_file="config.cfg")
    SpacyTraining(tmp_dir, config_file=tmp_dir / "config.cfg")

    SpacyTraining(
        tmp_dir,
        training_file="train.spacy",
        testing_file="dev.spacy",
        config_file="config.cfg",
    )
    with pytest.raises(FileNotFoundError):
        SpacyTraining(
            tmp_dir,
            training_file="noshow.spacy",
            testing_file="dev.spacy",
            config_file="config.cfg",
        )
    with pytest.raises(FileNotFoundError):
        SpacyTraining(
            tmp_dir,
            training_file="train.spacy",
            testing_file="noshow.spacy",
            config_file="config.cfg",
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


def test_SpacyTraining_training_testing(data_dir, config_file):
    tmp_dir = pathlib.Path(mkdtemp())

    test_setup = SpacySetup(data_dir, working_dir=tmp_dir)
    test_setup.export_training_testing_data()
    copy(config_file, tmp_dir)
    training_test = SpacyTraining(tmp_dir, config_file="config.cfg")

    training_test.train(overwrite={"training.max_epochs": 5})
    training_test.evaluate()
    with pytest.raises(NotImplementedError):
        training_test.test_model_with_string("Dies ist ein toller Test!")

    training_test.save_best_model(tmp_dir / "test_model")
    with pytest.raises(FileExistsError):
        training_test.save_best_model(".")
