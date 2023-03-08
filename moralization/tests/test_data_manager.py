from moralization.data_manager import DataManager
from tempfile import mkdtemp
from pathlib import Path
import pytest
import re
import numpy as np


def test_DataManager(data_dir):
    DataManager(data_dir)


def test_DataManager_return_analyzer_result(data_dir):
    dm = DataManager(data_dir)
    dm.return_analyzer_result()

    test1 = dm.return_analyzer_result("frequency")
    assert test1.shape == (25, 15)

    test2 = dm.return_analyzer_result("length")
    assert test2.shape == (25, 15)

    test3 = dm.return_analyzer_result("span_distinctiveness")
    assert test3.shape == (25, 15)

    test4 = dm.return_analyzer_result("boundary_distinctiveness")
    assert test4.shape == (25, 15)

    assert not np.array_equal(test1.values, test2.values)
    assert not np.array_equal(test2.values, test3.values)
    assert not np.array_equal(test3.values, test4.values)
    assert not np.array_equal(test1.values, test4.values)

    with pytest.raises(KeyError, match=re.escape("result_type")):
        dm.return_analyzer_result("something_else")


def test_DataManager_occurence_analysis(data_dir):
    dm = DataManager(data_dir)
    table = dm.occurence_analysis("table")
    assert table.shape == (18, 50)

    corr = dm.occurence_analysis("corr")
    assert corr.shape == (50, 50)

    heatmap = dm.occurence_analysis("heatmap")
    assert heatmap


def test_DataManager_interactive_analysis(data_dir):
    dm = DataManager(data_dir)
    dm.interactive_analysis().show()


def test_DataManager_visualize_data(data_dir):
    dm = DataManager(data_dir)
    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="all")

    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="test")

    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="train")

    with pytest.raises(KeyError):
        dm.visualize_data(_type="blub")


def test_DataManager_export_data_DocBin(data_dir):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    test_files = dm.export_data_DocBin(tmp_dir)
    assert test_files[0].stem == "train"
    assert test_files[1].stem == "dev"
    assert dm.spacy_docbin_files[0].stem == "train"
    assert dm.spacy_docbin_files[1].stem == "dev"
    assert dm.spacy_docbin_files == list(tmp_dir.glob("*.spacy"))

    # check if overwrite protection is triggered.
    with pytest.raises(FileExistsError):
        dm.export_data_DocBin(tmp_dir)
    dm.export_data_DocBin(tmp_dir, overwrite=True)

    # check for uninitilized directory
    dm.export_data_DocBin(tmp_dir / "a/")


def test_DataManager_import_data_DocBin(data_dir):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    dm.export_data_DocBin(tmp_dir)

    dm2 = DataManager(data_dir)
    dm2.import_data_DocBin((tmp_dir))

    assert dm.spacy_docbin_files == dm2.spacy_docbin_files


def test_DataManager_spacy_train(data_dir, config_file):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    test_files = dm.export_data_DocBin(tmp_dir)
    dm.spacy_train(working_dir=tmp_dir, config=config_file, n_epochs=5)
    assert (tmp_dir / "output" / "model-best").exists()

    # also test spacy_import_model
    dm2 = DataManager(data_dir)
    dm2.spacy_import_model(tmp_dir / "output" / "model-best")
    assert dm2.spacy_model_path == dm.spacy_model_path

    # test spacy_validation
    dm.spacy_validation()
    dm.spacy_validation(test_files[1])
    dm.spacy_validation(working_dir=tmp_dir)

    # test no validation file:
    with pytest.raises(FileNotFoundError):
        dm2.spacy_validation(working_dir=tmp_dir)

    # test spacy_test_with_string

    with pytest.raises(NotImplementedError):
        dm.spacy_test_string("Dies ist ein toller Test!")

    dm3 = DataManager(data_dir)

    with pytest.raises(ValueError):
        dm3.spacy_test_string("Dies ist ein toller Test!")
