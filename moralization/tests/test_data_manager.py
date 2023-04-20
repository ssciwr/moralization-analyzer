from moralization.data_manager import DataManager
from moralization.transformers_data_handler import TransformersDataHandler
from tempfile import mkdtemp
from pathlib import Path
import pytest
import re
import numpy as np


@pytest.fixture
def get_transformers_lists(data_dir):
    dm = DataManager(data_dir)
    tdh = TransformersDataHandler()
    example_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    doc_dict = dm.doc_dict
    tdh.get_data_lists(doc_dict=doc_dict, example_name=example_name)
    tdh.generate_labels(doc_dict=doc_dict, example_name=example_name)
    sentence_list, label_list = tdh.structure_labels()
    return dm, sentence_list, label_list

def test_data_manager(data_dir):
    DataManager(data_dir)


def test_return_analyzer_result(data_dir):
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


def test_occurence_analysis(data_dir):
    dm = DataManager(data_dir)
    table = dm.occurence_analysis("table")
    assert table.shape == (18, 50)

    corr = dm.occurence_analysis("corr")
    assert corr.shape == (50, 50)

    heatmap = dm.occurence_analysis("heatmap")
    assert heatmap


def test_interactive_analysis(data_dir):
    dm = DataManager(data_dir)
    dm.interactive_analysis().show()


def test_visualize_data(data_dir):
    dm = DataManager(data_dir)
    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="all")

    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="test")

    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="train")

    with pytest.raises(KeyError):
        dm.visualize_data(_type="blub")


def test_export_data_DocBin(data_dir):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    test_files = dm.export_data_DocBin(tmp_dir)
    assert test_files[0].stem == "train"
    assert test_files[1].stem == "dev"
    assert dm.spacy_docbin_files[0].stem == "train"
    assert dm.spacy_docbin_files[1].stem == "dev"
    assert sorted(dm.spacy_docbin_files) == sorted(list(tmp_dir.glob("*.spacy")))

    # check if overwrite protection is triggered.
    with pytest.raises(FileExistsError):
        dm.export_data_DocBin(tmp_dir)
    dm.export_data_DocBin(tmp_dir, overwrite=True)

    # check for uninitilized directory
    dm.export_data_DocBin(tmp_dir / "a/")


def test_import_data_DocBin(data_dir):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    dm.export_data_DocBin(tmp_dir)

    dm2 = DataManager(data_dir)
    dm2.import_data_DocBin((tmp_dir))

    assert dm.spacy_docbin_files == dm2.spacy_docbin_files


def test_lists_to_df(get_transformers_lists):
    dm = get_transformers_lists[0]
    sentence_list = get_transformers_lists[1]
    label_list = get_transformers_lists[2]
    dm.lists_to_df(sentence_list, label_list)
    data_frame = dm.data_in_frame
    ref_sentence = ['"', 'Dann', 'kann', 'man', 'die', 'KMK', 'auflösen', '"', '#', '#', '#']
    assert data_frame["Sentences"][3] == ref_sentence
    ref_labels = [-100, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100]
    assert data_frame["Labels"][3] == ref_labels


def test_df_to_dataset(get_transformers_lists, data_dir):
    dm = get_transformers_lists[0]
    sentence_list = get_transformers_lists[1]
    label_list = get_transformers_lists[2]
    dm.lists_to_df(sentence_list, label_list)
    dm.df_to_dataset()
    ref_sentence = ['"', 'Dann', 'kann', 'man', 'die', 'KMK', 'auflösen', '"', '#', '#', '#']
    assert dm.raw_data_set["Sentences"][3] == ref_sentence
    ref_labels = [-100, 0, 0, 0, 0, 0, 0, -100, -100, -100, -100]
    assert dm.raw_data_set["Labels"][3] == ref_labels
    assert dm.train_test_set
    assert dm.train_test_set["test"]
    dm_nosplit = DataManager(data_dir)
    dm_nosplit.lists_to_df(sentence_list, label_list)
    dm_nosplit.df_to_dataset(split=False)
    assert not hasattr(dm_nosplit, "train_test_set")