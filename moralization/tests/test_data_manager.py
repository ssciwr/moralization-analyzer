from moralization.data_manager import DataManager
from tempfile import mkdtemp
from pathlib import Path
import pytest
import re
import numpy as np
from datasets import load_dataset


@pytest.fixture
def get_dataset():
    data_set = load_dataset("iulusoy/test-data")
    return data_set


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


def test_interactive_correlation_analysis(data_dir):
    dm = DataManager(data_dir)
    with pytest.raises(EnvironmentError):
        dm.interactive_correlation_analysis()


def test_interactive_data_visualization(data_dir):
    dm = DataManager(data_dir)
    with pytest.raises(EnvironmentError):
        dm.interactive_data_visualization()


def test_interactive_data_analysis(data_dir):
    dm = DataManager(data_dir)
    with pytest.raises(EnvironmentError):
        dm.interactive_data_analysis()


def test_visualize_data(data_dir):
    dm = DataManager(data_dir)
    dm.visualize_data(_type="all")

    dm.visualize_data(_type="test")

    dm.visualize_data(_type="train")

    dm.visualize_data(_type="all", spans_key="task1")

    with pytest.raises(NotImplementedError):
        dm.visualize_data(_type="all", spans_key=["task1", "task2"])

    with pytest.raises(KeyError):
        dm.visualize_data(_type="blub")


def test_export_data_DocBin(data_dir):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    test_files = dm.export_data_DocBin(tmp_dir, check_data_integrity=False)
    assert test_files[0].stem == "train"
    assert test_files[1].stem == "dev"
    assert dm.spacy_docbin_files[0].stem == "train"
    assert dm.spacy_docbin_files[1].stem == "dev"
    assert sorted(dm.spacy_docbin_files) == sorted(list(tmp_dir.glob("*.spacy")))

    # check if overwrite protection is triggered.
    with pytest.raises(FileExistsError):
        dm.export_data_DocBin(tmp_dir, check_data_integrity=False)

    # check integrity check working
    with pytest.raises(ValueError):
        dm.export_data_DocBin(tmp_dir, check_data_integrity=True)

    dm.export_data_DocBin(tmp_dir, overwrite=True, check_data_integrity=False)

    # check for uninitilized directory
    dm.export_data_DocBin(tmp_dir / "a/", check_data_integrity=False)


def test_import_data_DocBin(data_dir):
    dm = DataManager(data_dir)
    tmp_dir = Path(mkdtemp())
    dm.export_data_DocBin(tmp_dir, check_data_integrity=False)

    dm2 = DataManager(data_dir)
    dm2.import_data_DocBin((tmp_dir))

    assert dm.spacy_docbin_files == dm2.spacy_docbin_files


def test_check_data_integrity(data_dir):
    dm = DataManager(data_dir)
    dm.check_data_integrity()


def test_docdict_to_lists(data_dir):
    dm = DataManager(data_dir)
    dm.docdict_to_lists()
    ref_sentence = [
        "SEP.01673",
        "Hamburger",
        "Morgenpost",
        ",",
        "16.09.2007",
        ",",
        "S.",
        "46",
        ";",
    ]
    ref_labels = [0, 0, 0, -100, 0, -100, 0, 0, -100]
    assert dm.sentence_list[32] == ref_sentence
    assert dm.label_list[32] == ref_labels


def test_lists_to_df(data_dir):
    dm = DataManager(data_dir)
    dm.docdict_to_lists()
    dm.lists_to_df()
    ref_sentence = ["BERLIN"]
    assert dm.data_in_frame["Sentences"][3] == ref_sentence
    ref_labels = [0]
    assert dm.data_in_frame["Labels"][3] == ref_labels


def test_df_to_dataset(data_dir):
    dm = DataManager(data_dir)
    dm.docdict_to_lists()
    dm.lists_to_df()
    dm.df_to_dataset(split=False)
    ref_sentence = ["BERLIN"]
    assert dm.raw_data_set["Sentences"][3] == ref_sentence
    ref_labels = [0]
    dm.df_to_dataset()
    assert dm.raw_data_set["Labels"][3] == ref_labels
    assert dm.train_test_set["test"]
    assert dm.train_test_set["train"]


def test_push_dataset_to_hub(data_dir, monkeypatch):
    dm = DataManager(data_dir)
    repo_id = "test-data-2"
    dm.docdict_to_lists()
    dm.lists_to_df()
    dm.df_to_dataset(split=False)
    dm.push_dataset_to_hub(repo_id)
    dataset = dm.raw_data_set
    dm.push_dataset_to_hub(repo_id, dataset)
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    with pytest.raises(ValueError):
        dm.push_dataset_to_hub(repo_id)


def test_set_dataset_info(data_dir, get_dataset):
    dm = DataManager(data_dir)
    description = "Something"
    dataset = dm.set_dataset_info(get_dataset["test"], description=description)
    assert dataset.info.description == description
    description = "Something else"
    dataset = dm.set_dataset_info(get_dataset["test"], description=description)
    assert dataset.info.description == description
    version = "0.0.1"
    dataset = dm.set_dataset_info(get_dataset["test"], version=version)
    assert dataset.info.version == version
    license_ = "MIT"
    dataset = dm.set_dataset_info(get_dataset["test"], license_=license_)
    assert dataset.info.license == license_
    citation = "My-DOI"
    dataset = dm.set_dataset_info(get_dataset["test"], citation=citation)
    assert dataset.info.citation == citation
    homepage = "My-homepage"
    dataset = dm.set_dataset_info(get_dataset["test"], homepage=homepage)
    assert dataset.info.homepage == homepage
    dm = DataManager(data_dir)
    dm.docdict_to_lists()
    dm.lists_to_df()
    dm.df_to_dataset(split=False)
    dataset = dm.set_dataset_info(homepage=homepage)
    assert dataset.info.homepage == homepage
