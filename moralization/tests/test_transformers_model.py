from moralization.transformers_model import TransformersSetup
import pytest


def test_TransformersSetup(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_doc_dict(data_dir)
    test_filenames = list(test_obj.doc_dict.keys())
    reference_filenames = [
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    ]
    assert sorted(test_filenames) == sorted(reference_filenames)


def test_get_data_lists(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_doc_dict(data_dir)
    example_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    test_obj.get_data_lists(example_name=example_name)
    assert test_obj.label_list[0] == [0]
    assert test_obj.sentence_list[2][0] == "JUL.02661"
    assert test_obj.token_list[1][0].text == "T07"


def test_generate_labels(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_doc_dict(data_dir)
    example_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    test_obj.get_data_lists(example_name=example_name)
    test_obj.generate_labels(example_name=example_name)
    assert test_obj.labels[10] == 0
    assert test_obj.labels[624] == 2
    assert test_obj.labels[625] == 1
    assert test_obj.labels[671] == 1
    assert test_obj.labels[672] == 0


def test_structure_labels(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_doc_dict(data_dir)
    example_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    test_obj.get_data_lists(example_name=example_name)
    test_obj.generate_labels(example_name=example_name)
    test_obj.structure_labels()
    ref_sentence = ["Ich", "zitiere", "mal", "einen", "Kollegen", ":"]
    ref_labels = [0, 0, 0, 0, 0, -100]
    ref_labels2 = [
        2,
        1,
        1,
        1,
        1,
        1,
        -100,
        1,
        -100,
        -100,
        1,
        1,
        1,
        1,
        1,
        1,
        -100,
        1,
        1,
        1,
        -100,
        1,
        1,
        1,
        1,
        -100,
    ]
    assert test_obj.sentence_list[7] == ref_sentence
    assert test_obj.label_list[7] == ref_labels
    assert test_obj.label_list[44] == ref_labels2


def test_lists_to_df(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_doc_dict(data_dir)
    example_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    test_obj.get_data_lists(example_name=example_name)
    test_obj.generate_labels(example_name=example_name)
    test_obj.structure_labels()
    test_obj.lists_to_df()
    assert test_obj.train_test_set["train"].shape == (60, 2)


def test_init_model():
    test_obj = TransformersSetup()
    test_obj.init_model()
    assert test_obj.tokenizer.is_fast
    with pytest.raises(ValueError):
        test_obj.init_model(model_name="Testing")
