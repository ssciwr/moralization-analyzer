from moralization.transformers_data_handler import TransformersDataHandler
import pytest
from moralization import DataManager

EXAMPLE_NAME = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"


@pytest.fixture
def doc_dict(data_dir):
    data_manager = DataManager(data_dir)
    return data_manager.doc_dict


@pytest.fixture
def gen_instance():
    return TransformersDataHandler()


def test_get_data_lists(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict, example_name=EXAMPLE_NAME)
    assert gen_instance.label_list[0] == [0]
    assert gen_instance.sentence_list[2][0] == "JUL.02661"
    assert gen_instance.token_list[1][0].text == "T07"


def test_generate_labels(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict, example_name=EXAMPLE_NAME)
    gen_instance.generate_labels(doc_dict=doc_dict, example_name=EXAMPLE_NAME)
    assert gen_instance.labels[10] == 0
    assert gen_instance.labels[624] == 2
    assert gen_instance.labels[625] == 1
    assert gen_instance.labels[671] == 1
    assert gen_instance.labels[672] == 0


def test_structure_labels(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict, example_name=EXAMPLE_NAME)
    gen_instance.generate_labels(doc_dict=doc_dict, example_name=EXAMPLE_NAME)
    gen_instance.structure_labels()
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
    assert gen_instance.sentence_list[7] == ref_sentence
    assert gen_instance.label_list[7] == ref_labels
    assert gen_instance.label_list[44] == ref_labels2