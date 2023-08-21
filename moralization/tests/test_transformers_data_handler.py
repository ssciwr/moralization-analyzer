from moralization.transformers_data_handler import TransformersDataHandler
import pytest
from moralization import DataManager


@pytest.fixture
def doc_dict(data_dir):
    data_manager = DataManager(data_dir)
    return data_manager.doc_dict


@pytest.fixture
def gen_instance():
    return TransformersDataHandler()


def test_get_data_lists(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict)
    # relying on that the dict is sorted by insertion for the wr
    print(gen_instance.label_list[0])
    print(gen_instance.sentence_list[2][0])
    print(gen_instance.token_list[1][0].text)
    assert gen_instance.label_list[0] == [0]
    assert gen_instance.sentence_list[2][0] == "ALG"
    assert gen_instance.token_list[1][0].text == "HMP05"


def test_generate_labels(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict)
    gen_instance.generate_labels(doc_dict=doc_dict)
    assert gen_instance.labels[510] == 0
    assert gen_instance.labels[1124] == 2
    assert gen_instance.labels[1125] == 1
    assert gen_instance.labels[1171] == 1
    assert gen_instance.labels[1172] == 0
    assert len(gen_instance.labels) == 1346


def test_generate_spans_task1(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict)
    gen_instance.generate_labels(doc_dict=doc_dict)
    gen_instance.generate_spans(doc_dict=doc_dict)
    assert gen_instance.span_list[0] == []
    assert gen_instance.span_list[4] == [(23, 44, "Moralisierung explizit")]
    gen_instance.get_data_lists(doc_dict=doc_dict)
    gen_instance.generate_labels(doc_dict=doc_dict, task="task2")
    gen_instance.generate_spans(doc_dict=doc_dict, task="task2")
    assert gen_instance.span_list[3] == []
    assert gen_instance.span_list[4] == [(33, 37, "Care")]
    assert gen_instance.span_list[8] == [
        (109, 111, "Care"),
        (106, 107, "Oppression"),
        (112, 114, "Fairness"),
    ]


def test_structure_labels(doc_dict, gen_instance):
    gen_instance.get_data_lists(doc_dict=doc_dict)
    gen_instance.generate_labels(doc_dict=doc_dict)
    gen_instance.generate_spans(doc_dict=doc_dict)
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
    assert gen_instance.sentence_list[43] == ref_sentence
    assert gen_instance.label_list[43] == ref_labels
    assert gen_instance.label_list[80] == ref_labels2
