from moralization.transformers_data_handler import TransformersDataHandler
import pytest
from moralization import DataManager
from datasets import load_dataset

EXAMPLE_NAME = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"


@pytest.fixture
def doc_dict(data_dir):
    data_manager = DataManager(data_dir)
    return data_manager.doc_dict


@pytest.fixture
def gen_instance():
    return TransformersDataHandler()


@pytest.fixture(scope="module")
def raw_dataset():
    return load_dataset("iulusoy/test-data")


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


def test_init_tokenizer(gen_instance):
    gen_instance.init_tokenizer()
    assert gen_instance.tokenizer.is_fast
    with pytest.raises(OSError):
        gen_instance.init_tokenizer(model_name="abcd")
    with pytest.raises(ValueError):
        gen_instance.init_tokenizer(kwargs={"use_fast": False})


def test_tokenize(raw_dataset):
    tdh = TransformersDataHandler()
    tdh.init_tokenizer()
    tdh.tokenize(raw_dataset["test"]["word"])
    assert tdh.inputs["input_ids"][0] == 101
    assert tdh.inputs["input_ids"][2] == 1821
    assert tdh.inputs["input_ids"][-1] == 102
    assert len(set(tdh.inputs["attention_mask"])) == 1
    new_tokens = tdh.inputs.tokens()
    ref_tokens = [
        "[CLS]",
        "I",
        "am",
        "working",
        "on",
        "the",
        "#",
        "#",
        "#",
        "moral",
        "##ization",
        "project",
        ".",
        "[SEP]",
    ]
    assert new_tokens == ref_tokens


def test_align_labels_with_tokens(raw_dataset):
    tdh = TransformersDataHandler()
    tdh.init_tokenizer()
    tdh.tokenize(raw_dataset["test"]["word"])
    new_labels = tdh._align_labels_with_tokens(
        raw_dataset["test"]["label"], tdh.inputs.word_ids()
    )
    assert new_labels[0] == -100
    assert new_labels[2] == 0
    assert new_labels[3] == 2
    assert new_labels[4] == 1
    assert new_labels[10] == 1
    assert new_labels[-1] == -100
    tdh.tokenize(["moralization"])
    new_labels = tdh._align_labels_with_tokens([2, 2], tdh.inputs.word_ids())
    ref_labels = [-100, 2, 1, -100]
    assert new_labels == ref_labels


def test_add_labels_to_inputs(raw_dataset):
    tdh = TransformersDataHandler()
    tdh.init_tokenizer()
    tdh.tokenize(raw_dataset["test"]["word"])
    # test if list of strings is working
    tdh.add_labels_to_inputs(labels=raw_dataset["test"]["label"])
    ref_labels = [[-100, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -100]]
    assert tdh.inputs["labels"] == ref_labels
    # test if list of list is working
    tdh.inputs = None
    tdh.tokenize(raw_dataset["test"]["word"])
    labels_list = [[0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0]]
    tdh.add_labels_to_inputs(labels=labels_list)
    assert tdh.inputs["labels"] == ref_labels
    # test if labels from self is working
    tdh.inputs = None
    tdh.tokenize(raw_dataset["test"]["word"])
    tdh.label_list = labels_list
    tdh.add_labels_to_inputs()
    assert tdh.inputs["labels"] == ref_labels
