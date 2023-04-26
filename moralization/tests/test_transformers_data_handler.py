from moralization.transformers_data_handler import TransformersDataHandler
import pytest
from moralization import DataManager
from datasets import load_dataset, Dataset, DatasetDict

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


@pytest.fixture(scope="module")
def train_test_dataset():
    return load_dataset("iulusoy/test-data", split="test").train_test_split(
        test_size=0.1
    )


@pytest.fixture
def long_dataset():
    datadict = {
        "word": [["random", "words", "for", "testing"], ["and", "some", "more", "#"]],
        "label": [[0, 2, 1, 1], [0, 0, 0, 0]],
    }
    ds = Dataset.from_dict(datadict, split="train")
    ds_dict = DatasetDict({"train": ds})
    return ds_dict


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
    assert tdh.inputs["input_ids"][0][0] == 101
    assert tdh.inputs["input_ids"][0][2] == 1821
    assert tdh.inputs["input_ids"][0][-1] == 102
    assert len(set(tdh.inputs["attention_mask"][0])) == 1
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
    # try with no wordlist
    tdh.token_list = raw_dataset["test"]["word"]
    tdh.tokenize()
    assert new_tokens == ref_tokens


def test_check_is_nested():
    tdh = TransformersDataHandler()
    sample = ["something"]
    ref_result = [["something"]]
    sample = tdh._check_is_nested(sample)
    assert sample == ref_result
    sample = [["something"]]
    sample = tdh._check_is_nested(sample)
    assert sample == ref_result
    sample = [["something"], "something"]
    ref_result = [["something"], ["something"]]
    sample = tdh._check_is_nested(sample)
    assert sample == ref_result


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


def test_map_dataset(train_test_dataset, long_dataset):
    tdh = TransformersDataHandler()
    tdh.init_tokenizer()
    tokenized_dataset = tdh.map_dataset(train_test_dataset)
    assert isinstance(tokenized_dataset["train"], Dataset)
    # try with more than one sentence
    del tdh
    tdh = TransformersDataHandler()
    tdh.init_tokenizer()
    tokenized_dataset = tdh.map_dataset(long_dataset)
    ref_input_ids = [
        [101, 7091, 1734, 1111, 5193, 102],
        [101, 1105, 1199, 1167, 108, 102],
    ]
    ref_labels = [[-100, 0, 2, 1, 1, -100], [-100, 0, 0, 0, 0, -100]]
    assert tokenized_dataset["train"]["input_ids"] == ref_input_ids
    assert tokenized_dataset["train"]["labels"] == ref_labels
