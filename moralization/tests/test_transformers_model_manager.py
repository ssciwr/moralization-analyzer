from moralization.transformers_model_manager import TransformersModelManager
import pytest
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification


@pytest.fixture
def gen_instance():
    return TransformersModelManager()


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


def test_init_tokenizer(gen_instance):
    gen_instance.init_tokenizer()
    assert gen_instance.tokenizer.is_fast
    with pytest.raises(OSError):
        gen_instance.init_tokenizer(model_name="abcd")
    with pytest.raises(ValueError):
        gen_instance.init_tokenizer(kwargs={"use_fast": False})


def test_tokenize(raw_dataset, gen_instance):
    gen_instance.init_tokenizer()
    gen_instance.tokenize(raw_dataset["test"]["word"])
    assert gen_instance.inputs["input_ids"][0][0] == 101
    assert gen_instance.inputs["input_ids"][0][2] == 1821
    assert gen_instance.inputs["input_ids"][0][-1] == 102
    assert len(set(gen_instance.inputs["attention_mask"][0])) == 1
    new_tokens = gen_instance.inputs.tokens()
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
    gen_instance.token_list = raw_dataset["test"]["word"]
    gen_instance.tokenize()
    assert new_tokens == ref_tokens


def test_check_is_nested(gen_instance):
    sample = ["something"]
    ref_result = [["something"]]
    sample = gen_instance._check_is_nested(sample)
    assert sample == ref_result
    sample = [["something"]]
    sample = gen_instance._check_is_nested(sample)
    assert sample == ref_result
    sample = [["something"], "something"]
    ref_result = [["something"], ["something"]]
    sample = gen_instance._check_is_nested(sample)
    assert sample == ref_result


def test_align_labels_with_tokens(raw_dataset, gen_instance):
    gen_instance.init_tokenizer()
    gen_instance.tokenize(raw_dataset["test"]["word"])
    new_labels = gen_instance._align_labels_with_tokens(
        raw_dataset["test"]["label"], gen_instance.inputs.word_ids()
    )
    assert new_labels[0] == -100
    assert new_labels[2] == 0
    assert new_labels[3] == 2
    assert new_labels[4] == 1
    assert new_labels[10] == 1
    assert new_labels[-1] == -100
    gen_instance.tokenize(["moralization"])
    new_labels = gen_instance._align_labels_with_tokens(
        [2, 2], gen_instance.inputs.word_ids()
    )
    ref_labels = [-100, 2, 1, -100]
    assert new_labels == ref_labels


def test_add_labels_to_inputs(raw_dataset, gen_instance):
    gen_instance.init_tokenizer()
    gen_instance.tokenize(raw_dataset["test"]["word"])
    # test if list of strings is working
    gen_instance.add_labels_to_inputs(labels=raw_dataset["test"]["label"])
    ref_labels = [[-100, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -100]]
    assert gen_instance.inputs["labels"] == ref_labels
    # test if list of list is working
    gen_instance.inputs = None
    gen_instance.tokenize(raw_dataset["test"]["word"])
    labels_list = [[0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0]]
    gen_instance.add_labels_to_inputs(labels=labels_list)
    assert gen_instance.inputs["labels"] == ref_labels
    # test if labels from self is working
    gen_instance.inputs = None
    gen_instance.tokenize(raw_dataset["test"]["word"])
    gen_instance.label_list = labels_list
    gen_instance.add_labels_to_inputs()
    assert gen_instance.inputs["labels"] == ref_labels


def test_map_dataset(train_test_dataset, long_dataset):
    tmm = TransformersModelManager()
    tmm.init_tokenizer()
    tokenized_dataset = tmm.map_dataset(train_test_dataset)
    assert isinstance(tokenized_dataset["train"], Dataset)
    # try with more than one sentence
    del tmm
    tmm = TransformersModelManager()
    tmm.init_tokenizer()
    tokenized_dataset = tmm.map_dataset(long_dataset)
    ref_input_ids = [
        [101, 7091, 1734, 1111, 5193, 102],
        [101, 1105, 1199, 1167, 108, 102],
    ]
    ref_labels = [[-100, 0, 2, 1, 1, -100], [-100, 0, 0, 0, 0, -100]]
    assert tokenized_dataset["train"]["input_ids"] == ref_input_ids
    assert tokenized_dataset["train"]["labels"] == ref_labels


def test_init_data_collator(gen_instance):
    gen_instance.init_tokenizer()
    gen_instance.init_data_collator()
    print(type(gen_instance.data_collator))
    assert type(gen_instance.data_collator) == DataCollatorForTokenClassification
    assert gen_instance.data_collator.padding
    assert gen_instance.data_collator.return_tensors == "pt"
    assert not gen_instance.data_collator.pad_to_multiple_of


def test_create_batch(gen_instance, long_dataset):
    gen_instance.init_tokenizer()
    tokenized_datasets = gen_instance.map_dataset(long_dataset)
    gen_instance.init_data_collator()
    batch = gen_instance.create_batch(tokenized_datasets)
    ref_labels = [-100, 0, 2, 1, 1, -100]
    assert batch["labels"][0].tolist() == ref_labels
    ref_input_ids = [101, 7091, 1734, 1111, 5193, 102]
    assert batch["input_ids"][0].tolist() == ref_input_ids
    ref_attention_mask = [1, 1, 1, 1, 1, 1]
    assert batch["attention_mask"][0].tolist() == ref_attention_mask
