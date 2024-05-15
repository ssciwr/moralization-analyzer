from moralization.transformers_model_manager import TransformersModelManager
from moralization.transformers_model_manager import (
    _import_or_create_metadata,
    _update_model_meta,
)
from moralization.data_manager import DataManager
import pytest
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification
import frontmatter
from huggingface_hub import HfApi


@pytest.fixture
def gen_instance(tmp_path):
    return TransformersModelManager(tmp_path)


@pytest.fixture
def gen_instance_dm(data_dir):
    dm = DataManager(data_dir)
    dm.df_to_dataset(split=True)
    return dm


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


def test_update_model_meta(tmp_path):
    # check with no file present
    _update_model_meta(tmp_path, {"license": "MIT"})
    # create a temp README file
    meta = {
        "language": ["en"],
        "thumbnail": None,
        "tags": ["token classification"],
        "license": "MIT",
        "datasets": ["iulusoy/test-data-3"],
        "metrics": ["seqeval"],
    }
    meta_file = tmp_path / "README.md"
    post = frontmatter.Post(content="# My model", **meta)
    with open(meta_file, "wb") as f:
        frontmatter.dump(post, f)
    _update_model_meta(tmp_path, {"license": "MIT"})
    with open(meta_file) as f:
        meta_changed = frontmatter.load(f)
    assert meta_changed["language"] == ["en"]
    assert meta_changed["license"] == "MIT"
    assert str(meta_changed) == "# My model"


def test_import_or_create_metadata(tmp_path):
    meta = _import_or_create_metadata(tmp_path)
    assert meta["language"] == ["en"]
    assert meta["license"] == "mit"
    meta_file = tmp_path / "README.md"
    assert meta_file.is_file()


def test_init_tokenizer(gen_instance):
    assert gen_instance.tokenizer.is_fast
    with pytest.raises(OSError):
        gen_instance._init_tokenizer(model_name="abcd")
    with pytest.raises(ValueError):
        gen_instance._init_tokenizer(kwargs={"use_fast": False})


def test_tokenize(raw_dataset, gen_instance):
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


def test_map_dataset(train_test_dataset, long_dataset, tmp_path):
    tmm = TransformersModelManager(tmp_path)
    tokenized_dataset = tmm.map_dataset(train_test_dataset)
    assert isinstance(tokenized_dataset["train"], Dataset)
    # try with more than one sentence
    del tmm
    tmm = TransformersModelManager(tmp_path)
    tokenized_dataset = tmm.map_dataset(long_dataset)
    ref_input_ids = [
        [101, 7091, 1734, 1111, 5193, 102],
        [101, 1105, 1199, 1167, 108, 102],
    ]
    ref_labels = [[-100, 0, 2, 1, 1, -100], [-100, 0, 0, 0, 0, -100]]
    assert tokenized_dataset["train"]["input_ids"] == ref_input_ids
    assert tokenized_dataset["train"]["labels"] == ref_labels


def test_init_data_collator(gen_instance):
    assert isinstance(gen_instance.data_collator, DataCollatorForTokenClassification)
    assert gen_instance.data_collator.padding
    assert gen_instance.data_collator.return_tensors == "pt"
    assert not gen_instance.data_collator.pad_to_multiple_of


def test_create_batch(gen_instance, long_dataset):
    tokenized_datasets = gen_instance.map_dataset(long_dataset)
    batch = gen_instance.create_batch(tokenized_datasets)
    ref_labels = [-100, 0, 2, 1, 1, -100]
    assert batch["labels"][0].tolist() == ref_labels
    ref_input_ids = [101, 7091, 1734, 1111, 5193, 102]
    assert batch["input_ids"][0].tolist() == ref_input_ids
    ref_attention_mask = [1, 1, 1, 1, 1, 1]
    assert batch["attention_mask"][0].tolist() == ref_attention_mask


def test_load_evaluation_metric(gen_instance, tmp_path):
    gen_instance._load_evaluation_metric()
    assert gen_instance.metric.module_type == "metric"
    assert gen_instance.metric.name == "seqeval"
    assert gen_instance.label_names == ["0", "M", "M-BEG"]
    gen_instance._load_evaluation_metric(eval_metric="precision")
    assert gen_instance.metric.name == "precision"


def test_load_model(gen_instance):
    gen_instance.label_names = ["A", "B", "C"]
    gen_instance._load_model()
    assert gen_instance.model.name_or_path == "bert-base-cased"
    assert gen_instance.model.num_labels == 3
    gen_instance._load_model(model_name="bert-base-uncased")
    assert gen_instance.model.name_or_path == "bert-base-uncased"
    with pytest.raises(OSError):
        gen_instance._load_model(model_name="ber-base-uncased")


def test_load_dataloader(gen_instance, train_test_dataset):
    tokenized_dataset = gen_instance.map_dataset(train_test_dataset)
    gen_instance._load_dataloader(tokenized_dataset)
    assert gen_instance.train_dataloader.batch_size == 8
    assert gen_instance.eval_dataloader.batch_size == 8
    gen_instance._load_dataloader(tokenized_dataset, batch_size=16)
    assert gen_instance.train_dataloader.batch_size == 16
    assert gen_instance.eval_dataloader.batch_size == 16


def test_load_optimizer(gen_instance):
    gen_instance.label_names = ["A", "B", "C"]
    gen_instance._load_optimizer(learning_rate=1e-3)
    assert gen_instance.optimizer.defaults["lr"] == pytest.approx(1e-3, 1e-4)
    gen_instance._load_optimizer(learning_rate=1e-3, kwargs={"weight_decay": 0.015})
    assert gen_instance.optimizer.defaults["weight_decay"] == pytest.approx(0.015, 1e-3)


def test_load_scheduler(gen_instance, train_test_dataset):
    tokenized_dataset = gen_instance.map_dataset(train_test_dataset)
    gen_instance._load_dataloader(tokenized_dataset)
    gen_instance.label_names = ["A", "B", "C"]
    gen_instance._load_model()
    gen_instance._load_optimizer(learning_rate=2e-5)
    gen_instance._load_scheduler(num_train_epochs=3)
    assert gen_instance.lr_scheduler.base_lrs == [2e-5]
    assert gen_instance.num_training_steps == 3
    # now test the exceptions
    del gen_instance.optimizer
    with pytest.raises(ValueError):
        gen_instance._load_scheduler(num_train_epochs=3)
    del gen_instance.train_dataloader
    with pytest.raises(ValueError):
        gen_instance._load_scheduler(num_train_epochs=3)


def test_train_evaluate(gen_instance, gen_instance_dm):
    model_path = gen_instance._model_path
    token_column_name = "Sentences"
    label_column_name = "Labels"
    num_train_epochs = 1
    learning_rate = 1e-5
    gen_instance.train(
        gen_instance_dm,
        token_column_name,
        label_column_name,
        num_train_epochs,
        learning_rate,
    )
    assert gen_instance.results["overall_precision"] == pytest.approx(0.0, 1e-3)
    assert (model_path / "model.safetensors").is_file()
    assert (model_path / "special_tokens_map.json").is_file()
    assert (model_path / "config.json").is_file()
    evaluate_result = gen_instance.evaluate("Python ist toll.")
    assert evaluate_result[0]["score"]
    # check that column names throw error if not given correctly
    label_column_name = "something"
    with pytest.raises(ValueError):
        gen_instance.train(
            gen_instance_dm,
            token_column_name,
            label_column_name,
            num_train_epochs,
            learning_rate,
        )
    token_column_name = "something"
    with pytest.raises(ValueError):
        gen_instance.train(
            gen_instance_dm,
            token_column_name,
            label_column_name,
            num_train_epochs,
            learning_rate,
        )
    del gen_instance._model_path
    with pytest.raises(ValueError):
        gen_instance.evaluate("Python ist toll.")


def test_publish(gen_instance):
    with pytest.raises(ValueError):
        gen_instance.publish()
    with pytest.raises(ValueError):
        gen_instance.publish(repo_name="temp")
    with pytest.raises(ValueError):
        gen_instance.publish(hf_namespace="user")
    with pytest.raises(RuntimeError):
        gen_instance.publish(repo_name="temp", hf_namespace="iulusoy")
    gen_instance._model_is_trained = True
    # now publish with a new repo name
    # create mock files to publish
    p = gen_instance.model_path / "README.md"
    p.write_text("something")
    p = gen_instance.model_path / "config.json"
    p.write_text("something")
    p = gen_instance.model_path / "pytorch_model.bin"
    p.write_text("something")
    url = gen_instance.publish(
        repo_name="temp", hf_namespace="iulusoy", create_new_repo=True
    )
    assert url == "https://huggingface.co/iulusoy/temp/tree/main/"
    # delete the repo again
    api = HfApi()
    # check that existing folder is not overwritten
    with pytest.raises(ValueError):
        gen_instance.publish(
            repo_name="temp", hf_namespace="iulusoy", create_new_repo=True
        )
    # now push to existing repo
    commit = gen_instance.publish(repo_name="temp", hf_namespace="iulusoy")
    assert commit.commit_message == "Upload BertForTokenClassification"
    api.delete_repo(repo_id="iulusoy/temp")


def test_publish_missing_metadata(gen_instance):
    # test for missing metadata
    gen_instance.metadata["license"] = None
    with pytest.raises(RuntimeError):
        gen_instance.publish(repo_name="temp", hf_namespace="iulusoy")
