from moralization.spacy_model_manager import SpacyModelManager
from moralization.data_manager import DataManager
import spacy
import pytest
import spacy_huggingface_hub
import huggingface_hub
from typing import Any
from pathlib import Path


@pytest.fixture
def data_dir_large(data_dir):
    data_dir_large = data_dir / "large_input_data"
    return data_dir_large


def test_spacy_model_manager_train_new_model(tmp_path, data_dir_large):
    # non-existent model_path: new model created with default config/meta
    # we need a larger data set than test data, otherwise it is not guaranteed that
    # there is an annotation in the test data
    model_path = tmp_path / "idontexist"
    assert not model_path.is_dir()
    model = SpacyModelManager(model_path)
    assert str(model_path) in str(model)
    assert model_path.is_dir()
    assert (model_path / "config.cfg").is_file()
    assert (model_path / "meta.json").is_file()
    # model is not yet trained
    assert not (model_path / "model-best").is_dir()
    assert not (model_path / "model-last").is_dir()
    data_manager = DataManager(data_dir_large)
    # train model
    model.train(
        data_manager, overrides={"training.max_epochs": 5}, check_data_integrity=False
    )
    # evaluate trained model
    evaluation = model.evaluate(data_manager)
    assert "Moralisierung explizit" in evaluation["spans_task1_per_type"]
    # create instance with pre-existing config
    path_to_config = model_path / "config.cfg"
    # save model to other path
    model_path = tmp_path / "idontexist2"
    _ = SpacyModelManager(model_path, base_config_file=path_to_config.as_posix())
    # try with config not found
    with pytest.raises(ValueError):
        SpacyModelManager(
            model_path, base_config_file="./config", overwrite_existing_files=True
        )


def test_spacy_model_manager_train_new_model_task(tmp_path, data_dir):
    # non-existent model_path: new model created with default config/meta
    model_path = tmp_path / "idontexist"
    assert not model_path.is_dir()
    model = SpacyModelManager(model_path, language="en", task="task2")
    assert str(model_path) in str(model)
    assert model_path.is_dir()
    assert (model_path / "config.cfg").is_file()
    assert (model_path / "meta.json").is_file()
    # model is not yet trained
    assert not (model_path / "model-best").is_dir()
    assert not (model_path / "model-last").is_dir()
    data_manager = DataManager(data_dir, task="task2")
    # train model
    model.train(
        data_manager, overrides={"training.max_epochs": 5}, check_data_integrity=False
    )
    # evaluate trained model
    evaluation = model.evaluate(data_manager)
    assert "Fairness" in evaluation["spans_task2_per_type"]
    # try with wrong task in model train
    model = SpacyModelManager(model_path, language="en", task="task1")
    with pytest.raises(ValueError):
        model.train(
            data_manager,
            overrides={"training.max_epochs": 5},
            check_data_integrity=False,
        )


def test_spacy_model_manager_existing_invalid_model_path(tmp_path):
    with pytest.raises(OSError) as e:
        SpacyModelManager(tmp_path)
    assert "already exists" in str(e.value)


def test_spacy_model_manager_existing_invalid_model_path_overwrite(tmp_path):
    subfolder = tmp_path / "hi"
    subfolder.mkdir()
    assert subfolder.is_dir()
    model = SpacyModelManager(tmp_path, overwrite_existing_files=True)
    assert str(tmp_path) in str(model)
    assert not subfolder.is_dir()
    assert (tmp_path / "config.cfg").is_file()
    assert (tmp_path / "meta.json").is_file()


def test_spacy_model_manager_valid_model_path(spacy_model_path):
    model = SpacyModelManager(spacy_model_path)
    assert str(spacy_model_path) in str(model)


def test_spacy_model_manager_test(spacy_model_path, monkeypatch):
    args = {}

    def store_doc_dict(doc_dict, **kwargs: Any) -> None:
        args["doc_dict"] = doc_dict

    # monkey patch moralization.plot.visualize_data to just store doc_dict argument
    monkeypatch.setattr(
        "moralization.spacy_model_manager.return_displacy_visualization", store_doc_dict
    )

    model = SpacyModelManager(spacy_model_path)
    test_string = "Das ist ein gutes Beispiel"
    model.test(test_string)
    assert len(args["doc_dict"]) == 1
    for doc in args["doc_dict"].values():
        assert doc.text == test_string


def test_spacy_model_manager_modify_metadata(spacy_model_path):
    model = SpacyModelManager(spacy_model_path)
    # update metadata values and save model
    keys = ["name", "version", "description", "author", "email", "url", "license"]
    for key in keys:
        model.metadata[key] = f"{key}"
    model.save()
    for key in keys:
        assert model.metadata[key] == f"{key}"
    # re-load model
    model = SpacyModelManager(spacy_model_path)
    for key in keys:
        assert model.metadata[key] == f"{key}"
    # load trained models directly in spacy and check their meta have also been updated
    for folder in ["model-best", "model-last"]:
        nlp = spacy.load(spacy_model_path / folder)
        for key in keys:
            assert nlp.meta[key] == f"{key}"


def test_spacy_model_manager_modify_metadata_fixable_invalid_names(spacy_model_path):
    model = SpacyModelManager(spacy_model_path)
    for invalid_name, valid_name in [("!hm & __OK?,...", "hm_ok"), ("Im - S", "im_s")]:
        model.metadata["name"] = invalid_name
        assert model.metadata["name"] == invalid_name
        # name is made valid on call to save()
        model.save()
        assert model.metadata["name"] == valid_name
        for folder in ["model-best", "model-last"]:
            nlp = spacy.load(spacy_model_path / folder)
            assert nlp.meta["name"] == valid_name


def test_spacy_model_manager_modify_metadata_unfixable_invalid_names(spacy_model_path):
    model = SpacyModelManager(spacy_model_path)
    for unfixable_invalid_name in ["", "_", "ü"]:
        model.metadata["name"] = unfixable_invalid_name
        with pytest.raises(ValueError) as e:
            model.save()
        assert "invalid" in str(e.value).lower()


def test_spacy_model_manager_publish_untrained(tmp_path):
    model = SpacyModelManager(tmp_path / "my_model")
    with pytest.raises(RuntimeError) as e:
        model.publish()
    assert "trained" in str(e.value).lower()


def test_spacy_model_manager_publish_invalid_token_env(spacy_model_path, monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "invalid")
    model = SpacyModelManager(spacy_model_path)
    with pytest.raises(ValueError) as e:
        model.publish()
    assert "token" in str(e.value).lower()


def test_spacy_model_manager_publish_invalid_token_arg(spacy_model_path):
    model = SpacyModelManager(spacy_model_path)
    with pytest.raises(ValueError) as e:
        model.publish(hugging_face_token="invalid")
    assert "token" in str(e.value).lower()


def test_spacy_model_manager_publish_missing_metadata(spacy_model_path):
    model = SpacyModelManager(spacy_model_path)
    model.metadata["author"] = ""
    with pytest.raises(RuntimeError) as e:
        model.publish()
    assert "author" in str(e.value).lower()


def test_spacy_model_manager_publish_mock_push(spacy_model_path, monkeypatch, tmp_path):
    def mock_spacy_huggingface_hub_push(whl_path: Path):
        whl_path.rename(tmp_path / whl_path.name)
        return {}

    # monkey patch spacy_huggingface_hub.push() to just move the supplied wheel to a temporary path
    monkeypatch.setattr(spacy_huggingface_hub, "push", mock_spacy_huggingface_hub_push)

    def do_nothing(*args: Any, **kwargs: Any) -> None:
        return

    # monkey patch huggingface_hub.login() to do nothing
    monkeypatch.setattr(huggingface_hub, "login", do_nothing)

    model = SpacyModelManager(spacy_model_path)
    # set name and version - these determine the name of the compiled wheel
    model.metadata["name"] = "my_new_pipeline"
    model.metadata["version"] = "1.2.3"
    model.publish(hugging_face_token="abc123")
    wheel_path = tmp_path / "de_my_new_pipeline-1.2.3-py3-none-any.whl"
    assert wheel_path.is_file()
