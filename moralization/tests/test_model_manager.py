from moralization.model_manager import ModelManager
import spacy
import pytest
import spacy_huggingface_hub
import huggingface_hub
from typing import Any
from pathlib import Path


def test_model_manager_valid_path(model_path):
    model = ModelManager(model_path)
    assert model.spacy_model is not None
    assert model.spacy_model.lang == "de"
    assert model.spacy_model.path == model_path


def test_model_manager_modify_metadata(model_path):
    model = ModelManager(model_path)
    # update metadata values and save model
    keys = ["name", "version", "description", "author", "email", "url", "license"]
    for key in keys:
        model.metadata[key] = f"{key}"
    model.save()
    for key in keys:
        assert model.metadata[key] == f"{key}"
    # re-load model
    model.load(model_path)
    for key in keys:
        assert model.metadata[key] == f"{key}"
    # load model directly in spacy and check its meta has also been updated
    nlp = spacy.load(model_path)
    for key in keys:
        assert nlp.meta[key] == f"{key}"


def test_model_manager_modify_metadata_fixable_invalid_names(model_path):
    model = ModelManager(model_path)
    for invalid_name, valid_name in [("!hm & __OK?,...", "hm_ok"), ("Im - S", "im_s")]:
        model.metadata["name"] = invalid_name
        assert model.metadata["name"] == invalid_name
        # name is made valid on call to save()
        model.save()
        assert model.metadata["name"] == valid_name
        nlp = spacy.load(model_path)
        assert nlp.meta["name"] == valid_name


def test_model_manager_modify_metadata_unfixable_invalid_names(model_path):
    model = ModelManager(model_path)
    for unfixable_invalid_name in ["", "_", "ü"]:
        model.metadata["name"] = unfixable_invalid_name
        with pytest.raises(ValueError) as e:
            model.save()
        assert "invalid" in str(e.value).lower()


def test_model_manager_publish_no_token(model_path, monkeypatch):
    monkeypatch.delenv("HUGGING_FACE_TOKEN", raising=False)
    model = ModelManager(model_path)
    with pytest.raises(ValueError) as e:
        model.publish()
    assert "token" in str(e.value).lower()


def test_model_manager_publish_invalid_token_env(model_path, monkeypatch):
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "invalid")
    model = ModelManager(model_path)
    with pytest.raises(ValueError) as e:
        model.publish()
    assert "token" in str(e.value).lower()


def test_model_manager_publish_invalid_token_arg(model_path):
    model = ModelManager(model_path)
    with pytest.raises(ValueError) as e:
        model.publish(hugging_face_token="invalid")
    assert "token" in str(e.value).lower()


def test_model_manager_publish_mock_push(model_path: Path, monkeypatch, tmp_path):
    def mock_spacy_huggingface_hub_push(whl_path: Path):
        whl_path.rename(tmp_path / whl_path.name)
        return {}

    # monkey patch spacy_huggingface_hub.push() to just move the supplied wheel to a temporary path
    monkeypatch.setattr(spacy_huggingface_hub, "push", mock_spacy_huggingface_hub_push)

    def do_nothing(*args: Any, **kwargs: Any) -> None:
        return

    # monkey patch huggingface_hub.login() to do nothing
    monkeypatch.setattr(huggingface_hub, "login", do_nothing)

    model = ModelManager(model_path)
    # set name and version - these determine the name of the compiled wheel
    model.metadata["name"] = "my_new_pipeline"
    model.metadata["version"] = "1.2.3"
    model.publish(hugging_face_token="abc123")
    wheel_path = tmp_path / "de_my_new_pipeline-1.2.3-py3-none-any.whl"
    assert wheel_path.is_file()
