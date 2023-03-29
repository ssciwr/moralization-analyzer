import huggingface_hub
import spacy_huggingface_hub
import os
import json
import spacy
from pathlib import Path
from spacy.cli.init_config import save_config as spacy_cli_save_config
from typing import Union, Optional, List, Dict, Any
import tempfile
import re
import logging
from moralization.data_manager import DataManager
from moralization.plot import visualize_data
import shutil


def _construct_wheel_path(model_path: Path, meta: Dict[str, Any]) -> Path:
    full_name = f"{meta['lang']}_{meta['name']}-{meta['version']}"
    return model_path / full_name / "dist" / f"{full_name}-py3-none-any.whl"


def _make_valid_package_name(name: str) -> str:
    """Attempt to make name valid, throw exception if we fail

    see https://packaging.python.org/en/latest/specifications/name-normalization
    """
    valid_name = re.sub(r"[-_.,<>!@#$%^&*()+ /?]+", "_", name).lower().strip("_")
    if name != valid_name:
        logging.warning(
            f"'{name}' not a valid package name, using '{valid_name}' instead"
        )
    if (
        re.match("^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", valid_name, re.IGNORECASE)
        is None
    ):
        raise ValueError(
            "Invalid package name: Can only contain ASCII letters, numbers and underscore."
        )
    return valid_name


def _update_spacy_model_meta(model_path: Path, metadata: Dict):
    """
    Update matching keys in the spacy meta.json file with values from the supplied metadata dict
    """
    meta_file = model_path / "meta.json"
    if not meta_file.is_file():
        return
    with open(meta_file) as f:
        meta = json.load(f)
        for k, v in metadata.items():
            if k in meta:
                meta[k] = v
    with open(meta_file, "w") as f:
        json.dump(meta, f)


def _import_or_create_metadata(model_path: Path) -> Dict[str, Any]:
    meta_file = model_path / "meta.json"
    default_metadata = {
        "name": "pipeline",
        "version": "0.0.0",
        "description": "",
        "author": "",
        "email": "",
        "url": "",
        "license": "",
    }
    if not meta_file.is_file():
        with open(meta_file, "w") as f:
            json.dump(default_metadata, f)
    with open(meta_file) as f:
        return json.load(f)


def _create_model(
    model_path: Union[str, Path],
    config_file: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
):
    """
    Create a new model from the supplied config file and DataManager.

    If no config file is provided a default german language spancat one is created.
    """
    if model_path.is_dir():
        if overwrite:
            shutil.rmtree(model_path)
        else:
            raise IOError(
                f"Cannot create new model: folder '{model_path}' already exists."
                f"To overwrite add `overwrite=True`."
            )
    Path(model_path).mkdir(parents=True)
    if config_file is None:
        config = spacy.cli.init_config(lang="de", pipeline=["spancat"])
        spacy_cli_save_config(config, model_path / "config.cfg", silent=True)
    else:
        spacy.cli.fill_config(model_path / "config.cfg", Path(config_file))


class SpacyModelManager:
    """
    Create, import, modify, train and publish spacy models.

    Models can be trained on data from a DataManager, and published to hugging face.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        base_config_file: Optional[Union[str, Path]] = None,
        overwrite_existing_files: bool = False,
    ):
        """
        Imports an existing model from the `model_path` folder if found.

        If the `model_path` folder does not exist, or if `base_config_file`
        is supplied, or if `overwrite_existing_files` is True,
        creates a new model in the `model_path` folder.

        Resulting folder structure inside `model_path`:

        - /config.cfg: model config file
        - /meta.json: user-editable metadata (also exported to trained models)

        If the model has been trained the folder will also contain:

        - /data/train.spacy: training dataset generated by DataManager
        - /data/dev.spacy: testing dataset generated by DataManager
        - /model-best/config.cfg: best trained model
        - /model-last/config.cfg: last trained model

        Args:
            model_path (str or Path): Folder where the model is (or will be) stored
            base_config_file (str or Path, optional): If supplied this base config will be used to create a new model
            overwrite_existing_files (bool): If true any existing files in `model_path` are removed
        """
        self._model_path = Path(model_path)
        self._best_model_path = self._model_path / "model-best"
        self._last_model_path = self._model_path / "model-last"
        existing_model = (
            self._model_path.is_dir() and (self._model_path / "config.cfg").is_file()
        )
        if base_config_file or overwrite_existing_files or not existing_model:
            _create_model(self._model_path, base_config_file, overwrite_existing_files)
        self.metadata = _import_or_create_metadata(self._model_path)

    def __repr__(self) -> str:
        name = f"{self.metadata['name']}-{self.metadata['version']}"
        path = f"{self._model_path.resolve()}"
        return f"SpacyModelManager('{path}' [{name}])"

    def train(
        self,
        data_manager: DataManager,
        use_gpu: int = -1,
        overrides: Optional[Dict] = None,
        check_data_integrity=True,
    ):
        """Train the model on the data contained in `data_manager`.

        Args:
            data_manager (DataManager): the DataManager that contains the training data
            use_gpu (int): The index of the GPU to use (default: -1 which means no GPU)
            overrides (dict): An optional dictionary of parameters to override in the model config
            check_data_integrity (bool): Wether or not to test the data integrity.
        """
        self.save()
        if overrides is None:
            overrides = {}
        # use data from data_manager for training
        data_train, data_dev = self._get_data_manager_docbin_files(
            data_manager, check_data_integrity=check_data_integrity
        )
        overrides["paths.train"] = str(data_train)
        overrides["paths.dev"] = str(data_dev)
        spacy.cli.train.train(
            self._model_path / "config.cfg",
            self._model_path,
            use_gpu=use_gpu,
            overrides=overrides,
        )

    def evaluate(self, data_manager: DataManager) -> Dict[str, Any]:
        """Evaluate the model against the test dataset in `data_manager`"""
        self._check_model_is_trained_before_it_can_be("evaluated")
        _, data_dev = self._get_data_manager_docbin_files(data_manager)
        return spacy.cli.evaluate(str(self._best_model_path), data_dev)

    def save(self):
        """Save any changes made to the model metadata."""
        self.metadata["name"] = _make_valid_package_name(self.metadata.get("name"))
        with open(self._model_path / "meta.json", "w") as f:
            json.dump(self.metadata, f)
        for model_path in [self._best_model_path, self._last_model_path]:
            _update_spacy_model_meta(model_path, self.metadata)

    def test(self, test_string: str, style: str = "span"):
        """Test the model output with a test string"""
        self._check_model_is_trained_before_it_can_be("tested")
        nlp = spacy.load(self._best_model_path)
        doc_dict = {"test_doc": nlp(test_string)}
        return visualize_data(doc_dict, style=style)

    def publish(self, hugging_face_token: Optional[str] = None) -> Dict[str, str]:
        """Publish the model to Hugging Face.

        This requires a User Access Token from https://huggingface.co/

        The token can either be passed via the `hugging_face_token` argument,
        or it can be set via the `HUGGING_FACE_TOKEN` environment variable.

        Args:
            hugging_face_token (str, optional): Hugging Face User Access Token
        Returns:
            dict: URLs of the published model and the pip-installable wheel
        """
        self._check_model_is_trained_before_it_can_be("published")
        self.save()
        if hugging_face_token is None:
            hugging_face_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hugging_face_token is None:
            raise ValueError(
                "API TOKEN required: pass as string or set the HUGGING_FACE_TOKEN environment variable."
            )
        huggingface_hub.login(token=hugging_face_token)
        with tempfile.TemporaryDirectory() as tmpdir:
            # convert model to a python package including binary wheel
            package_path = Path(tmpdir)
            spacy.cli.package(self._best_model_path, package_path, create_wheel=True)
            # construct path to binary wheel
            nlp = spacy.load(self._best_model_path)
            wheel_path = _construct_wheel_path(package_path, nlp.meta)
            # push the package to hugging face
            return spacy_huggingface_hub.push(wheel_path)

    def _check_model_is_trained_before_it_can_be(self, action: str = "used"):
        if not self._best_model_path.is_dir():
            raise RuntimeError(f"Model must be trained before it can be {action}.")

    def _get_data_manager_docbin_files(
        self, data_manager: DataManager, check_data_integrity=True
    ) -> List[Path]:
        """
        Returns `[train_data_path, dev_data_path]` from data_manager.

        If the supplied DataManager has no docbin files we first export them to `model_path/data`.
        """
        data_files = data_manager.spacy_docbin_files
        data_files_exist = data_files is not None and all(
            [data_file.is_file() for data_file in data_files]
        )
        if not data_files_exist:
            data_path = self._model_path.resolve() / "data"
            Path(data_path).mkdir(parents=True, exist_ok=True)
            data_manager.export_data_DocBin(
                data_path, overwrite=True, check_data_integrity=check_data_integrity
            )
        return data_manager.spacy_docbin_files
