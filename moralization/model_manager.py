import huggingface_hub
import spacy_huggingface_hub
import os
import spacy
from pathlib import Path
from typing import Union, Optional, Dict, Any
import tempfile
import re
import logging


def _construct_wheel_path(model_path: Path, meta: Dict[str, Any]) -> Path:
    full_name = f"{meta['lang']}_{meta['name']}-{meta['version']}"
    return model_path / full_name / "dist" / f"{full_name}-py3-none-any.whl"


def _make_valid_package_name(name: str) -> str:
    # attempt to make name valid, throw exception if we fail
    # https://packaging.python.org/en/latest/specifications/name-normalization
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


class ModelManager:
    """
    Import, modify and publish models to hugging face.
    """

    _meta_keys_to_expose_to_user = [
        "name",
        "version",
        "description",
        "author",
        "email",
        "url",
        "license",
    ]

    def __init__(self, model_path: Union[str, Path] = None):
        self.load(model_path)

    def load(self, model_path: Union[str, Path]):
        """Load a spacy model from `model_path`."""
        self.model_path = Path(model_path)
        self.spacy_model = spacy.load(model_path)
        self.metadata = {
            k: self.spacy_model.meta.get(k, "")
            for k in self._meta_keys_to_expose_to_user
        }

    def save(self):
        """Save any changes made to the model metadata."""
        self._update_metadata()
        self.spacy_model.to_disk(self.model_path)

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
        self.save()
        if hugging_face_token is None:
            hugging_face_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hugging_face_token is None:
            raise ValueError(
                "API TOKEN required: pass as string or set the HUGGING_FACE_TOKEN environment variable."
            )
        huggingface_hub.login(token=hugging_face_token)
        with tempfile.TemporaryDirectory() as tmpdir:
            # convert model to a python package incl binary wheel
            output_path = Path(tmpdir)
            spacy.cli.package(self.model_path, output_path, create_wheel=True)
            # push the package to hugging face
            return spacy_huggingface_hub.push(
                _construct_wheel_path(output_path, self.spacy_model.meta)
            )

    def _update_metadata(self):
        self.metadata["name"] = _make_valid_package_name(self.metadata.get("name"))
        for k, v in self.metadata.items():
            if k in self.spacy_model.meta:
                self.spacy_model.meta[k] = v
