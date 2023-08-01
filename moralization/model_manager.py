from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Dict, Any
from moralization.data_manager import DataManager
import os
import huggingface_hub


class ModelManager(ABC):
    """
    Create, import, modify, train and publish a model.

    Models can be trained on data from a DataManager, and published to Hugging Face.
    """

    @abstractmethod
    def __init__(self, model_path: Union[str, Path]):
        self._model_path = Path(model_path)

    @property
    def model_path(self) -> Path:
        """
        The folder where the model is stored.
        """
        return self._model_path

    def __repr__(self) -> str:
        return f"ModelManager('{self.model_path}')"

    @abstractmethod
    def train(self, data_manager: DataManager) -> None:
        """Train the model on the data contained in `data_manager`.

        Args:
            data_manager (DataManager): the DataManager that contains the training data
        """
        pass

    @abstractmethod
    def evaluate(self, data_manager: DataManager) -> Dict[str, Any]:
        """Evaluate the model against the test dataset in `data_manager`.
        Args:
            data_manager (DataManager): the DataManager that contains the training data
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Save any changes made to the model."""
        pass

    @abstractmethod
    def test(self, test_string: str, style: str = "span"):
        """Test the model output with a test string."""
        pass

    @abstractmethod
    def publish(self, hugging_face_token: Optional[str] = None) -> str:
        """Publish the model to Hugging Face.

        This requires a User Access Token from https://huggingface.co/

        The token can either be passed via the `hugging_face_token` argument,
        or it can be set via the `HUGGING_FACE_TOKEN` environment variable.

        Args:
            hugging_face_token (str, optional): Hugging Face User Access Token
        Returns:
            str: URL of the published model
        """
        pass

    def _login_to_huggingface(self, hugging_face_token: Optional[str] = None) -> None:
        """Login to hugging face using the supplied token

                The token can either be passed via the `hugging_face_token` argument,
                or it can be set via the `HUGGING_FACE_TOKEN` environment variable.

                Args:
                    hugging_face_token (str, optional): Hugging Face User Access Token
        0"""
        if hugging_face_token is None:
            hugging_face_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hugging_face_token is None:
            print("Obtaining token directly from user..")
            huggingface_hub.login()
        else:
            huggingface_hub.login(token=hugging_face_token)
