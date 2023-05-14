import importlib_metadata as metadata
from moralization.data_manager import DataManager
from moralization.spacy_model_manager import SpacyModelManager
from moralization.transformers_data_handler import TransformersDataHandler
from moralization.transformers_model_manager import TransformersModelManager

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata


__all__ = [
    "DataManager",
    "SpacyModelManager",
    "TransformersDataHandler",
    "TransformersModelManager",
]
