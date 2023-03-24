import importlib_metadata as metadata
from moralization.data_manager import DataManager
from moralization.spacy_model_manager import SpacyModelManager

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata


__all__ = [
    "DataManager",
    "SpacyModelManager",
]
