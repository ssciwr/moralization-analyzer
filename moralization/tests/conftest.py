import pytest
from moralization import input_data
from moralization.data_manager import DataManager
from moralization.spacy_model_manager import SpacyModelManager
import pathlib


@pytest.fixture(scope="session")
def data_dir():
    return pathlib.Path(__file__).parents[1].resolve() / "data"


@pytest.fixture(scope="session")
def ts_file(data_dir):
    return data_dir / "TypeSystem.xml"


@pytest.fixture(scope="session")
def data_file(data_dir):
    return (
        data_dir / "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi"
    )


@pytest.fixture(scope="session")
def config_file(data_dir):
    return data_dir / "config.cfg"


@pytest.fixture(scope="session")
def spacy_model_path(data_dir, config_file, tmp_path_factory) -> pathlib.Path:
    """
    Returns a temporary path containing a trained SpacyModelManager model with valid metadata.
    This is only created once and re-used for the entire pytest session.
    """
    data_manager = DataManager(data_dir)
    model_path = tmp_path_factory.mktemp("spacy_model") / "my_model"
    model = SpacyModelManager(model_path, config_file)
    model.metadata = {
        "name": "pytest_pipeline",
        "version": "0.1.2",
        "description": "Test pipeline generated for testing",
        "author": "SSC",
        "email": "ssc@iwr.uni-heidelberg.de",
        "url": "https://ssc.iwr.uni-heidelberg.de/",
        "license": "MIT",
    }
    model.train(
        data_manager, overrides={"training.max_epochs": 5}, check_data_integrity=False
    )
    yield model_path


@pytest.fixture
def doc_dicts(data_dir):
    return input_data.InputOutput.read_data(str(data_dir))
