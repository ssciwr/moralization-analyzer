import pytest
from moralization import input_data
from moralization.data_manager import DataManager
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
def model_path(data_dir, config_file, tmp_path_factory) -> pathlib.Path:
    """
    Returns a temporary path containing a trained model.
    This is only created once and re-used for the entire pytest session.
    """
    dm = DataManager(data_dir)
    dm.export_data_DocBin()
    tmp_path = tmp_path_factory.mktemp("model")
    dm.spacy_train(working_dir=tmp_path, config=config_file, n_epochs=1)
    yield tmp_path / "output" / "model-best"


@pytest.fixture
def doc_dicts(data_dir):
    return input_data.InputOutput.read_data(str(data_dir))
