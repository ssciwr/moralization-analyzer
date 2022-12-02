from moralization.input import InputOutput
import pytest


def test_InputOutput_get_file_type(data_dir):
    filename = data_dir.joinpath(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi"
    )
    assert InputOutput.get_file_type(filename) == "xmi"
    filename = data_dir.joinpath("TypeSystem.xml")
    assert InputOutput.get_file_type(filename) == "xml"


def test_InputOutput_read_typesystem(ts_file, data_file):
    _ = InputOutput.read_typesystem()
    _ = InputOutput.read_typesystem(ts_file)
    # test wrong filetype
    with pytest.raises(Warning):
        _ = InputOutput.read_typesystem(data_file)


def test_InputOutput_read_cas_file(data_file):
    ts = InputOutput.read_typesystem()
    _, file_type = InputOutput.read_cas_file(data_file, ts)
    assert file_type == "xmi"


def test_InputOutput_get_input_file(data_dir):
    filename = data_dir.joinpath(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi"
    )
    _ = InputOutput.get_input_file(filename)


def test_InputOutput_get_input_dir(data_dir):
    data_dict = InputOutput.get_input_dir(data_dir)
    with pytest.raises(FileNotFoundError):
        InputOutput.get_input_dir("./not_real_dir/")
    with pytest.raises(FileNotFoundError):
        InputOutput.get_input_dir(".")

    data_dict = InputOutput.get_input_dir(data_dir)
    assert data_dict
    data_dict = InputOutput.get_input_dir(data_dir, use_custom_ts=True)
    testlist = list(data_dict.keys())
    correctlist = [
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    ]
    assert set(testlist) == set(correctlist)
