from moralization.inout import InputOutput
import pathlib
import pytest

data_dir = pathlib.Path("../moralization_data/Test_Data/XMI_11/")
ts_file = pathlib.Path("../moralization_data/Test_Data/XMI_11/TypeSystem.xml")
data_file = pathlib.Path(
    "../moralization_data/Test_Data/XMI_11/test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi"
)


def test_InputOutput_get_file_type():
    filename = data_dir.joinpath(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi"
    )
    assert InputOutput.get_file_type(filename) == "xmi"
    filename = data_dir.joinpath("TypeSystem.xml")
    assert InputOutput.get_file_type(filename) == "xml"


def test_InputOutput_read_typesystem():
    ts = InputOutput.read_typesystem()
    ts = InputOutput.read_typesystem(ts_file)
    # test wrong filetype
    with pytest.raises(Warning):
        ts = InputOutput.read_typesystem(data_file)


def test_InputOutput_read_cas_file():
    ts = InputOutput.read_typesystem()
    cas, file_type = InputOutput.read_cas_file(data_file, ts)
    assert file_type == "xmi"


def test_InputOutput_get_input_file():
    filename = data_dir.joinpath(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi"
    )
    data = InputOutput.get_input_file(filename)


def test_InputOutput_get_input_dir():
    data_dict = InputOutput.get_input_dir(data_dir)
    data_dict = InputOutput.get_input_dir(data_dir, use_custom_ts=True)
    print(list(data_dict.keys()))
    assert (
        list(data_dict.keys()).sort()
        == [
            "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
            "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
        ].sort()
    )
