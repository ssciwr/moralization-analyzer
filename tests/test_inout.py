from moralization import inout
import pathlib
import cassis

data_dir = pathlib.Path("../moralization_data/Test_Data/XMI_11/")


def test_InputOutput_get_file_type():
    filename = data_dir.joinpath(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi"
    )
    assert inout.InputOutput.get_file_type(filename) == "xmi"
    filename = data_dir.joinpath("TypeSystem.xml")
    assert inout.InputOutput.get_file_type(filename) == "xml"


def test_InputOutput_read_typesystem():
    filename = data_dir.joinpath("TypeSystem.xml")
    ts = inout.InputOutput.read_typesystem(filename)
    assert isinstance(ts, cassis.TypeSystem)


def test_InputOutput_get_input_file():
    filename = data_dir.joinpath(
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi"
    )
    data = inout.InputOutput.get_input_file(filename)


def test_InputOutput_get_input_dir():
    data_dict = inout.InputOutput.get_input_dir(data_dir)
