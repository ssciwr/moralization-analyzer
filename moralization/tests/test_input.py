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


def test_InputOutput_add_custom_instance_to_ts(ts_file, data_file):
    # ts = InputOutput.read_typesystem(ts_file)
    # cas = InputOutput.read_cas_file(data_file, ts)
    # cas, ts = InputOutput.add_custom_instance_to_ts(cas, ts)
    pass


def test_InputOutput_get_multiple_input(data_dir):
    data_files, ts_file = InputOutput.get_multiple_input(data_dir)
    with pytest.raises(FileNotFoundError):
        InputOutput.get_multiple_input("./not_real_dir/")
    with pytest.raises(FileNotFoundError):
        InputOutput.get_multiple_input(".")
    data_files = [i.parts[-1] for i in data_files]
    test_files = [
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB.xmi",
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi",
    ]
    assert set(data_files) == set(test_files)
    assert ts_file.parts[-1] == "TypeSystem.xml"


def test_InputOutput_read_cas_content(data_dir):
    data_files, ts_file = InputOutput.get_multiple_input(data_dir)
    # read in the ts
    ts = InputOutput.read_typesystem(ts_file)
    data_dict = InputOutput.read_cas_content(data_files, ts)
    testlist = list(data_dict.keys())
    correctlist = [
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    ]
    assert set(testlist) == set(correctlist)
    # add more assert statements here
