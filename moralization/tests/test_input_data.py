from moralization.input_data import InputOutput
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


def test_InputOutput_span_merge(doc_dicts):

    all_categories = [
        "sc",
        "paragraphs",
        "KAT1-Moralisierendes Segment",
        "KAT2-Moralwerte",
        "KAT2-Subjektive Ausdrücke",
        "KAT3-Gruppe",
        "KAT3-Rolle",
        "KAT3-own/other",
        "KAT4-Kommunikative Funktion",
        "KAT5-Forderung explizit",
        "task1",
        "task2",
        "task3",
        "task4",
        "task5",
    ]

    # default merge dict
    for doc_dict in doc_dicts:
        merged_dict = InputOutput._merge_span_categories(doc_dict)
        for doc in merged_dict.values():
            generated_categories = list(doc.spans.keys())
            assert all_categories == generated_categories

    # custom merge dict
    merge_dict = {
        "sc": "all",
        "task1": ["KAT1-Moralisierendes Segment"],
        "task2": ["KAT2-Moralwerte", "KAT2-Subjektive Ausdrücke"],
        "task3": ["KAT3-Rolle", "KAT3-Gruppe", "KAT3-own/other"],
        "task4": ["KAT4-Kommunikative Funktion"],
        "task5": ["KAT5-Forderung explizit"],
    }

    for doc_dict in doc_dicts:
        merged_dict = InputOutput._merge_span_categories(doc_dict, merge_dict)
        for doc in merged_dict.values():
            generated_categories = list(doc.spans.keys())
            assert all_categories == generated_categories


def test_InputOutput_read_data(data_dir):
    doc_dict, train_dict, test_dict = InputOutput.read_data(data_dir)
    testFilenameList = sorted(doc_dict.keys())
    correctlist = sorted(
        [
            "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
            "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
        ]
    )

    assert testFilenameList == correctlist

    spans_set = {
        "sc",
        "paragraphs",
        "KAT1-Moralisierendes Segment",
        "KAT2-Moralwerte",
        "KAT2-Subjektive Ausdrücke",
        "KAT3-Gruppe",
        "KAT3-Rolle",
        "KAT3-own/other",
        "KAT4-Kommunikative Funktion",
        "KAT5-Forderung explizit",
        "task1",
        "task2",
        "task3",
        "task4",
        "task5",
    }
    # assert categories
    assert set(doc_dict[correctlist[0]].spans.keys()) == spans_set

    # assert spans
    test_string = (
        "HMP05/AUG.00228 Hamburger Morgenpost, 03.08.2005, S. 5; "
        + "ALG II ist mit der Menschenwürde vereinbar ### BERLIN Das "
        + "Arbeitslosengeld II ist nicht so niedrig, dass dadurch die "
        + "grundgesetzlich garantierte Menschenwürde verletzt wird, "
        + "urteilte das Sozialgericht Berlin."
    )
    assert doc_dict[correctlist[0]].spans["paragraphs"][0].text.strip() == test_string
    assert doc_dict[correctlist[0]].spans["paragraphs"][0].start == 1
    assert doc_dict[correctlist[0]].spans["paragraphs"][0].end == 45

    for train_file, test_file, main_file in zip(
        train_dict.values(), test_dict.values(), doc_dict.values()
    ):

        assert len(main_file.spans["sc"]) == len(test_file.spans["sc"]) + len(
            train_file.spans["sc"]
        )
        assert len(test_file.spans["sc"]) * 4 <= len(train_file.spans["sc"])
