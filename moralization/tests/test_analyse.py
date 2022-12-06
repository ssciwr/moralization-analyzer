from moralization import input, analyse
import pytest
from collections import defaultdict


@pytest.fixture
def get_data_dict(ts_file, data_file):
    ts = input.InputOutput.read_typesystem(ts_file)
    data_dict = input.InputOutput.read_cas_content(data_file, ts)
    return data_dict


@pytest.fixture
def occurrence_obj(get_data_dict):
    return analyse.AnalyseOccurrence(get_data_dict)


def test_validate_data_dict():
    with pytest.raises(ValueError):
        analyse.validate_data_dict({})
    testdict = {"something": {}}
    with pytest.raises(ValueError):
        analyse.validate_data_dict(testdict)
    testdict = {"something": ["empty_list"]}
    with pytest.raises(ValueError):
        analyse.validate_data_dict(testdict)
    testdict = {"something": {"data": "my_data"}}
    with pytest.raises(ValueError):
        analyse.validate_data_dict(testdict)


def test_get_spans(data_file):
    ts = input.InputOutput.read_typesystem()
    cas, _ = input.InputOutput.read_cas_file(data_file, ts)
    span_dict = analyse.get_spans(cas, ts)
    assert list(span_dict.keys()) == [
        "KAT1MoralisierendesSegment",
        "KommunikativeFunktion",
        "Protagonistinnen2",
        "Protagonistinnen",
        "Protagonistinnen3",
        "Moralwerte",
        "KAT2Subjektive_Ausdrcke",
        "Forderung",
    ]
    assert list(span_dict["KAT1MoralisierendesSegment"].keys()) == [
        "Moralisierung explizit",
        "Moralisierung interpretativ",
        "Keine Moralisierung",
    ]


def test_get_paragraphs(data_file):
    ts = input.InputOutput.read_typesystem()
    cas, _ = input.InputOutput.read_cas_file(data_file, ts)
    paragraph_dict = analyse.get_paragraphs(cas, ts)
    assert list(paragraph_dict.keys()) == ["span", "sofa"]
    assert len(paragraph_dict["span"]) == len(paragraph_dict["sofa"])


def test_list_categories():
    # unravel a nested dict
    mydict = {"one": {"a": 0, "b": 0}, "two": {"c": 1, "d": 1}}
    mylist = analyse.list_categories(mydict)
    correctlist = [("one", "a"), ("one", "b"), ("two", "c"), ("two", "d")]
    assert mylist == correctlist


def test_AnalyseOccurrence(data_dict):
    with pytest.raises(ValueError):
        analyse.AnalyseOccurrence({})
    with pytest.raises(ValueError):
        false_data_dict = {"foo": {}}
        analyse.AnalyseOccurrence(false_data_dict)
    with pytest.raises(ValueError):
        false_data_dict = {"foo": "bar"}
        analyse.AnalyseOccurrence(false_data_dict)
    with pytest.raises(ValueError):
        false_data_dict = {"foo": {"bar": "test"}}
        analyse.AnalyseOccurrence(false_data_dict)
    df_instances = analyse.AnalyseOccurrence(data_dict, mode="instances").df
    df_spans = analyse.AnalyseOccurrence(data_dict, mode="spans").df
    df_sindex = analyse.AnalyseOccurrence(data_dict, mode="span_index").df
    # check that map_expressions was applied correctly
    with pytest.raises(KeyError):
        df_spans.loc["KAT2Subjektive_Ausdrcke"]
    with pytest.raises(KeyError):
        df_instances.loc["KAT2Subjektive_Ausdrcke"]
    with pytest.raises(KeyError):
        df_sindex.loc["KAT2Subjektive_Ausdrcke"]
    assert len(df_instances.loc["KAT2-Subjektive Ausdrücke"]) == 6
    assert len(df_spans.loc["KAT2-Subjektive Ausdrücke"]) == 6
    assert len(df_sindex.loc["KAT2-Subjektive Ausdrücke"]) == 6


def test_AnalyseOccurrence_initialize_files(occurrence_obj):
    correctlist = [
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
    ]
    testlist = occurrence_obj._initialize_files(None)
    assert set(testlist) == set(correctlist)
    testlist = occurrence_obj._initialize_files(correctlist[0])
    assert set(testlist) == set(correctlist)


def test_AnalyseOccurrence_initialize_dict(occurrence_obj):
    assert isinstance(occurrence_obj._initialize_dict(), defaultdict)


def test_AnalyseOccurrence_initialize_df(occurrence_obj):
    occurrence_obj._initialize_df()
    correctindex = ["Main Category", "Sub Category"]
    assert occurrence_obj.df.index.names == correctindex


def test_AnalyseOccurrence_get_categories(occurrence_obj):
    file_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    span_dict = occurrence_obj.data_dict[file_name]["data"]
    testlist = occurrence_obj._get_categories(span_dict, file_name)
    test_key = ("KAT1MoralisierendesSegment", "Moralisierung explizit")
    assert testlist[file_name][test_key] == 3


def test_AnalyseOccurrence_add_total(occurrence_obj):
    file_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    occurrence_obj._initialize_df()
    occurrence_obj._add_total()
    assert occurrence_obj.df.loc[("total instances", "with invalid")][file_name] == 79


def test_AnalyseSpans_report_occurrence_per_paragraph(data_dict, data_file):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    assert len(df_sentence_occurrence) == 9
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict, data_file
    )
    assert len(df_sentence_occurrence) == 5
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict, "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi"
    )
    assert len(df_sentence_occurrence) == 5
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict,
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    )
    assert len(df_sentence_occurrence) == 4
