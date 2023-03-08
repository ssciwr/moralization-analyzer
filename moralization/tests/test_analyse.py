from moralization import analyse
import pytest
from collections import defaultdict


def test__reduce_cat_list():
    test_list = ["sc", "paragraphs", "test"]
    reduced_list = analyse._reduce_cat_list(test_list)
    assert reduced_list == ["test"]


def test_get_paragraphs(doc_dicts):
    filename = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    doc = doc_dicts[0][filename]
    print(analyse._get_paragraphs(doc))
    paragraph_list = [
        [1, 87],
        [88, 194],
        [195, 331],
        [332, 485],
        [486, 543],
        [544, 671],
        [672, 799],
        [800, 845],
    ]

    assert analyse._get_paragraphs(doc) == paragraph_list


def test__find_spans_in_paragraph(doc_dicts):
    doc = list(doc_dicts[0].values())[0]
    with pytest.raises(KeyError):
        analyse._find_spans_in_paragraph(doc, "test")

    span_key = "KAT1-Moralisierendes Segment"
    spans_idx = analyse._find_spans_in_paragraph(doc, span_key)
    print(spans_idx)

    assert len(spans_idx) == 11
    assert spans_idx[0] == (0, "Moralisierung explizit")


def test_summarize_span_occurences(doc_dicts):
    doc = list(doc_dicts[0].values())[0]

    df = analyse._summarize_span_occurences(doc)
    # check occurences of first row
    assert list(df.iloc[0].values) == [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    column_names = list(df.columns)
    real_names = [
        ("KAT1-Moralisierendes Segment", "Moralisierung explizit"),
        ("KAT1-Moralisierendes Segment", "Keine Moralisierung"),
        ("KAT1-Moralisierendes Segment", "Moralisierung"),
        ("KAT2-Moralwerte", "Care"),
        ("KAT2-Subjektive Ausdrücke", "Fairness"),
        ("KAT2-Subjektive Ausdrücke", "Oppression"),
        ("KAT2-Subjektive Ausdrücke", "Cheating"),
        ("KAT3-Gruppe", "Institution"),
        ("KAT3-Gruppe", "Individuum"),
        ("KAT3-Gruppe", "soziale Gruppe"),
        ("KAT3-Rolle", "Forderer:in"),
        ("KAT3-Rolle", "Benefizient:in"),
        ("KAT3-own/other", "Neutral"),
        ("KAT4-Kommunikative Funktion", "Darstellung"),
        ("KAT4-Kommunikative Funktion", "Appell"),
        ("KAT5-Forderung explizit", "explizit"),
        ("task1", "Moralisierung explizit"),
        ("task1", "Keine Moralisierung"),
        ("task1", "Moralisierung"),
        ("task2", "Care"),
        ("task2", "Fairness"),
        ("task2", "Oppression"),
        ("task2", "Cheating"),
        ("task3", "Forderer:in"),
        ("task3", "Benefizient:in"),
        ("task3", "Institution"),
        ("task3", "Individuum"),
        ("task3", "soziale Gruppe"),
        ("task4", "Darstellung"),
        ("task4", "Appell"),
        ("task5", "explizit"),
    ]
    assert sorted(column_names) == sorted(real_names)


def test_loop_over_files(doc_dicts):
    def _index_to_dict(index_list):
        index_dict = defaultdict(list)
        for file_index, sentence in index_list:
            index_dict[file_index].append(sentence)
        return index_dict

    file1 = "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB"
    file2 = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"

    # wrong filter
    with pytest.raises(KeyError):
        analyse._loop_over_files(doc_dicts[0], "test_filter")

    # default filter
    default_df = analyse._loop_over_files(doc_dicts[0], None)
    index_list = list(default_df.index)
    index_dict = _index_to_dict(index_list)
    assert len(index_dict) == 2
    assert list(index_dict.keys())[0] == file1
    assert list(index_dict.keys())[1] == file2

    # manual filter
    manual_df = analyse._loop_over_files(doc_dicts[0], [file1, file2])
    index_list = list(manual_df.index)
    index_dict = _index_to_dict(index_list)
    assert len(index_dict) == 2
    assert list(index_dict.keys())[0] == file1
    assert list(index_dict.keys())[1] == file2

    # one filter as list
    one_list_df = analyse._loop_over_files(doc_dicts[0], [file2])
    index_list = list(one_list_df.index)
    index_dict = _index_to_dict(index_list)
    assert len(index_dict) == 1
    print(index_dict.keys())
    assert list(index_dict.keys())[0] == file2

    # one filter as str
    ond_str_df = analyse._loop_over_files(doc_dicts[0], file2)
    index_list = list(ond_str_df.index)
    index_dict = _index_to_dict(index_list)
    assert len(index_dict) == 1
    assert list(index_dict.keys())[0] == file2
