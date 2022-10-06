from moralization import inout, analyse
import pytest


def test_get_spans(data_file):
    ts = inout.InputOutput.read_typesystem()
    cas, file_type = inout.InputOutput.read_cas_file(data_file, ts)
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
    ts = inout.InputOutput.read_typesystem()
    cas, file_type = inout.InputOutput.read_cas_file(data_file, ts)
    paragraph_dict = analyse.get_paragraphs(cas, ts)
    assert list(paragraph_dict.keys()) == ["span", "sofa"]
    assert len(paragraph_dict["span"]) == len(paragraph_dict["sofa"])


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


def test_AnalyseSpans_report_occurrence_per_paragraph(data_dict, data_file):

    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    assert len(df_sentence_occurrence) == 17

    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict, data_file
    )
    assert len(df_sentence_occurrence) == 7

    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict, "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi"
    )
    assert len(df_sentence_occurrence) == 7

    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict,
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    )
    assert len(df_sentence_occurrence) == 10


def test_PlotSpans_report_occurrence_heatmap(data_dict):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    analyse.PlotSpans.report_occurrence_heatmap(df_sentence_occurrence)
    analyse.PlotSpans.report_occurrence_heatmap(
        df_sentence_occurrence, filter=["KAT1MoralisierendesSegment", "Neutral", "Care"]
    )
    # check support for old and new category labels.
    analyse.PlotSpans.report_occurrence_heatmap(
        df_sentence_occurrence,
        filter=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )


def test_PlotSpans_report_occurrence_matrix(data_dict):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    analyse.PlotSpans.report_occurrence_matrix(df_sentence_occurrence)
    analyse.PlotSpans.report_occurrence_matrix(
        df_sentence_occurrence, filter=["KAT1MoralisierendesSegment", "Neutral", "Care"]
    )
    # check support for old and new category labels.
    analyse.PlotSpans.report_occurrence_matrix(
        df_sentence_occurrence,
        filter=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )
