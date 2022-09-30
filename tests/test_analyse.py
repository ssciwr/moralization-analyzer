from moralization import inout, analyse
import pytest
import pathlib

data_dir = pathlib.Path("../moralization_data/Test_Data/XMI_11/")
data_dict = inout.InputOutput.get_input_dir(data_dir)
ts_file = pathlib.Path("../moralization_data/Test_Data/XMI_11/TypeSystem.xml")
data_file = pathlib.Path(
    "../moralization_data/Test_Data/XMI_11/test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi"
)


def test_get_spans():
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


def test_get_paragraphs():
    ts = inout.InputOutput.read_typesystem()
    cas, file_type = inout.InputOutput.read_cas_file(data_file, ts)
    paragraph_dict = analyse.get_paragraphs(cas, ts)
    assert list(paragraph_dict.keys()) == ["span", "sofa"]
    assert len(paragraph_dict["span"]) == len(paragraph_dict["sofa"])


def test_AnalyseOccurence():
    df_instances = analyse.AnalyseOccurence(data_dict, mode="instances").df
    df_spans = analyse.AnalyseOccurence(data_dict, mode="spans").df
    df_sindex = analyse.AnalyseOccurence(data_dict, mode="span_index").df

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


def test_AnalyseSpans_report_occurence_per_paragraph():
    df_sentence_occurence = analyse.AnalyseSpans.report_occurence_per_paragraph(
        data_dict
    )
    assert len(df_sentence_occurence) == 17

    df_sentence_occurence = analyse.AnalyseSpans.report_occurence_per_paragraph(
        data_dict, data_file
    )
    assert len(df_sentence_occurence) == 7

    df_sentence_occurence = analyse.AnalyseSpans.report_occurence_per_paragraph(
        data_dict, "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW.xmi"
    )
    assert len(df_sentence_occurence) == 7

    df_sentence_occurence = analyse.AnalyseSpans.report_occurence_per_paragraph(
        data_dict,
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    )
    assert len(df_sentence_occurence) == 10


def test_PlotSpans_report_occurence_heatmap():
    df_sentence_occurence = analyse.AnalyseSpans.report_occurence_per_paragraph(
        data_dict
    )
    analyse.PlotSpans.report_occurence_heatmap(df_sentence_occurence)
    analyse.PlotSpans.report_occurence_heatmap(
        df_sentence_occurence, filter=["KAT1MoralisierendesSegment", "Neutral", "Care"]
    )
    # check support for old and new category labels.
    analyse.PlotSpans.report_occurence_heatmap(
        df_sentence_occurence,
        filter=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )


def test_PlotSpans_report_occurence_matrix():
    df_sentence_occurence = analyse.AnalyseSpans.report_occurence_per_paragraph(
        data_dict
    )
    analyse.PlotSpans.report_occurence_matrix(df_sentence_occurence)
    analyse.PlotSpans.report_occurence_matrix(
        df_sentence_occurence, filter=["KAT1MoralisierendesSegment", "Neutral", "Care"]
    )
    # check support for old and new category labels.
    analyse.PlotSpans.report_occurence_matrix(
        df_sentence_occurence,
        filter=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )
