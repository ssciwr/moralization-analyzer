from moralization import analyse, plot
import pytest


def test_PlotSpans_report_occurrence_heatmap(data_dict):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    plot.PlotSpans.report_occurrence_heatmap(df_sentence_occurrence)
    plot.PlotSpans.report_occurrence_heatmap(
        df_sentence_occurrence, filter=["KAT1MoralisierendesSegment", "Neutral", "Care"]
    )
    # check support for old and new category labels.
    plot.PlotSpans.report_occurrence_heatmap(
        df_sentence_occurrence,
        filter=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )


def test_PlotSpans_report_occurrence_matrix(data_dict):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    plot.PlotSpans.report_occurrence_matrix(df_sentence_occurrence)
    plot.PlotSpans.report_occurrence_matrix(
        df_sentence_occurrence, filter=["KAT1MoralisierendesSegment", "Neutral", "Care"]
    )
    # check support for old and new category labels.
    plot.PlotSpans.report_occurrence_matrix(
        df_sentence_occurrence,
        filter=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )
