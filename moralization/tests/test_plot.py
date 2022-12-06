from moralization import analyse, plot
import matplotlib


def test_PlotSpans_report_occurrence_heatmap(data_dict):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    plot.PlotSpans.report_occurrence_heatmap(df_sentence_occurrence)
    plot.PlotSpans.report_occurrence_heatmap(
        df_sentence_occurrence,
        filter_=["KAT1MoralisierendesSegment", "Neutral", "Care"],
    )
    # check support for old and new category labels.
    plot.PlotSpans.report_occurrence_heatmap(
        df_sentence_occurrence,
        filter_=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )


def test_PlotSpans_report_occurrence_matrix(data_dict):
    df_sentence_occurrence = analyse.AnalyseSpans.report_occurrence_per_paragraph(
        data_dict
    )
    plot.PlotSpans.report_occurrence_matrix(df_sentence_occurrence)
    plot.PlotSpans.report_occurrence_matrix(
        df_sentence_occurrence,
        filter_=["KAT1MoralisierendesSegment", "Neutral", "Care"],
    )
    # check support for old and new category labels.
    plot.PlotSpans.report_occurrence_matrix(
        df_sentence_occurrence,
        filter_=["KAT1-Moralisierendes Segment", "Neutral", "Care"],
    )


def test_InteractiveCategoryPlot(data_dict):
    # removes automatic window opening by matplotlib
    matplotlib.use("Agg")

    heatmap = plot.InteractiveCategoryPlot(data_dict, figsize=(15, 10))
    heatmap.show()

    heatmap = plot.InteractiveCategoryPlot(data_dict)
    heatmap.show()

    heatmap = plot.InteractiveCategoryPlot(
        data_dict, plot_callback=plot.PlotSpans.report_occurrence_heatmap
    )
    heatmap.show()
