from moralization import plot
import matplotlib
from moralization.analyse import _loop_over_files
import pytest
import seaborn as sns
from moralization.data_manager import DataManager


def test_report_occurrence_heatmap(doc_dicts, monkeypatch):
    df = _loop_over_files(doc_dicts[0])

    # test corr without filter
    corr_df = plot.report_occurrence_heatmap(df, _type="corr")
    all_columns = [
        ("KAT1-Moralisierendes Segment", "Keine Moralisierung"),
        ("KAT1-Moralisierendes Segment", "Moralisierung"),
        ("KAT1-Moralisierendes Segment", "Moralisierung explizit"),
        ("KAT1-Moralisierendes Segment", "Moralisierung interpretativ"),
        ("KAT2-Moralwerte", "Care"),
        ("KAT2-Moralwerte", "Cheating"),
        ("KAT2-Moralwerte", "Fairness"),
        ("KAT2-Moralwerte", "Harm"),
        ("KAT2-Moralwerte", "Liberty"),
        ("KAT2-Moralwerte", "Oppression"),
        ("KAT2-Subjektive Ausdrücke", "Care"),
        ("KAT2-Subjektive Ausdrücke", "Cheating"),
        ("KAT2-Subjektive Ausdrücke", "Fairness"),
        ("KAT2-Subjektive Ausdrücke", "Harm"),
        ("KAT2-Subjektive Ausdrücke", "Liberty"),
        ("KAT2-Subjektive Ausdrücke", "Oppression"),
        ("KAT3-Gruppe", "Individuum"),
        ("KAT3-Gruppe", "Institution"),
        ("KAT3-Gruppe", "Menschen"),
        ("KAT3-Gruppe", "soziale Gruppe"),
        ("KAT3-Rolle", "Adresassat:in"),
        ("KAT3-Rolle", "Benefizient:in"),
        ("KAT3-Rolle", "Forderer:in"),
        ("KAT3-Rolle", "Kein Bezug"),
        ("KAT3-own/other", "Neutral"),
        ("KAT3-own/other", "Other Group"),
        ("KAT3-own/other", "Own Group"),
        ("KAT4-Kommunikative Funktion", "Appell"),
        ("KAT4-Kommunikative Funktion", "Darstellung"),
        ("KAT5-Forderung explizit", "explizit"),
        ("task1", "Keine Moralisierung"),
        ("task1", "Moralisierung"),
        ("task1", "Moralisierung explizit"),
        ("task1", "Moralisierung interpretativ"),
        ("task2", "Care"),
        ("task2", "Cheating"),
        ("task2", "Fairness"),
        ("task2", "Harm"),
        ("task2", "Liberty"),
        ("task2", "Oppression"),
        ("task3", "Adresassat:in"),
        ("task3", "Benefizient:in"),
        ("task3", "Forderer:in"),
        ("task3", "Individuum"),
        ("task3", "Institution"),
        ("task3", "Kein Bezug"),
        ("task3", "soziale Gruppe"),
        ("task4", "Appell"),
        ("task4", "Darstellung"),
        ("task5", "explizit"),
    ]

    assert sorted(list(corr_df.columns)) == sorted(all_columns)

    # test corr wrong filter
    with pytest.raises(KeyError):
        plot.report_occurrence_heatmap(df, _filter="test", _type="corr")

    # test corr with filter main cat:
    corr_df = plot.report_occurrence_heatmap(df, _filter="KAT3-Gruppe", _type="corr")
    filtered_columns_main = [
        ("KAT3-Gruppe", "Individuum"),
        ("KAT3-Gruppe", "Institution"),
        ("KAT3-Gruppe", "Menschen"),
        ("KAT3-Gruppe", "soziale Gruppe"),
    ]

    assert sorted(list(corr_df.columns)) == sorted(filtered_columns_main)

    # filter based on sub cat
    filtered_columns_sub = [
        ("KAT3-Gruppe", "Individuum"),
        ("KAT3-Gruppe", "Institution"),
        ("task3", "Individuum"),
        ("task3", "Institution"),
    ]
    corr_df = plot.report_occurrence_heatmap(
        df, _filter=["Individuum", "Institution"], _type="corr"
    )
    assert sorted(list(corr_df.columns)) == sorted(filtered_columns_sub)

    # test filter both cats

    filtered_columns_both = [
        ("KAT3-Gruppe", "Individuum"),
        ("KAT3-Gruppe", "Institution"),
    ]
    corr_df = plot.report_occurrence_heatmap(
        df, _filter=["KAT3-Gruppe", "Individuum", "Institution"], _type="corr"
    )
    assert sorted(list(corr_df.columns)) == sorted(filtered_columns_both)

    # test heatmap without filter
    monkeypatch.setattr(sns, "heatmap", lambda x, cmap: list(x.columns))
    heatmap_columns_full = plot.report_occurrence_heatmap(df, _type="heatmap")
    assert sorted(heatmap_columns_full) == sorted(all_columns)

    # test heatmap with filter
    heatmap_columns_filtered = plot.report_occurrence_heatmap(
        df, _filter="KAT3-Gruppe", _type="heatmap"
    )
    assert heatmap_columns_filtered == filtered_columns_main


def test_InteractiveAnalyzerResults(data_dir):
    matplotlib.use("Agg")

    dm = DataManager(data_dir)
    test_interactive = plot.InteractiveAnalyzerResults(dm.return_analyzer_result("all"))
    test_interactive.visualize_analyzer_result()

    with pytest.raises(KeyError):
        test_interactive.visualize_analyzer_result(span_label="test")
    with pytest.raises(KeyError):
        test_interactive.visualize_analyzer_result(analysis_type="test")


def test_InteractiveCategoryPlot(doc_dicts):
    df = _loop_over_files(doc_dicts[0])
    filenames = list(doc_dicts[0].keys())

    # removes automatic window opening by matplotlib
    matplotlib.use("Agg")

    heatmap = plot.InteractiveCategoryPlot(df, filenames, figsize=(15, 10))
    heatmap.show()

    heatmap = plot.InteractiveCategoryPlot(
        df,
        filenames,
    )
    heatmap.show()

    heatmap = plot.InteractiveCategoryPlot(
        df, filenames, plot_callback=plot.report_occurrence_heatmap
    )

    heatmap.show()


def test_spacy_datah_andler_visualize_data(doc_dicts):

    with pytest.raises(NotImplementedError):
        plot.visualize_data(doc_dicts[0], spans_key=["task1", "sc"])

    # test NotImplementedException when not in Jupyter Notebook
    with pytest.raises(NotImplementedError):
        plot.visualize_data(doc_dicts[0])
