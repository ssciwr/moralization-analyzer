from moralization import plot
import matplotlib
from moralization.analyse import _loop_over_files
import pytest
import seaborn as sns


def test_report_occurrence_heatmap(doc_dict, monkeypatch):
    df = _loop_over_files(doc_dict)

    # test corr without filter
    corr_df = plot.report_occurrence_heatmap(df, _type="corr")
    all_columns = [
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
        ("KAT1-Moralisierendes Segment", "Moralisierung interpretativ"),
        ("KAT2-Moralwerte", "Cheating"),
        ("KAT2-Moralwerte", "Oppression"),
        ("KAT2-Moralwerte", "Fairness"),
        ("KAT2-Moralwerte", "Liberty"),
        ("KAT2-Moralwerte", "Harm"),
        ("KAT2-Subjektive Ausdrücke", "Care"),
        ("KAT2-Subjektive Ausdrücke", "Harm"),
        ("KAT2-Subjektive Ausdrücke", "Liberty"),
        ("KAT3-Gruppe", "Menschen"),
        ("KAT3-Rolle", "Adresassat:in"),
        ("KAT3-Rolle", "Kein Bezug"),
        ("KAT3-own/other", "Own Group"),
        ("KAT3-own/other", "Other Group"),
    ]
    assert list(corr_df.columns) == all_columns

    # test corr wrong filter
    with pytest.raises(KeyError):
        corr_df = plot.report_occurrence_heatmap(df, _filter="test", _type="corr")

    # test corr with filter:
    corr_df = plot.report_occurrence_heatmap(df, _filter="KAT3-Gruppe", _type="corr")
    filtered_columns = [
        ("KAT3-Gruppe", "Individuum"),
        ("KAT3-Gruppe", "Institution"),
        ("KAT3-Gruppe", "Menschen"),
        ("KAT3-Gruppe", "soziale Gruppe"),
    ]
    assert list(corr_df.columns) == filtered_columns

    # test heatmap without filter
    monkeypatch.setattr(sns, "heatmap", lambda x, cmap: list(x.columns))
    heatmap_columns_full = plot.report_occurrence_heatmap(df, _type="heatmap")
    assert heatmap_columns_full == all_columns

    # test heatmap with filter
    heatmap_columns_filtered = plot.report_occurrence_heatmap(
        df, _filter="KAT3-Gruppe", _type="heatmap"
    )
    assert heatmap_columns_filtered == filtered_columns


def test_InteractiveCategoryPlot(doc_dict):
    df = _loop_over_files(doc_dict)
    filenames = list(doc_dict.keys())

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
