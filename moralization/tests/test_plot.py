from moralization import plot
from moralization.analyse import _loop_over_files
import pytest
import seaborn as sns
from moralization.data_manager import DataManager


def test_report_occurrence_heatmap(doc_dict, monkeypatch):
    df = _loop_over_files(doc_dict)

    # test corr without filter
    corr_df = plot.report_occurrence_heatmap(df, _type="corr")
    all_columns = [
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
        ("KAT5-Forderung implizit", "implizit"),
        ("task1", "Moralisierung explizit"),
        ("task1", "Moralisierung interpretativ"),
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
    dm = DataManager(data_dir)

    test_interactive_analyzer = plot.InteractiveAnalyzerResults(
        dm.return_analyzer_result("all"), dm.return_categories()
    )
    span_key_list = sorted(
        [
            "KAT1-Moralisierendes Segment",
            "KAT2-Moralwerte",
            "KAT2-Subjektive Ausdrücke",
            "KAT3-Gruppe",
            "KAT3-Rolle",
            "KAT3-own/other",
            "KAT4-Kommunikative Funktion",
            "KAT5-Forderung explizit",
            "KAT5-Forderung implizit",
            "paragraphs",
            "sc",
            "task1",
        ]
    )

    for mode in [
        "frequency",
        "length",
        "span_distinctiveness",
        "boundary_distinctiveness",
    ]:
        # test dropdown
        assert (
            sorted(
                test_interactive_analyzer.change_analyzer_key(
                    mode, "KAT1-Moralisierendes Segment"
                )[0]
            )
            == span_key_list
        )
        assert (
            test_interactive_analyzer.change_analyzer_key(
                mode, "KAT1-Moralisierendes Segment"
            )[1]
            == "KAT1-Moralisierendes Segment"
        )

        # test graph
        test_interactive_analyzer.update_graph(mode, span_key_list)
        test_interactive_analyzer.update_graph(mode, "sc")

    with pytest.raises(KeyError):
        test_interactive_analyzer.change_analyzer_key(
            "bla", "KAT1-Moralisierendes Segment"
        )[0]
        test_interactive_analyzer.update_graph("bla", "sc")


def test_InteractiveCategoryPlot(data_dir):
    dm = DataManager(data_dir)
    span_key_list = sorted(
        [
            "KAT1-Moralisierendes Segment",
            "KAT2-Moralwerte",
            "KAT2-Subjektive Ausdrücke",
            "KAT3-Gruppe",
            "KAT3-Rolle",
            "KAT3-own/other",
            "KAT4-Kommunikative Funktion",
            "KAT5-Forderung explizit",
            "KAT5-Forderung implizit",
            "task1",
        ]
    )
    file_names = list(dm.doc_dict.keys())

    test_interactive_heatmap = plot.InteractiveCategoryPlot(dm)
    assert test_interactive_heatmap.update_filename(file_names) == (
        span_key_list,
        span_key_list[0],
    )
    assert test_interactive_heatmap.update_filename(file_names[0]) == (
        span_key_list,
        span_key_list[0],
    )
    assert test_interactive_heatmap.update_filename([]) == (
        [0],
        0,
    )

    assert test_interactive_heatmap.update_category(span_key_list[:2]) == (
        [
            {
                "label": "Moralisierung explizit",
                "value": "KAT1-Moralisierendes Segment___Moralisierung explizit",
            },
            {"label": "Care", "value": "KAT2-Moralwerte___Care"},
        ],
        [
            "KAT1-Moralisierendes Segment___Moralisierung explizit",
            "KAT2-Moralwerte___Care",
        ],
    )
    assert test_interactive_heatmap.update_category(span_key_list[0]) == (
        [
            {
                "label": "Moralisierung explizit",
                "value": "KAT1-Moralisierendes Segment___Moralisierung explizit",
            },
        ],
        [
            "KAT1-Moralisierendes Segment___Moralisierung explizit",
        ],
    )
    assert test_interactive_heatmap.update_category([]) == (
        ["please select a filename"],
        ["please select a filename"],
    )
    assert test_interactive_heatmap.update_category(0) == (
        ["please select a filename"],
        ["please select a filename"],
    )

    test_interactive_heatmap.update_subcat(
        test_interactive_heatmap.update_category(span_key_list[0])[1]
    )
    test_interactive_heatmap.update_subcat(
        test_interactive_heatmap.update_category(span_key_list[:2])[1]
    )


def test_InteractiveVisualization(data_dir):
    dm = DataManager(data_dir)
    test_interactive_vis = plot.InteractiveVisualization(dm)
    with pytest.raises(EnvironmentError):
        test_interactive_vis.run_app()
    with pytest.raises(EnvironmentError):
        test_interactive_vis.change_span_cat("KAT1-Moralisierendes Segment")
    with pytest.raises(EnvironmentError):
        test_interactive_vis.change_span_cat("")


def test_spacy_data_handler_visualize_data(doc_dict):
    with pytest.raises(EnvironmentError):
        plot.return_displacy_visualization(doc_dict, spans_key=["task1", "sc"])
    with pytest.raises(EnvironmentError):
        plot.return_displacy_visualization(doc_dict)
    with pytest.raises(EnvironmentError):
        plot.return_displacy_visualization(doc_dict, spans_key="task2")
