"""
Contains plotting functionality.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from moralization.utils import is_interactive
from spacy import displacy

from dash import dcc, html, Input, Output, State
from jupyter_dash import JupyterDash
import plotly.express as px


def report_occurrence_heatmap(
    occurence_df: pd.DataFrame, _filter=None, _type="heatmap"
):
    """Returns the occurrence heatmap for the given dataframe.
    Can also filter based on both main_cat and sub_cat keys.

    Args:
        df_sentence_occurrence(pd.DataFrame): The sentence occurrence dataframe.
        _filter(str, optional): Filter values for the dataframe. (Default value = None)
        _filter(str, optional): Filter values for the dataframe.
    Defaults to None.
        df_paragraph_occurrence: pd.DataFrame:

    Returns:
        plt.figure: The heatmap figure.

    """

    if _type not in ["corr", "heatmap"]:
        raise ValueError(
            f"_type argument can only be `corr` or `heatmap` but is {_type}"
        )

    df_corr = _generate_corr_df(occurence_df, _filter=_filter)

    if _type == "corr":
        return df_corr
    elif _type == "heatmap":
        plt.figure(figsize=(16, 16))
        heatmap = sns.heatmap(df_corr, cmap="cividis")
        return heatmap


def _get_filter_multiindex(occurence_df: pd.DataFrame, filters):
    """Search through the given filters and return all sub_cat_keys
    when a main_cat_key is given.

    Args:
        df(pd.Dataframe): The sentence occurrence dataframe.
        filters(str): Filter values for the dataframe.
        occurence_df: pd.DataFrame:

    Returns:
        list: the filter strings of only the sub_cat_keys

    """
    if not isinstance(filters, list):
        filters = [filters]
    filter_dict = {"main": [], "sub": []}

    for _filter in filters:
        # for main_cat_filter append all sub cats:
        if _filter in occurence_df.columns.levels[0]:
            filter_dict["main"].append(_filter)

        elif _filter in occurence_df.columns.levels[1]:
            filter_dict["sub"].append(_filter)

        else:
            raise KeyError(f"Filter key: `{ _filter}` not in dataframe columns.")

    if filter_dict["main"] == []:
        filter_dict["main"] = slice(None)
    if filter_dict["sub"] == []:
        filter_dict["sub"] = slice(None)

    return filter_dict


def _generate_corr_df(occurence_df: pd.DataFrame, _filter=None) -> pd.DataFrame:
    """

    Args:
      df_paragraph_occurrence: pd.DataFrame:
      _filter:  (Default value = None)

    Returns:

    """
    if _filter is None:
        return occurence_df.sort_index(level=0).corr()
    else:
        _filter = _get_filter_multiindex(occurence_df, _filter)
        # Couldn't figure out how to easily select columns based on the
        # second level column name.
        # So the df is transposed, the multiindex can be filterd using
        # loc, and then transposed back to get the correct correlation matrix.
        return (
            occurence_df.T.loc[(_filter["main"], _filter["sub"]), :]
            .sort_index(level=0)
            .T.corr()
        )


class InteractiveAnalyzerResults:
    """Interactive plotting tool for the DataAnalyzer in jupyter notebooks"""

    def __init__(self, analyzer_results_all):
        self.analyzer_results_all = analyzer_results_all

        self.app = JupyterDash("DataAnalyzer")
        self.app.layout = html.Div(
            children=[
                html.Div(
                    [
                        "Interactive analyser results",
                        dcc.Dropdown(
                            options=list(self.analyzer_results_all.keys()),
                            value=list(self.analyzer_results_all.keys())[0],
                            id="dropdown_analyzer_key",
                        ),
                        dcc.Dropdown(id="dropdown_analyzer_span_cat", multi=True),
                    ],
                    style={"width": "40%"},
                ),
                dcc.Graph(id="graph_output"),
            ]
        )

        self.app.callback(
            Output("dropdown_analyzer_span_cat", "options"),
            Output("dropdown_analyzer_span_cat", "value"),
            Input("dropdown_analyzer_key", "value"),
        )(self.change_analyzer_key)

        self.app.callback(
            Output("graph_output", "figure"),
            Input("dropdown_analyzer_key", "value"),
            Input("dropdown_analyzer_span_cat", "value"),
        )(self.update_graph)

    def change_analyzer_key(self, key_input):
        analyzer_span_cat = sorted(list(self.analyzer_results_all[key_input].keys()))
        return analyzer_span_cat, analyzer_span_cat[0]

    def update_graph(self, input_analyzer_key, input_analyzer_span_cat):
        analyzer_result = self.analyzer_results_all[input_analyzer_key][
            input_analyzer_span_cat
        ]
        fig = px.scatter(
            analyzer_result,
            labels={"value": input_analyzer_key, "variable": "chosen categories"},
        )
        return fig

    def show(self):
        """Display the interactive plot"""
        return self.app.run_server(
            debug=True,
            port=8051,
            mode="inline",
            use_reloader=False,
        )


class InteractiveCategoryPlot:
    """Interactive plotting class for use in Jupyter notebooks

    User selects the filename and categories to plot using GUI widgets.
    The displayed plot is then automatically updated.
    A custom plotting callback can be provided to customize the plot.


    """

    def __init__(self, data_manager):
        """
        Args:
            data_dict (_type_): _description_
            plot_callback (_type_, optional): _description_. Defaults to None.
            figsize (_type_, optional): _description_. Defaults to None.
        """
        self.data_manager = data_manager
        self.app = JupyterDash("Heatmap")
        self.app.layout = html.Div(
            [
                html.Div(
                    children=[
                        dcc.Dropdown(
                            options=list(data_manager.doc_dict.keys()),
                            value=list(data_manager.doc_dict.keys()),
                            id="dropdown_filenames",
                            multi=True,
                        ),
                        dcc.Dropdown(options=[], id="dropdown_span_cat", multi=True),
                    ],
                    id="div_dropdown",
                    style={"width": "40%", "display": "inline-block"},
                ),
                dcc.Checklist(
                    options=["sub_cat1", "sub_cat2"],
                    inline=True,
                    id="checklist_subcat",
                    style={"width": "40%"},
                ),
                html.Div(
                    children=[
                        dcc.Graph(id="graph_output"),
                    ],
                    id="div_output",
                    style={"width": "100%"},
                ),
            ]
        )

        # normal dash decorators don't work inside functions.
        self.app.callback(
            Output("dropdown_span_cat", "options"),
            Output("dropdown_span_cat", "value"),
            Input("dropdown_filenames", "value"),
        )(self.update_filename)

        self.app.callback(
            Output("checklist_subcat", "options"),
            Output("checklist_subcat", "value"),
            Input("dropdown_span_cat", "value"),
            prevent_initial_call=True,
        )(self.update_category)

        self.app.callback(
            Output("graph_output", "figure"),
            Input("checklist_subcat", "value"),
            prevent_initial_call=True,
        )(self.update_subcat)

    # end init

    def update_filename(self, input_files):
        if input_files == []:
            return [0], 0

        # raise Warning("something")
        self.table = self.data_manager.occurence_analysis(
            "table", file_filter=input_files
        )
        main_cat_list = sorted(list(set(self.table.T.index.get_level_values(0))))
        return main_cat_list, main_cat_list[0]

    def update_category(self, span_cats):
        if span_cats == 0:
            return ["please select a filename"], ["please select a filename"]

        if isinstance(span_cats, str):
            span_cats = [span_cats]
        main_sub_combination = []
        for span_cat in span_cats:
            for cat in self.table[span_cat].columns:
                main_sub_combination.append(
                    {"label": cat, "value": f"{span_cat}___{cat}"}
                )
        return main_sub_combination, [
            values["value"] for values in main_sub_combination
        ]

    def update_subcat(self, subcats):
        if subcats == []:
            return []

        # filtered_table = app.table_corr[span_cat].loc[subcats]
        multi_index = [subcat.split("___") for subcat in subcats]
        labels = [index[1] for index in multi_index]

        filtered_table = self.table[multi_index]
        filtered_corr = filtered_table.corr()
        fig = px.imshow(
            filtered_corr, x=labels, y=labels, zmin=-1, zmax=1, width=600, height=600
        )
        return fig

    def run_app(self):
        self.app.run_server(debug=True, mode="inline", use_reloader=False)


class InteractiveVisualization:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.app = JupyterDash("DataVisualizer")

        self.app.layout = html.Div(
            [
                "Interactive Visualization",
                dcc.Dropdown(["all", "train", "test"], value="all", id="dropdown_mode"),
                dcc.Dropdown(id="dropdown_span_cat"),
                dcc.Markdown(id="markdown_displacy", dangerously_allow_html=True),
            ]
        )

        self.app.callback(
            Output("dropdown_span_cat", "options"),
            Output("dropdown_span_cat", "value"),
            Input("dropdown_mode", "value"),
        )(self.change_mode)

        self.app.callback(
            Output("markdown_displacy", "children"),
            Input("dropdown_span_cat", "value"),
            State("dropdown_mode", "value"),
        )(self.change_span_cat)

    def change_mode(self, mode):
        span_cats = []

        for doc in self.data_manager.doc_dict.values():
            [span_cats.append(span_cat) for span_cat in list(doc.spans.keys())]

        span_cats = list(set(span_cats))
        return sorted(span_cats), "sc"

    def change_span_cat(self, span_cat, mode):
        html_doc = self.data_manager.visualize_data(_type=mode, spans_key=span_cat)
        html_doc = html_doc.replace("\n", " ")
        return html_doc

    def run_app(self):
        self.app.run_server(
            debug=True,
            port=8052,
            mode="inline",
            use_reloader=False,
        )


def visualize_data(doc_dict, style="span", spans_key="sc"):
    """Use the displacy class offered by spacy to visualize the current dataset.
        use SpacySetup.span_keys to show possible keys or use 'sc' for all.


    Args:
        doc_dict(dict: doc, optional): The doc dict that is to be visualized.
        display_type(str, optional, optional): Specify is only the trainings,
        the testing or all datapoints should be shown,options are: "all", "test" and "train". Defaults to "all"
        type: the visualization type given to displacy, available are "dep", "ent" and "span,
        defaults to "span".
        style:  (Default value = "span")

    Returns:
        Displacy.render
    """

    if isinstance(spans_key, list):
        raise NotImplementedError(
            "spacy does no support viewing multiple categories at once."
        )
        # we could manually add multiple categories to one span cat and display this new category.

    if spans_key != "sc":
        for doc in doc_dict.values():
            if spans_key not in list(doc.spans.keys()):
                raise ValueError(
                    f"""The provided key: `{spans_key}` is not valid.
                    Please use one of the following {list(doc.spans.keys())}"""
                )

    # check if spans_key is present in all docs

    if not is_interactive():
        raise NotImplementedError(
            "Please only use this function in a jupyter notebook for the time being."
        )

    return displacy.render(
        [doc for doc in doc_dict.values()],
        style=style,
        options={"spans_key": spans_key},
    )
