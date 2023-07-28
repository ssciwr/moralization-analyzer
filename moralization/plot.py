"""
Contains plotting functionality.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spacy import displacy
from dash import dcc, html, Input, Output, State
from jupyter_dash import JupyterDash
import plotly.express as px

from moralization.utils import is_interactive


def report_occurrence_heatmap(
    occurrence_df: pd.DataFrame, _filter=None, _type="heatmap"
):
    """Returns the occurrence heatmap or correlation dataframe for the given dataframe.
    Can also filter based on both main_cat and sub_cat keys.

    Args:
        occurrence_df (pd.DataFrame): The sentence occurrence dataframe.
        _filter (str, optional): Filter values for the dataframe. (Default value = None)
        _type (str, optional): Type of output to generate. It can be 'heatmap' or 'corr'.
            Defaults to 'heatmap'.

    Raises:
        ValueError: If the _type argument is not 'corr' or 'heatmap'.

    Returns:
        Union[pd.DataFrame, plt.figure]: The heatmap figure or the correlation dataframe.

    """
    if _type not in ["corr", "heatmap"]:
        raise ValueError(
            f"_type argument can only be `corr` or `heatmap` but is {_type}"
        )

    df_corr = _generate_corr_df(occurrence_df, _filter=_filter)

    if _type == "corr":
        return df_corr
    elif _type == "heatmap":
        plt.figure(figsize=(16, 16))
        heatmap = sns.heatmap(df_corr, cmap="cividis")
        return heatmap


def _get_filter_multiindex(occurrence_df: pd.DataFrame, filters):
    """Search through the given filters and return all sub_cat_keys
    when a main_cat_key is given.

    Args:
        df(pd.Dataframe): The sentence occurrence dataframe.
        filters(str): Filter values for the dataframe.
        occurrence_df: pd.DataFrame:

    Returns:
        list: the filter strings of only the sub_cat_keys

    """
    if not isinstance(filters, list):
        filters = [filters]
    filter_dict = {"main": [], "sub": []}

    for _filter in filters:
        # for main_cat_filter append all sub cats:
        if _filter in occurrence_df.columns.levels[0]:
            filter_dict["main"].append(_filter)

        elif _filter in occurrence_df.columns.levels[1]:
            filter_dict["sub"].append(_filter)

        else:
            raise KeyError(f"Filter key: `{ _filter}` not in dataframe columns.")

    if filter_dict["main"] == []:
        filter_dict["main"] = slice(None)
    if filter_dict["sub"] == []:
        filter_dict["sub"] = slice(None)

    return filter_dict


def _generate_corr_df(occurrence_df: pd.DataFrame, _filter=None) -> pd.DataFrame:
    """

    Args:
      df_paragraph_occurrence: pd.DataFrame:
      _filter:  (Default value = None)

    Returns:

    """
    if _filter is None:
        return occurrence_df.sort_index(level=0).corr()
    else:
        _filter = _get_filter_multiindex(occurrence_df, _filter)
        # Couldn't figure out how to easily select columns based on the
        # second level column name.
        # So the df is transposed, the multiindex can be filterd using
        # loc, and then transposed back to get the correct correlation matrix.
        return (
            occurrence_df.T.loc[(_filter["main"], _filter["sub"]), :]
            .sort_index(level=0)
            .T.corr()
        )


class InteractiveAnalyzerResults:
    """Interactive plotting tool for the DataAnalyzer in jupyter notebooks"""

    def __init__(self, analyzer_results_all):
        """
        Initializes the InteractiveAnalyzerResults class.

        Args:
            analyzer_results_all: A dictionary containing the analyzer results for all categories and spans.
        """
        self.analyzer_results_all = analyzer_results_all

        self.app = JupyterDash("DataAnalyzer")
        self.app.layout = html.Div(
            children=[
                html.Div(
                    [
                        "Interactive analyzer results",
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
        """
        Changes the analyzer key in response to a user selection.

        Args:
            key_input: The user-selected analyzer key.

        Returns:
            A tuple containing the analyzer span categories and the first category in the list.
        """
        analyzer_span_cat = sorted(list(self.analyzer_results_all[key_input].keys()))
        return analyzer_span_cat, analyzer_span_cat[0]

    def update_graph(self, input_analyzer_key, input_analyzer_span_cat):
        """
        Updates the graph in response to a user selection.

        Args:
            input_analyzer_key: The user-selected analyzer key.
            input_analyzer_span_cat: The user-selected analyzer span category.

        Returns:
            A plotly figure.
        """
        analyzer_result = self.analyzer_results_all[input_analyzer_key][
            input_analyzer_span_cat
        ]
        fig = px.scatter(
            analyzer_result,
            labels={"value": input_analyzer_key, "variable": "chosen categories"},
        )
        return fig

    def run_app(self, port=8053):
        """
        Displays the interactive plot.
        """
        if not is_interactive():
            raise EnvironmentError(
                "Dash GUI is only available in an Ipython environment like Jupyter notebooks."
            )

        return self.app.run_server(
            debug=True,
            port=port,
            mode="inline",
            use_reloader=False,
        )


class InteractiveCategoryPlot:
    """Interactive plotting class for use in Jupyter notebooks.

    User selects the filename and categories to plot using GUI widgets.
    The displayed plot is then automatically updated.
    A custom plotting callback can be provided to customize the plot.

    Attributes:
        data_manager (DataManager): The DataManager object containing the data.
        app (JupyterDash): The JupyterDash application.
    """

    def __init__(self, data_manager):
        """Initialize the InteractiveCategoryPlot object.

        Args:
            data_manager (DataManager): The DataManager object containing the data.
        """
        self.data_manager = data_manager
        self.app = JupyterDash("Heatmap")
        self.app.layout = html.Div(
            [
                # Dropdown for selecting filenames
                html.Div(
                    children=[
                        dcc.Dropdown(
                            options=list(data_manager.doc_dict.keys()),
                            value=list(data_manager.doc_dict.keys()),
                            id="dropdown_filenames",
                            multi=True,
                        ),
                        # Dropdown for selecting main categories
                        dcc.Dropdown(options=[], id="dropdown_span_cat", multi=True),
                    ],
                    id="div_dropdown",
                    style={"width": "40%", "display": "inline-block"},
                ),
                # Checklist for selecting subcategories
                dcc.Checklist(
                    options=["sub_cat1", "sub_cat2"],
                    inline=True,
                    id="checklist_subcat",
                    style={"width": "40%"},
                ),
                # Graph output for displaying the heatmap
                html.Div(
                    children=[
                        dcc.Graph(id="graph_output"),
                    ],
                    id="div_output",
                    style={"width": "100%"},
                ),
            ]
        )

        # Dash callbacks for updating the GUI components
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

    def update_filename(self, input_files):
        """Update the dropdown options for selecting main categories.

        Args:
            input_files (list): The list of selected filenames.

        Returns:
            tuple: The main category list and the default main category.
        """
        if input_files == []:
            return [0], 0

        self.table = self.data_manager.occurrence_analysis(
            "table", file_filter=input_files
        )
        main_cat_list = sorted(list(set(self.table.T.index.get_level_values(0))))
        return main_cat_list, main_cat_list[0]

    def update_category(self, span_cats):
        """Updates the categories available for selection in the Dash app.

        Args:
            span_cats (list or str): List of span categories or a single span category.

        Returns:
            tuple: A tuple containing two lists. The first list contains a dictionary of
                main_category-sub_category combinations
                as 'label' and 'value' respectively.
                The second list contains the values of the 'value' key in the first list.
        """
        # Check if span_cats is empty or None
        if span_cats == 0 or span_cats is None or span_cats == []:
            return ["please select a filename"], ["please select a filename"]

        # Check if span_cats is a single string, and convert to list if necessary
        if isinstance(span_cats, str):
            span_cats = [span_cats]

        # Generate a list of main category-sub category combinations as a list of dictionaries
        main_sub_combination = []
        for span_cat in span_cats:
            for cat in self.table[span_cat].columns:
                # Add a dictionary with 'label' and 'value' keys to the list
                main_sub_combination.append(
                    {"label": cat, "value": f"{span_cat}___{cat}"}
                )

        # Return a tuple containing the main_sub_combination list and a list of its 'value' keys
        return main_sub_combination, [
            values["value"] for values in main_sub_combination
        ]

    def update_subcat(self, subcats):
        """Generates a correlation heatmap for selected subcategories of a given category.

        Args:
            subcats (list): List of subcategories selected by the user.

        Returns:
            plotly.graph_objs._figure.Figure: A correlation heatmap showing the correlation
            between the selected subcategories.
        """
        # Check if subcats is empty
        if subcats == []:
            return []

        # Split the subcategories into main category and subcategory labels
        multi_index = [subcat.split("___") for subcat in subcats]
        labels = [index[1] for index in multi_index]

        # Filter the table based on the selected subcategories and calculate the correlation
        filtered_table = self.table[multi_index]
        filtered_corr = filtered_table.corr()

        # Generate a correlation heatmap using Plotly
        fig = px.imshow(
            filtered_corr, x=labels, y=labels, zmin=-1, zmax=1, width=600, height=600
        )

        # Return the correlation heatmap
        return fig

    def run_app(self, port=8051):
        """Runs the Dash app with the specified settings."""
        if not is_interactive():
            raise EnvironmentError(
                "Dash GUI is only available in an Ipython environment like Jupyter notebooks."
            )

        self.app.run_server(debug=True, mode="inline", use_reloader=False, port=port)


class InteractiveVisualization:
    def __init__(self, data_manager):
        """
        Initializes InteractiveVisualization with a DataManager instance and creates the JupyterDash app.

        Args:
            data_manager (DataManager): An instance of DataManager.
        """
        self.data_manager = data_manager
        self.app = JupyterDash("DataVisualizer")

        # Define the layout of the app
        self.app.layout = html.Div(
            [
                "Interactive Visualization",
                dcc.Dropdown(["all", "train", "test"], value="all", id="dropdown_mode"),
                dcc.Dropdown(id="dropdown_span_cat"),
                dcc.Markdown(id="markdown_displacy", dangerously_allow_html=True),
            ]
        )

        # Define the callback to update the dropdown_span_cat options and default value based on the selected mode
        self.app.callback(
            Output("dropdown_span_cat", "options"),
            Output("dropdown_span_cat", "value"),
            Input("dropdown_mode", "value"),
        )(self.change_mode)

        # Define the callback to update the visualization based on the selected span category and mode
        self.app.callback(
            Output("markdown_displacy", "children"),
            Input("dropdown_span_cat", "value"),
            State("dropdown_mode", "value"),
        )(self.change_span_cat)

    def change_mode(self, mode):
        """
        Changes the mode of the visualization.

        This method retrieves the span categories for the selected mode and returns them as
        options for the dropdown_span_cat dropdown.

        Args:
            mode (str): The selected mode of the visualization.

        Returns:
            A tuple containing a list of span categories and the default value for the dropdown_span_cat.
        """
        span_cats = []

        # Retrieve the span categories for the selected mode
        for doc in self.data_manager.doc_dict.values():
            [span_cats.append(span_cat) for span_cat in list(doc.spans.keys())]

        span_cats = list(set(span_cats))
        return sorted(span_cats), "sc"

    def change_span_cat(self, span_cat, mode):
        """
        Changes the selected span category.

        This method visualizes the selected span category for the selected mode and returns the
        visualized data as an HTML document.

        Args:
            span_cat (str): The selected span category.
            mode (str): The selected mode of the visualization.

        Returns:
            The visualized data as an HTML document.
        """
        # Visualize the selected span category for the selected mode
        html_doc = self.data_manager.visualize_data(_type=mode, spans_key=span_cat)
        html_doc = html_doc.replace("\n", " ")
        return html_doc

    def run_app(self, port=8052):
        if not is_interactive():
            raise EnvironmentError(
                "Dash GUI is only available in an Ipython environment like Jupyter notebooks."
            )

        """Runs the JupyterDash application."""
        self.app.run_server(
            debug=True,
            port=port,
            mode="inline",
            use_reloader=False,
        )


def visualize_data(doc_dict, style="span", spans_key="sc"):
    """Use the displacy class offered by spacy to visualize the current dataset.
        use SpacySetup.span_keys to show possible keys or use 'sc' for all.


    Args:
        doc_dict (dict): A dictionary of Spacy Doc objects.
        style (str, optional): The visualization type given to `displacy`. Available options
            are "dep", "ent", and "span". Defaults to "span".
        spans_key (str, optional): The key of the span category that should be visualized. If
            set to "sc", all span categories in the Spacy Doc objects will be visualized.
            Defaults to "sc".
    Raises:

        ValueError: Raised if `spans_key` is not a valid span category in any of the Spacy
            Doc objects in `doc_dict`.

    Returns:
        displacy.render: Renders a visualization of the Spacy Doc objects.
    """

    if isinstance(spans_key, list):
        # `displacy` does not support viewing multiple categories at once, so we raise an
        # error if a list is passed in `spans_key`.
        raise NotImplementedError("Multiple categories cannot be viewed at once.")

    # If `spans_key` is set to a specific span category, we check if it is a valid category
    # in any of the Spacy Doc objects in `doc_dict`.
    if spans_key != "sc":
        for doc in doc_dict.values():
            if spans_key not in list(doc.spans.keys()):
                raise ValueError(
                    f"The provided key: `{spans_key}` is not valid. "
                    f"Please use one of the following: {list(doc.spans.keys())}"
                )

    # Finally, we call `displacy.render` with the `doc_dict` values and the specified
    # options.
    return displacy.render(
        [doc for doc in doc_dict.values()],
        style=style,
        options={"spans_key": spans_key},
    )
