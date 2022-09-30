from collections import defaultdict
import pandas as pd
import numpy as np
import bisect
import seaborn as sns
import matplotlib.pyplot as plt

map_expressions = {
    "KAT1MoralisierendesSegment": "KAT1-Moralisierendes Segment",
    "Moralwerte": "KAT2-Moralwerte",
    "KAT2Subjektive_Ausdrcke": "KAT2-Subjektive Ausdrücke",
    "Protagonistinnen2": "KAT3-Gruppe",
    "Protagonistinnen": "KAT3-Rolle",
    "Protagonistinnen3": "KAT3-own/other",
    "KommunikativeFunktion": "KAT4-Kommunikative Funktion",
    "Forderung": "KAT5-Forderung explizit",
    "KAT5Ausformulierung": "KAT5-Forderung implizit",
    "Kommentar": "KOMMENTAR",
}


# select all custom Spans and store them in an ordered dict,
# where the first dimension is the used inception category (Protagonistinnen, Forderung, etc...)
# and the second dimension is the corresponding value of this category ('Forderer:in', 'Adresassat:in', 'Benefizient:in')
# dict[category][entry value] = span
def get_spans(cas: object, ts: object, span_str="custom.Span") -> defaultdict:

    span_type = ts.get_type(span_str)
    span_dict = defaultdict(lambda: defaultdict(list))

    # list of all interesting categories
    # as there are:
    # KAT1MoralisierendesSegment - Moralisierung explizit, Moralisierung Kontext, Moralisierung Weltwissen, Moralisierung interpretativ, Keine Moralisierung
    # Moralwerte - Care, Harm, Fairness, Cheating, …
    # KAT2Subjektive_Ausdrcke - Care, Harm, Fairness, Cheating, …
    # Protagonistinnen2 - Individuum, Menschen, Institution, Soziale Gruppe, OTHER
    # Protagonistinnen - Forderer:in, Adressat:in, Benefizient:in, Kein Bezug
    # Protagonistinnen3 - Own group, Other group, Neutral
    # KommunikativeFunktion - Darstellung, Appell, Expression, Beziehung, OTHER
    # Forderung - explizit
    # KAT5Ausformulierung - implizit
    # KOMMENTAR - tag for duplicates that should be excluded

    cat_list = [
        "KAT1MoralisierendesSegment",
        "Moralwerte",
        "KAT2Subjektive_Ausdrcke",
        "Protagonistinnen2",
        "Protagonistinnen",
        "Protagonistinnen3",
        "KommunikativeFunktion",
        "Forderung",
        # "KAT5Ausformulierung",
        # "KOMMENTAR",
    ]

    for span in cas.select(span_type.name):

        for cat in cat_list:
            # this excludes any unwanted datapoints
            # also ignore the ones with no moralization
            if (
                span[cat]
                and span["KOMMENTAR"] != "Dopplung"
                # and span[cat] != "Keine Moralisierung"
            ):
                # Here we could also exclude unnecessary information
                span_dict[cat][span[cat]].append(span)
    # for span_dict_key, span_dict_sub_kat in span_dict.items():
    #     print(f"{span_dict_key}: {[key for key in span_dict_sub_kat.keys()]}")
    return span_dict


def get_paragraphs(cas: object, ts: object) -> defaultdict:
    span_type = ts.get_type(
        "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    sentence_dict = defaultdict(list)
    for span in cas.select(span_type.name):
        sentence_dict["span"].append((span.begin, span.end))
        sentence_dict["sofa"].append(span.get_covered_text())
    return sentence_dict


class AnalyseOccurence:
    """Contains statistical information methods about the data."""

    def __init__(
        self,
        data_dict: dict,
        mode: str = "instances",
        file_names: str = None,
        mapping: bool = True,
    ) -> None:
        self.mode = mode
        self.data_dict = data_dict
        self.mapping = mapping
        self.mode_dict = {
            "instances": self.report_instances,
            "spans": self.report_spans,
        }
        self.file_names = self._initialize_files(file_names)
        self.instance_dict = self._initialize_dict()
        # call the analysis method
        self.mode_dict[self.mode]()
        # map the df columns to the expressions given
        # we skip this here for now if paragraph correlation is analyzed
        if self.mapping:
            self.map_categories()

    def _initialize_files(self, file_names: str) -> list:
        """Helper method to get file names in list."""
        # get the file names from the global dict of dicts
        if file_names is None:
            file_names = list(self.data_dict.keys())
        # or use the file names that were passed explicitly
        elif isinstance(file_names, str):
            file_names = [file_names]
        return file_names

    def _initialize_dict(self) -> defaultdict:
        """Helper method to initialize dict."""
        return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def _initialize_df(self):
        """Helper method to initialize data frame."""
        self.df = pd.DataFrame(self.instance_dict)
        self.df.index = self.df.index.set_names((["Main Category", "Sub Category"]))

    def _get_categories(self, span_dict, file_name):
        """Helper method to initialize a dict with the given main and sub categories."""
        for main_cat_key, main_cat_value in span_dict.items():
            for sub_cat_key, sub_cat_value in main_cat_value.items():
                # the tuple index makes it easy to convert the dict into a pandas dataframe
                self.instance_dict[file_name][(main_cat_key, sub_cat_key)] = len(
                    sub_cat_value
                )
        return self.instance_dict

    def _add_total(self):
        """Helper method to set additional headers in data frame."""
        self.df.loc[("total instances", "with invalid"), :] = self.df.sum(axis=0).values
        self.df.loc[("total instances", "without invalid"), :] = (
            self.df.loc[("total instances", "with invalid"), :].values
            - self.df.loc["KAT1MoralisierendesSegment", "Keine Moralisierung"].values
        )

    def _clean_df(self):
        """Helper method to sort data frame and clean up values."""
        self.df = self.df.sort_values(
            by=[
                "Main Category",
                "Sub Category",
                # self.file_names[0],
            ],
            ascending=True,
        )
        # fill NaN with 0 for instances or None for spans
        if self.mode == "instances":
            self.df = self.df.fillna(0)
        if self.mode == "spans":
            self.df = self.df.replace({np.nan: None})
            # remove quotes - not sure if this is necessary
            # self.df = self.df.applymap(lambda x: x.replace('"','') if isinstance(x, str) else x)

    def report_instances(self):
        """Reports number of occurences of a category per text source."""
        # instances reports the number of occurences
        # filename: main_cat: sub_cat: instances
        for file_name in self.file_names:
            span_dict = self.data_dict[file_name]["data"]
            # initilize total instances rows for easier setting later.
            # only for mode instances
            self.instance_dict[file_name][("total instances", "with invalid")] = 0
            self.instance_dict[file_name][("total instances", "without invalid")] = 0
            self.instance_dict = self._get_categories(span_dict, file_name)
        # initialize data frame
        self._initialize_df()
        # add rows for total instances
        # only do this for mode instances
        self._add_total()

    def report_spans(self):
        """Reports spans of a category per text source."""
        # span reports the spans of the annotations separated by separator-token
        self.instance_dict = self._get_categories(
            self.data_dict[self.file_names[0]]["data"], self.file_names[0]
        )
        self._initialize_df()
        self.df[:] = self.df[:].astype("object")
        for file_name in self.file_names:
            span_dict = self.data_dict[file_name]["data"]
            span_text = self.data_dict[file_name]["sofa"]
            for main_cat_key, main_cat_value in span_dict.items():
                for sub_cat_key in main_cat_value.keys():
                    # save the span begin and end character index for further analysis
                    # span_dict[main_cat_key][sub_cat_key] =
                    # find the text for each span
                    span_annotated_text = [
                        span_text[span["begin"] : span["end"]]
                        for span in span_dict[main_cat_key][sub_cat_key]
                    ]
                    # clean the spans from #
                    span_annotated_text = [
                        span.replace("#", "") for span in span_annotated_text
                    ]
                    # clean the spans from "
                    # span_annotated_text = [
                    #     span.replace('"', "") for span in span_annotated_text
                    # ]
                    # convert list to &-separated spans
                    span_annotated_text = " & ".join(span_annotated_text)
                    self.df.at[
                        (main_cat_key, sub_cat_key),
                        file_name,
                    ] = span_annotated_text

    def map_categories(self):
        self.df = self.df.rename(map_expressions)
        self._clean_df()


class AnalyseSpans:

    # TODO refactor complexity

    @staticmethod
    def _find_occurence(
        sentence_dict,
        span_annotated_tuples,
        sentence_span_list_per_file,
        sentence_str_list_per_file,
        main_cat_key,
        sub_cat_key,
    ):
        """Find occurence of category in a sentence."""
        for occurence in span_annotated_tuples:
            # with bisect.bisect we can search for the index of the sentece in which the current category occurence falls.
            sentence_idx = bisect.bisect(sentence_span_list_per_file, occurence)
            # when we found a sentence index we can use this to add the sentence string to our dict and add +1 to the (main_cat_key, sub_cat_key) cell.
            if sentence_idx > 0:
                sentence_dict[sentence_str_list_per_file[sentence_idx - 1]][
                    (main_cat_key, sub_cat_key)
                ] += 1

        return sentence_dict

    @staticmethod
    def _find_all_cat_in_paragraph(data_dict):

        # sentence, main_cat, sub_cat : occurence with the default value of 0 to allow adding of +1 at a later point.
        sentence_dict = defaultdict(lambda: defaultdict(lambda: 0))

        # iterate over the data_dict entries
        for file_dict in data_dict.values():
            # from the file dict we extract the sentence span start and end points as a list of tuples (eg [(23,45),(65,346)])
            # as well as the corresponding string
            sentence_span_list_per_file = file_dict["paragraph"]["span"]
            sentence_str_list_per_file = file_dict["paragraph"]["sofa"]
            for main_cat_key, main_cat_value in file_dict["data"].items():
                for sub_cat_key in main_cat_value.keys():
                    # find the beginning and end of each span as a tuple
                    span_annotated_tuples = [
                        (span["begin"], span["end"])
                        for span in file_dict["data"][main_cat_key][sub_cat_key]
                    ]
                    # if the type of span_annotated_tuples is not a list it means there is no occurence of this category in the given file
                    # this should only happen in the test dataset
                    if not isinstance(span_annotated_tuples, list):
                        continue

                    # now we have a list of the span beginnings and endings for each category in a given file.
                    sentence_dict = AnalyseSpans._find_occurence(
                        sentence_dict,
                        span_annotated_tuples,
                        sentence_span_list_per_file,
                        sentence_str_list_per_file,
                        main_cat_key,
                        sub_cat_key,
                    )

        # transform dict into multicolumn pd.DataFrame
        df_sentence_occurence = (
            pd.DataFrame(sentence_dict).fillna(0).sort_index(level=0).transpose()
        )
        df_sentence_occurence.index = df_sentence_occurence.index.set_names(
            (["Sentence"])
        )
        # map the category names to the updated ones
        df_sentence_occurence = df_sentence_occurence.rename(columns=map_expressions)

        return df_sentence_occurence

    @staticmethod
    def report_occurence_per_paragraph(data_dict, filter_docs=None) -> pd.DataFrame:
        """Returns a Pandas dataframe where each sentence is its own index
        and the column values are the occurences of the different categories.

        Args:
            data_dict (dict): the dict where all categories are stored.
            filter_docs (str, optional): The filenames for which to filter. Defaults to None.

        Returns:
            pd.DataFrame: Category occurences per sentence.
        """
        if filter_docs is not None:
            if not isinstance(filter_docs, list):
                filter_docs = [filter_docs]
            data_dict = {
                filter_doc: data_dict[filter_doc] for filter_doc in filter_docs
            }

        df_sentence_occurence = AnalyseSpans._find_all_cat_in_paragraph(data_dict)
        return df_sentence_occurence


class PlotSpans:
    @staticmethod
    def _get_filter_multiindex(df_sentence_occurence: pd.DataFrame, filters):
        """Search through the given filters and return all sub_cat_keys when a main_cat_key is given.

        Args:
            df (pd.Dataframe): The sentence occurence dataframe.
            filters (str, list(str)): Filter values for the dataframe.

        Raises:
            Warning: Filter not in dataframe columns
        Returns:
            list: the filter strings of only the sub_cat_keys
        """
        if not isinstance(filters, list):
            filters = [filters]
        sub_cat_filter = []
        for filter in filters:
            if filter in df_sentence_occurence.columns.levels[0]:
                [
                    sub_cat_filter.append(key)
                    for key in (df_sentence_occurence[filter].keys())
                ]
            elif filter in df_sentence_occurence.columns.levels[1]:
                sub_cat_filter.append(filter)
            else:
                raise Warning(f"Filter key: {filter} not in dataframe columns.")

        return sub_cat_filter

    @staticmethod
    def _generate_corr_df(
        df_sentence_occurence: pd.DataFrame, filter=None
    ) -> pd.DataFrame:
        if filter is None:
            return df_sentence_occurence.corr().sort_index(level=0)
        else:
            filter = PlotSpans._get_filter_multiindex(df_sentence_occurence, filter)
            # Couldn't figure out how to easily select columns based on the second level column name.
            # So the df is transposed, the multiindex can be filterd using loc, and then transposed back to get the correct correlation matrix.
            return (
                df_sentence_occurence.T.loc[(slice(None), filter), :]
                .sort_index(level=0)
                .T.corr()
            )

    @staticmethod
    def report_occurence_heatmap(df_sentence_occurence: pd.DataFrame, filter=None):
        """Returns the occurence heatmap for the given dataframe.
        Can also filter based on both main_cat and sub_cat keys.

        Args:
            df_sentence_occurence (pd.DataFrame): The sentence occurence dataframe.
            filter (str,list(str), optional): Filter values for the dataframe. Defaults to None.

        Returns:
            plt.figure : The heatmap figure.
        """

        # df_sentence_occurence.columns = df_sentence_occurence.columns.droplevel()
        plt.figure(figsize=(16, 16))
        df_corr = PlotSpans._generate_corr_df(df_sentence_occurence, filter=filter)

        heatmap = sns.heatmap(df_corr, cmap="cividis")
        return heatmap

    @staticmethod
    def report_occurence_matrix(
        df_sentence_occurence: pd.DataFrame, filter_vals=None
    ) -> pd.DataFrame:
        """
        Returns the correlation matrix in regards to the given filters.
        Args:
            filter_vals (str,list(str), optional): Filter values for the dataframe. Defaults to None.
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        return PlotSpans._generate_corr_df(df_sentence_occurence, filter_vals)
