from collections import defaultdict
import pandas as pd
import numpy as np
import bisect
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import re

map_expressions = {
    "KAT1-Moralisierendes Segment": "KAT1MoralisierendesSegment",
    "KAT2-Moralwerte": "Moralwerte",
    "KAT2-Subjektive_Ausdrücke": "KAT2Subjektive_Ausdrcke",
    "KAT3-Gruppe": "Protagonistinnen2",
    "KAT3-Rolle": "Protagonistinnen",
    "KAT3-own/other": "Protagonistinnen3",
    "KAT4-Kommunikative Funktion": "KommunikativeFunktion",
    "KAT5-Forderung_explizit": "Forderung",
    "KAT5-Forderung_implizit": "KAT5Ausformulierung",
    "Kommentar": "KOMMENTAR",
}


# select all custom Spans and store them in an ordered dict,
# where the first dimension is the used inception category (Protagonistinnen, Forderung, etc...)
# and the second dimension is the corresponding value of this category ('Forderer:in', 'Adresassat:in', 'Benefizient:in')
# dict[category][entry value] = span
def sort_spans(cas: object, ts: object) -> defaultdict:

    span_type = ts.get_type("custom.Span")
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


def add_true_sentences(cas: object, ts: object) -> object:
    nlp = spacy.load("de_core_news_sm")
    text = cas.sofa_string
    sentences = [sentence for sentence in nlp(text).sents]
    # create new typesystem:
    try:
        True_sentence_type = ts.create_type(name="moralization.TrueSentence")
    except ValueError:
        True_sentence_type = ts.get_type("moralization.TrueSentence")

    for sentence in sentences:
        if (sentence.end - sentence.start) > 3:
            # spacy returns the token ids, not the character ids.
            # this is in missmatch with cassis.
            # to convert from token to character I search through the doc with a regex to find the correct ids.
            # the replace is to avoid regex errors in regards to brackets
            current_str = (
                str(sentence.doc[sentence.start : sentence.end])
                .replace("(", r"\(")
                .replace(")", r"\)")
            )
            sentence_span = [
                (m.start(0), m.end(0)) for m in re.finditer(current_str, text)
            ]
            for span in sentence_span:
                if (span[1] - span[0]) > 9:
                    cas.add_annotation(True_sentence_type(begin=span[0], end=span[1]))
    return cas, ts


def get_paragraphs(cas: object, ts: object) -> list:
    span_type = ts.get_type(
        "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    )
    paragraph_dict = defaultdict(list)
    for span in cas.select(span_type.name):
        paragraph_dict["span"].append((span.begin, span.end))
        paragraph_dict["sofa"].append(span.get_covered_text().replace("#", ""))

    return paragraph_dict


def get_sentences(cas: object, ts: object) -> list:
    span_type = ts.get_type("moralization.TrueSentence")
    sentence_dict = defaultdict(list)
    for span in cas.select(span_type.name):
        sentence_dict["span"].append((span.begin, span.end))
        sentence_dict["sofa"].append(span.get_covered_text().replace("#", ""))

    return sentence_dict


class AnalyseOccurence:
    """Contains statistical information methods about the data."""

    def __init__(
        self, data_dict: dict, mode: str = "instances", file_names: str = None
    ) -> None:
        self.mode = mode
        self.data_dict = data_dict
        self.mode_dict = {
            "instances": self.report_instances,
            "spans": self.report_spans,
        }
        self.file_names = self._initialize_files(file_names)
        self.instance_dict = self._initialize_dict()
        # call the analysis method
        self.mode_dict[self.mode]()

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
                self.file_names[0],
            ],
            ascending=False,
        )
        # fill NaN
        self.df = self.df.fillna(0)

    def report_instances(self):
        """Reports number of occurences of a category per text source."""
        # instances reports the number of occurences
        # filename: main_cat: sub_cat: instances
        for file_name in self.file_names:
            span_dict = self.data_dict[file_name]["data"]
            # initilize total instances rows for easier setting later.
            self.instance_dict[file_name][("total instances", "with invalid")] = 0
            self.instance_dict[file_name][("total instances", "without invalid")] = 0
            for main_cat_key, main_cat_value in span_dict.items():
                for sub_cat_key, sub_cat_value in main_cat_value.items():
                    # the tuple index makes it easy to convert the dict into a pandas dataframe
                    self.instance_dict[file_name][(main_cat_key, sub_cat_key)] = len(
                        sub_cat_value
                    )
        # initialize data frame
        self._initialize_df()
        # add rows for total instances
        self._add_total()
        # sort by index and occurence number
        self._clean_df()
        self.df = self.df[:].astype(int)

    def report_spans(self):
        """Reports spans of a category per text source."""
        # span reports the spans of the annotations in a list
        # this report_instances call makes it much easier to include the total number of spans
        # for each columns, as well as removes the need to duplicate the pandas setup.
        self.report_instances()
        self.df[:] = self.df[:].astype("object")
        for file_name in self.file_names:
            span_dict = self.data_dict[file_name]["data"]
            for main_cat_key, main_cat_value in span_dict.items():
                for sub_cat_key in main_cat_value.keys():
                    # report the beginning and end of each span as a tuple
                    span_list = [
                        (span["begin"], span["end"])
                        for span in span_dict[main_cat_key][sub_cat_key]
                    ]

                    # multiple options for how to report the spans are available
                    # first report the entire span object as a string
                    # span_list = [str(span) for span in span_dict[main_cat_key][sub_cat_key]]
                    # this would look like this:
                    # c.Span(Protagonistinnen=Forderer:in, Protagonistinnen2=Individuum, Protagonistinnen3=Own Group, begin=21822, end=21874);
                    # c.Span(Protagonistinnen=Benefizient:in, Protagonistinnen2=Institution, Protagonistinnen3=Own Group, begin=21974, end=21984);
                    # c.Span(Protagonistinnen=Forderer:in, Protagonistinnen2=Institution, Protagonistinnen3=Own Group, begin=66349, end=66352)
                    # maybe one should remove the c.Span() but i'm not sure what exactly is wanted here.
                    # second option is to report the end or beginning index for each span
                    # span_list=[str(span["end"]) for span in span_dict[main_cat_key][sub_cat_key] ]

                    # convert list to seperated str
                    # span_str = ";".join(span_list)
                    # span_str = span_str.replace("[", "").replace("]", "")

                    self.df.at[
                        (main_cat_key, sub_cat_key),
                        file_name,
                        # ] = span_str
                    ] = span_list


def _find_all_cat_in_sentence(data_dict, mode="sentences"):
    # mode can be sentences or paragraphs
    df_spans = AnalyseOccurence(data_dict, mode="spans").df

    # sentence, main_cat, sub_cat : occurence with the default value of 0 to allow adding of +1 at a later point.
    sentence_dict = defaultdict(lambda: defaultdict(lambda: 0))

    # iterate over the data_dict entries and the corresponding df columns with the span lists at the same time.
    for file_dict, df_file in zip(data_dict.values(), df_spans):
        # from the file dict we extract the sentence span start and end points as a list of tuples (eg [(23,45),(65,346)])
        # as well as the corresponding string
        sentence_span_list_per_file = file_dict[mode]["span"]
        sentence_str_list_per_file = file_dict[mode]["sofa"]
        # because the pandas multiindex is a tuple of (main_cat, sub_cat) for each subcat,#
        # we can now loop over each category pair in one loop instead of one for each index level.
        for main_cat_key, sub_cat_key in df_spans[df_file].index:
            # exclude the total instances columns as these are not needed here.
            if main_cat_key != "total instances":

                # if the type of the df cell is not a list it means there is no occurence of this category in the given file
                # this should only happen in the test dataset
                if isinstance(
                    df_spans[df_file].loc[[main_cat_key], [sub_cat_key]].values[0], list
                ):
                    # now we have a list of the span beginnings and endings for each category in a given file.
                    for occurence in (
                        df_spans[df_file].loc[[main_cat_key], [sub_cat_key]].values[0]
                    ):
                        # with bisect.bisect we can search for the index of the sentece in which the current category occurence falls.
                        sentence_idx = bisect.bisect(
                            sentence_span_list_per_file, occurence
                        )

                        # when we found a sentence index we can use this to add the sentence string to our dict and add +1 to the (main_cat_key, sub_cat_key) cell.
                        if sentence_idx > 0:
                            sentence_dict[sentence_str_list_per_file[sentence_idx - 1]][
                                (main_cat_key, sub_cat_key)
                            ] += 1

    # transform dict into multicolumn pd.DataFrame
    df_sentence_occurence = (
        pd.DataFrame(sentence_dict).fillna(0).sort_index(level=0).transpose()
    )
    df_sentence_occurence.index = df_sentence_occurence.index.set_names((["Sentence"]))
    df_sentence_occurence = df_sentence_occurence[:].astype(int)

    return df_sentence_occurence


def report_occurence_per_sentence(
    data_dict, filter_docs=None, mode="sentences"
) -> pd.DataFrame:
    """Returns a Pandas dataframe where each sentence is its own index
       and the column values are the occurences of the different categories.

    Args:
        data_dict (dict): the dict where all categories are stored.
        filter_docs (str, optional): The filenames for which to filter. Defaults to None.
        mode (str, optional): can report occurences based on `senteces` or `paragraphs`. Defaults to "sentences
    Returns:
        pd.DataFrame: Category occurences per sentence.
    """
    if filter_docs is not None:
        if not isinstance(filter_docs, list):
            filter_docs = [filter_docs]
        data_dict = {filter_doc: data_dict[filter_doc] for filter_doc in filter_docs}

    df_sentence_occurence = _find_all_cat_in_sentence(data_dict, mode=mode)
    return df_sentence_occurence


def _get_filter_multiindex(df: pd.DataFrame, filters):
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
        if filter in df.columns.levels[0]:
            [sub_cat_filter.append(key) for key in (df[filter].keys())]
        elif filter in df.columns.levels[1]:
            sub_cat_filter.append(filter)
        else:
            raise Warning(f"Filter key: {filter} not in dataframe columns.")

    return sub_cat_filter


def report_occurence_heatmap(df_sentence_occurence: pd.DataFrame, filter=None):
    """Returns the occurence heatmap for the given dataframe.
    Can also filter based on both main_cat and sub_cat keys.

    Args:
        df_sentence_occurence (pd.DataFrame): The sentence occurence dataframe.
        filter (str,list(str), optional): Filter values for the dataframe. Defaults to None.

    Returns:
        plt.figure : The heatmap figure.
    """

    df_sentence_occurence = df_sentence_occurence.copy()

    # df_sentence_occurence.columns = df_sentence_occurence.columns.droplevel()
    plt.figure(figsize=(16, 16))
    df_corr = report_occurence_matrix(df_sentence_occurence, filter=filter)

    heatmap = sns.heatmap(df_corr, cmap="cividis")
    return heatmap


def report_occurence_matrix(
    df_sentence_occurence: pd.DataFrame, filter=None
) -> pd.DataFrame:
    """Calculates the correlation matrix for the sentence occurence dataframe as well as handles its filtering.


    Args:
        df_sentence_occurence (pd.DataFrame): The sentence occurence dataframe.
        filter (str,list(str), optional): Filter values for the dataframe. Defaults to None.


    Returns:
        pd.DataFrame: Correlation matrix.
    """
    if filter is None:
        return df_sentence_occurence.corr().sort_index(level=0)
    else:
        filter = _get_filter_multiindex(df_sentence_occurence, filter)
        # Couldn't figure out how to easily select columns based on the second level column name.
        # So the df is transposed, the multiindex can be filterd using loc, and then transposed back to get the correct correlation matrix.
        return (
            df_sentence_occurence.T.loc[(slice(None), filter), :]
            .sort_index(level=0)
            .T.corr()
        )
