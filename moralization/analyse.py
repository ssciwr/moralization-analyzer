from collections import defaultdict
import pandas as pd
import numpy as np
import bisect
import seaborn as sns
import matplotlib.pyplot as plt

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


def get_sentences(cas: object, ts: object) -> list:
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


# find the overlaying category for an second dimension cat name
def find_cat_from_str(cat_entry, span_dict):
    for span_dict_key, span_dict_sub_kat in span_dict.items():
        if cat_entry in span_dict_sub_kat.keys():
            return span_dict_key
    raise RuntimeError(f"Category '{cat_entry}' not found in dataset")


def find_all_cat_in_sentence(data_dict):
    df_spans = AnalyseOccurence(data_dict, mode="spans").df

    # sentence, main_cat, sub_cat : occurence
    sentence_dict = defaultdict(lambda: defaultdict(lambda: 0))

    for file_dict, df_file in zip(data_dict.values(), df_spans):
        sentence_span_list_per_file = file_dict["sentences"]["span"]
        sentence_str_list_per_file = file_dict["sentences"]["sofa"]
        for main_cat_key, sub_cat_key in df_spans[df_file].index:
            if main_cat_key != "total instances":
                # print(main_cat_key,sub_cat_key, type(df_spans[df_file].loc[[main_cat_key],[sub_cat_key]].values[0]) )
                if isinstance(
                    df_spans[df_file].loc[[main_cat_key], [sub_cat_key]].values[0], list
                ):
                    for occurence in (
                        df_spans[df_file].loc[[main_cat_key], [sub_cat_key]].values[0]
                    ):
                        sentence_idx = bisect.bisect(
                            sentence_span_list_per_file, occurence
                        )
                        if sentence_idx > 0:
                            sentence_dict[sentence_str_list_per_file[sentence_idx - 1]][
                                (main_cat_key, sub_cat_key)
                            ] += 1

    return sentence_dict


def report_occurence_per_sentence(data_dict) -> pd.DataFrame:
    sentence_dict = find_all_cat_in_sentence(data_dict)
    df_sentence_occurence = pd.DataFrame(sentence_dict).fillna(0).transpose()
    df_sentence_occurence.index = df_sentence_occurence.index.set_names((["Sentence"]))
    return df_sentence_occurence


def report_occurence_heatmap(df_sentence_occurence, type="Heatmap"):
    if type == "Heatmap":
        df_sentence_occurence = df_sentence_occurence.copy()
        df_sentence_occurence.columns = df_sentence_occurence.columns.droplevel()
        plt.figure(figsize=(16, 16))
        return sns.heatmap(df_sentence_occurence.corr(), cmap="cividis")
    elif type == "Numbers":
        return df_sentence_occurence.corr()


# get overlap%
# so far this only works on a span basis and not a sentence basis.
def get_overlap_percent(cat_1, cat_2, data_dict, file_name, ret_occ=False):
    o_cat1 = find_cat_from_str(cat_1, data_dict[file_name]["data"])
    o_cat2 = find_cat_from_str(cat_2, data_dict[file_name]["data"])
    occurence = 0
    total = 0
    for span in data_dict[file_name]["data"][o_cat1][cat_1]:
        total += 1
        if span[o_cat2] == cat_2:
            occurence += 1
    if ret_occ:
        return occurence, total
    else:
        return round(occurence / total, 7)


def get_percent_matrix(data_dict, file_name, cat_list=None):
    if cat_list is None:
        cat_list = []
        for span_dict_key, span_dict_sub_kat in data_dict[file_name]["data"].items():
            [cat_list.append(key) for key in span_dict_sub_kat.keys()]
    percent_matrix = np.zeros((len(cat_list), len(cat_list)))
    for i, cat1 in enumerate(cat_list):
        for j, cat2 in enumerate(cat_list):
            percent_matrix[i, j] = get_overlap_percent(cat1, cat2, data_dict, file_name)
    df = pd.DataFrame(percent_matrix, index=cat_list)
    df.columns = cat_list
    return df
