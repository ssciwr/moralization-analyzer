"""
Contains statistical analysis.
"""
from collections import defaultdict
import pandas as pd
import numpy as np
import bisect
import pathlib

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


def validate_data_dict(data_dict):
    if not data_dict:
        raise ValueError("data_dict is empty")
    for data_file_name, data_file in data_dict.items():
        if not data_file:
            raise ValueError(f"The dict content under {data_file_name} is empty.")
        if not isinstance(data_file, dict):
            raise ValueError(
                f"The content of {data_file_name} is not a dict but {type(data_file)}."
            )

        validation_list = ["data", "file_type", "sofa", "paragraph"]
        missing_cats = []
        for category in validation_list:
            if category not in list(data_file.keys()):
                missing_cats.append(category)

        if missing_cats:
            raise ValueError(f"Data dict is missing categories: {missing_cats}")


# select all custom Spans and store them in an ordered dict,
# where the first dimension is the used inception category
# (Protagonistinnen, Forderung, etc...)
# and the second dimension is the corresponding value of this category
# ('Forderer:in', 'Adresassat:in', 'Benefizient:in')
# dict[category][entry value] = span
def get_spans(cas: object, ts: object, span_str="custom.Span") -> defaultdict:

    span_type = ts.get_type(span_str)
    span_dict = defaultdict(lambda: defaultdict(list))

    # list of all interesting categories
    # as there are:
    # KAT1MoralisierendesSegment - Moralisierung explizit, Moralisierung Kontext,
    # Moralisierung Weltwissen, Moralisierung interpretativ, Keine Moralisierung
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


def get_paragraphs(cas: object, ts: object, span_str=None) -> defaultdict:
    if span_str is None:
        span_type = ts.get_type(
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        )
    else:
        span_type = ts.get_type(span_str)
    paragraph_dict = defaultdict(list)
    for span in cas.select(span_type.name):
        paragraph_dict["span"].append((span.begin, span.end))
        paragraph_dict["sofa"].append(span.get_covered_text())
    return paragraph_dict


class AnalyseOccurrence:
    """Contains statistical information methods about the data."""

    def __init__(
        self,
        data_dict: dict,
        mode: str = "instances",
        file_names: str = None,
        mapping: bool = True,
    ) -> None:

        validate_data_dict(data_dict)

        self.mode = mode
        self.data_dict = data_dict
        self.mapping = mapping
        self.mode_dict = {
            "instances": self.report_instances,
            "spans": self.report_spans,
            "span_index": self.report_index,
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
                # the tuple index makes it easy to convert the dict into a pd dataframe
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

    def report_instances(self):
        """Reports number of occurrences of a category per text source."""
        # instances reports the number of occurrences
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

    def report_index(self):
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
                    self.df.at[
                        (main_cat_key, sub_cat_key),
                        file_name,
                    ] = span_list

    def map_categories(self):
        self.df = self.df.rename(map_expressions)
        self._clean_df()


class AnalyseSpans:

    # TODO refactor complexity

    @staticmethod
    def list_categories(mydict: dict) -> list:
        """Unravel the categories into a list of tuples."""
        mylist = []
        for main_cat_key, main_cat_value in mydict.items():
            for sub_cat_key in main_cat_value.keys():
                mylist.append((main_cat_key, sub_cat_key))
        return mylist

    @staticmethod
    def _find_occurrence(
        sentence_dict,
        span_annotated_tuples,
        sentence_span_list_per_file,
        sentence_str_list_per_file,
        main_cat_key,
        sub_cat_key,
    ):
        """Find occurrence of category in a sentence."""
        for occurrence in span_annotated_tuples:
            # with bisect.bisect we can search for the index of the
            # sentece in which the current category occurrence falls.
            sentence_idx = bisect.bisect(sentence_span_list_per_file, occurrence)
            # when we found a sentence index we can use this to add the sentence string
            # to our dict and add +1 to the (main_cat_key, sub_cat_key) cell.
            if sentence_idx > 0:
                sentence_dict[sentence_str_list_per_file[sentence_idx - 1]][
                    (main_cat_key, sub_cat_key)
                ] += 1

        return sentence_dict

    @staticmethod
    def _find_all_cat_in_paragraph(data_dict):

        # sentence, main_cat, sub_cat : occurrence with the default value of
        # 0 to allow adding of +1 at a later point.
        sentence_dict = defaultdict(lambda: defaultdict(lambda: 0))

        # iterate over the data_dict entries
        for file_dict in data_dict.values():
            # from the file dict we extract the sentence span start and end
            # points as a list of tuples (eg [(23,45),(65,346)])
            # as well as the corresponding string
            sentence_span_list_per_file = file_dict["paragraph"]["span"]
            sentence_str_list_per_file = file_dict["paragraph"]["sofa"]
            # get the main and sub category names
            category_names = AnalyseSpans.list_categories(file_dict["data"])
            for cat_tuple in category_names:

                # find the beginning and end of each span as a tuple
                span_annotated_tuples = [
                    (span["begin"], span["end"])
                    for span in file_dict["data"][cat_tuple[0]][cat_tuple[1]]
                ]
                # if the type of span_annotated_tuples is not a list it means
                # there is no occurrence of this category in the given file
                # this should only happen in the test dataset
                if not isinstance(span_annotated_tuples, list):
                    continue
                # now we have a list of the span beginnings and endings for
                # each category in a given file.
                sentence_dict = AnalyseSpans._find_occurrence(
                    sentence_dict,
                    span_annotated_tuples,
                    sentence_span_list_per_file,
                    sentence_str_list_per_file,
                    cat_tuple[0],
                    cat_tuple[1],
                )

        # transform dict into multicolumn pd.DataFrame
        df_sentence_occurrence = (
            pd.DataFrame(sentence_dict).fillna(0).sort_index(level=0).transpose()
        )
        df_sentence_occurrence.index = df_sentence_occurrence.index.set_names(
            (["Sentence"])
        )
        # map the category names to the updated ones
        df_sentence_occurrence = df_sentence_occurrence.rename(columns=map_expressions)

        return df_sentence_occurrence

    @staticmethod
    def report_occurrence_per_paragraph(data_dict, filter_docs=None) -> pd.DataFrame:
        """Returns a Pandas dataframe where each sentence is its own index
        and the column values are the occurrences of the different categories.

        Args:
            data_dict (dict): the dict where all categories are stored.
            filter_docs (str, optional): The filenames for which to filter.
            Defaults to None.

        Returns:
            pd.DataFrame: Category occurrences per sentence.
        """

        validate_data_dict(data_dict)

        if filter_docs is not None:
            if not isinstance(filter_docs, list):
                filter_docs = [filter_docs]

            # allows use of abs path or just filename with or without extension.
            for i, filter_doc in enumerate(filter_docs):
                filter_docs[i] = pathlib.PurePath(filter_doc).stem

            data_dict = {
                filter_doc: data_dict[filter_doc] for filter_doc in filter_docs
            }

        df_sentence_occurrence = AnalyseSpans._find_all_cat_in_paragraph(data_dict)
        return df_sentence_occurrence
