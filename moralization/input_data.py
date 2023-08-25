"""
Module that handles input reading.
"""
from cassis import load_typesystem, load_cas_from_xmi, typesystem
import pathlib
import importlib_resources
import logging
from lxml.etree import XMLSyntaxError
import spacy
from typing import List


pkg = importlib_resources.files("moralization")


def spacy_load_model(language_model):
    """Load the model for the selected language,
    download the model if it is missing.

    Args:
        language_model (str): The language model that should be loaded
            (corresponding to the language of the corpus).

    Returns:
        spaCy nlp object used downstream to parse the input files."""
    try:
        nlp = spacy.load(language_model)
    except OSError:
        logging.warning(
            "Required spaCy model {} was not found. \
                        Attempting to download it..".format(
                language_model
            )
        )
        try:
            spacy.cli.download(language_model)
        except SystemExit:
            raise SystemExit(
                "Model {} could not be found - please check that you selected one of \
                             the models from spaCy: https://spacy.io/usage/models".format(
                    language_model
                )
            )
        nlp = spacy.load(language_model)
    return nlp


class InputOutput:
    """Namespace class to handle input and output."""

    # this dict can be extended to contain more file formats
    input_type = {"xmi": load_cas_from_xmi}

    @staticmethod
    def get_file_type(filename: str):
        """

        Args:
          filename: str:

        Returns:

        """
        return pathlib.Path(filename).suffix[1:]

    @staticmethod
    def read_typesystem(filename: str = None) -> object:
        """

        Args:
          filename: str:  (Default value = None)

        Returns:

        """
        if filename is None:
            filename = pkg / "data" / "TypeSystem.xml"
        # read in the file system types
        with open(filename, "rb") as f:
            ts = load_typesystem(f)

        try:
            # this type exists for every typesystem created by inception
            # otherwise a .xmi data file can be loaded as a typesystem
            # without raising an error.
            ts.get_type("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
            return ts

        except typesystem.TypeNotFoundError:
            raise Warning(f"No valid type system found at {filename}")

    @staticmethod
    def read_cas_file(filename: str, ts: object):
        """

        Args:
          filename: str:
          ts: object:

        Returns:

        """
        file_type = InputOutput.get_file_type(filename)
        with open(filename, "rb") as f:
            cas = InputOutput.input_type[file_type](f, typesystem=ts)
        return cas, file_type

    @staticmethod
    def get_multiple_input(dir: str) -> tuple:
        """
         Get a list of input files from a given directory. Currently only xmi files.
        Args:
          dir: str:

        Returns:

        """

        # load multiple files into a list
        dir_path = pathlib.Path(dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Path {dir_path} does not exist")
        # convert generator to list to check if dir is emtpy
        # currently only xmi but maybe can be extended
        # also sort the input files.
        data_files = sorted(list(dir_path.glob("*.xmi")))
        if not data_files:
            raise FileNotFoundError(f"No input files found in {dir_path}")

        # look for a ts file
        ts_files = list(dir_path.glob("TypeSystem.xml"))
        if len(ts_files) == 0:
            ts_files = None
        ts_file = ts_files[0] if ts_files is not None else ts_files
        return data_files, ts_file

    @staticmethod
    def cas_to_doc(cas, ts, language_model: str = "de_core_news_sm"):
        """Transforms the cassis object into a spacy doc.
            Adds the paragraph and the different span categories to the doc object.
            Also maps the provided labels to a more user readable format.
        Args:
            cas (cassis.cas): The cassis object generated from the input files
            ts (typesystem): The provided typesytem
            language_model (str, optional): Language model of the corpus that is being read.
                Defaults to "de_core_news_sm" (small German model).

        Returns:
            spacy.Doc: A doc object with all the annotation categories present.
        """
        # list and remap of all span categories
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
            #       "Kommentar": "KOMMENTAR",
        }

        nlp = spacy_load_model(language_model)
        doc = nlp(cas.sofa_string)
        # initalize the SpanGroup objects
        doc.spans["sc"] = []
        doc.spans["paragraphs"] = []
        for cat in map_expressions.values():
            doc.spans[cat] = []

        # now put the paragraphs (instances/segments) into the SpanGroup "paragraphs"
        # these are defined as cas sentences in the input
        sentence_type = ts.get_type(
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        )
        paragraph_list = cas.select(sentence_type.name)
        doc = InputOutput._get_paragraphs(doc, paragraph_list)

        # now put the different categories of the custom spans (ie Kat1, etc) into
        # SpanGroups
        span_type = ts.get_type("custom.Span")
        span_list = cas.select(span_type.name)
        # now assign the spans and labels in the doc object from the cas object
        doc = InputOutput._assign_span_labels(doc, span_list, map_expressions)
        return doc

    @staticmethod
    def _get_paragraphs(doc, paragraph_list):
        # add original cassis sentence as paragraph span
        for paragraph in paragraph_list:
            doc.spans["paragraphs"].append(
                doc.char_span(
                    paragraph.begin,
                    paragraph.end,
                    label="paragraph",
                )
            )
        return doc

    @staticmethod
    def _warn_empty_span(doc, span, cat_old):
        logging_warning = (
            f"The char span for {span.get_covered_text()} ({span}) returned None.\n"
        )
        logging_warning += "It might be due to a mismatch between char indices. \n"
        if logging.root.level > logging.DEBUG:
            logging_warning += (
                "Skipping span! Enable Debug Logging for more information."
            )
        logging.warning(logging_warning)
        logging.debug(
            f"""Token should be: \n \t'{span.get_covered_text()}', but is '{
                    doc.char_span(
                    span.begin,
                    span.end,
                    alignment_mode="expand",
                    label=span[cat_old],
                )}'\n"""
        )

    @staticmethod
    def _assign_span_labels(doc, span_list, map_expressions):
        # put the custom spans into the categories
        # we also need to delete "Moralisierung" and "Keine Moralisierung"
        labels_to_delete = ["Keine Moralisierung", "Moralisierung"]
        for span in span_list:
            for cat_old, cat_new in map_expressions.items():
                # not all of these categories have values in every span.
                if span[cat_old] and span[cat_old] not in labels_to_delete:
                    # we need to attach each span category on its own, as well as all together in "sc"

                    if cat_old == "KAT5Ausformulierung":
                        char_span = doc.char_span(
                            span.begin,
                            span.end,
                            label="implizit",
                        )

                    else:
                        char_span = doc.char_span(
                            span.begin,
                            span.end,
                            label=span[cat_old],
                        )
                    if char_span:
                        doc.spans[cat_new].append(char_span)
                        doc.spans["sc"].append(char_span)

                    # char_span returns None when the given indices do not match a token begin and end.
                    # e.G ".Ich" instead of ". Ich"
                    # The problem stems from a mismatch between spacy token beginnings and cassis token beginning.
                    # This might be due to the fact that spacy tokenizes on whitespace and cassis on punctuation.
                    # This leads to a mismatch between the indices of the tokens,
                    # where spacy sees ".Ich" as a single token
                    # cassis on the other hand returns only the indices for I and h as start and end point,
                    # thus spacy complains that the start ID is not actually the beginning of the token.
                    # We could fix this by trying reduce the index by 1 and check if the token is not complete.
                    # However this would give us some tokens that are not actually Words and
                    # thus are not useful for training.

                    # print a warning that this span cannot be used
                    elif char_span is None:
                        InputOutput._warn_empty_span(doc, span, cat_old)

        return doc

    @staticmethod
    def files_to_docs(
        data_files: List or str, ts: object, language_model: str = "de_core_news_sm"
    ):
        """

        Args:
          data_files: list or str:
          ts: object:
            language_model (str, optional): Language model of the corpus that is being read.
                Defaults to "de_core_news_sm" (small German model).

        Returns:

        """
        doc_dict = {}

        for file in data_files:
            logging.info(f"Reading ./{file}")
            try:
                cas, _ = InputOutput.read_cas_file(file, ts)
                doc = InputOutput.cas_to_doc(cas, ts, language_model)
                doc_dict[file.stem] = doc

            except XMLSyntaxError as e:
                logging.warning(
                    f"WARNING: skipping file '{file}' due to XMLSyntaxError: {e}"
                )

        return doc_dict

    @staticmethod
    def _merge_span_categories(doc_dict, merge_dict=None, task=None):
        """Take the new_dict_cat dict and add its key as a main_cat to data_dict.
        The values are the total sub_dict_entries of the given list.

        Args:
            doc_dict(dict: doc): The provided doc dict.
            merge_dict_cat(dict, optional): map new category to list of existing_categories.
                merge_dict = {
                    "task1": ["KAT1-Moralisierendes Segment"],
                    "task2": ["KAT2-Moralwerte", "KAT2-Subjektive Ausdrücke"],
                    "task3": ["KAT3-Rolle", "KAT3-Gruppe", "KAT3-own/other"],
                    "task4": ["KAT4-Kommunikative Funktion"],
                    "task5": ["KAT5-Forderung explizit",  "KAT5-Forderung implizit"],
                }
            Defaults to None.
            task (str, optional): The task from which the labels are selected.
            By default task 1 is selected. Default is None.
        Return:
            dict: The data_dict with new span categories.
        """
        if merge_dict is None:
            merge_dict = {
                "task1": ["KAT1-Moralisierendes Segment"],
                "task2": ["KAT2-Moralwerte", "KAT2-Subjektive Ausdrücke"],
                "task3": ["KAT3-Rolle", "KAT3-Gruppe", "KAT3-own/other"],
                "task4": ["KAT4-Kommunikative Funktion"],
                "task5": ["KAT5-Forderung explizit", "KAT5-Forderung implizit"],
            }
        if task is None:
            task = "task1"

        if task not in merge_dict.keys():
            raise KeyError(
                f"{task} not in merge_dict. Please provide a valid task or include the given task in the merge dict."
            )

        # now we only need to merge categories for the given task.
        merge_categories = merge_dict[task]

        for file in doc_dict.keys():
            # initilize new span_group
            doc_dict[file].spans[task] = []

            for old_main_cat in merge_categories:
                try:
                    doc_dict[file].spans[task].extend(
                        doc_dict[file].spans[old_main_cat]
                    )

                except KeyError:
                    raise KeyError(
                        f"{old_main_cat} not found in doc_dict[file].spans which"
                        + f" has {list(doc_dict[file].spans.keys())} as keys."
                    )
        return doc_dict

    @staticmethod
    def read_data(
        dir: str, language_model: str = "de_core_news_sm", merge_dict=None, task=None
    ):
        """Convenience method to handle input reading in one go.

        Args:
            dir (str): Path to the data directory.
            language_model (str, optional): Language model of the corpus that is being read.
                Defaults to "de_core_news_sm" (German).
            merge_dict_cat(dict, optional): map new category to list of existing_categories.
            task (str, optional): which task to use in the merge. Defaults to None.
        Returns:
            doc_dict (dict): Dictionary of with all the available data in one.
        """
        data_files, ts_file = InputOutput.get_multiple_input(dir)
        # read in the ts
        ts = InputOutput.read_typesystem(ts_file)
        doc_dict = InputOutput.files_to_docs(
            data_files, ts, language_model=language_model
        )
        doc_dict = InputOutput._merge_span_categories(doc_dict, merge_dict, task)
        return doc_dict
