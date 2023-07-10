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

try:
    # german
    import de_core_news_sm
    # english
    import en_core_web_sm
    # french
    import fr_core_news_sm
    # italian
    import it_core_news_sm

except ImportError:
    logging.warning(
        "Required Spacy model was not found. Attempting to download it.."
    )
    # german
    spacy.cli.download("de_core_news_sm")
    import de_core_news_sm
    # english
    spacy.cli.download("en_core_web_sm")
    import en_core_web_sm
    # french
    spacy.cli.download("fr_core_news_sm")
    import fr_core_news_sm
    # italian
    spacy.cli.download("it_core_news_sm")
    import it_core_news_sm

pkg = importlib_resources.files("moralization")


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
    def cas_to_doc(cas, ts, language):
        """Transforms the cassis object into a spacy doc.
            Adds the paragraph and the different span categories to the doc object.
            Also maps the provided labels to a more user readable format.
        Args:
            cas (cassis.cas): The cassis object generated from the input files
            ts (typesystem): The provided typesytem

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
            "Forderung": "KAT5-Forderung_explizit",
            #       "KAT5Ausformulierung": "KAT5-Forderung implizit",
            #       "Kommentar": "KOMMENTAR",
        }

        supported_languages = ['english', 'french', 'german', 'italian']
        if language not in supported_languages:
            raise ValueError("Your language is not supported. It must be one of {}".format(supported_languages))
        
        if language == 'english':
            nlp = en_core_web_sm.load()
        elif language == 'french':
            nlp = fr_core_news_sm.load()
        elif language == 'german':
            nlp = de_core_news_sm.load()
        elif language == 'italian':
            nlp = it_core_news_sm.load()

        doc = nlp(cas.sofa_string)

        doc_train = nlp(cas.sofa_string)
        doc_test = nlp(cas.sofa_string)

        # add original cassis sentence as paragraph span
        sentence_type = ts.get_type(
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        )

        # initilize all span categories
        for doc_object in [doc, doc_train, doc_test]:
            doc_object.spans["sc"] = []
            doc_object.spans["paragraphs"] = []
            for cat in map_expressions.values():
                doc_object.spans[cat] = []

            paragraph_list = cas.select(sentence_type.name)
            for paragraph in paragraph_list:
                doc_object.spans["paragraphs"].append(
                    doc_object.char_span(
                        paragraph.begin,
                        paragraph.end,
                        label="paragraph",
                    )
                )
        span_type = ts.get_type("custom.Span")

        span_list = cas.select(span_type.name)

        doc, doc_train, doc_test = InputOutput._split_train_test(
            doc, doc_train, doc_test, span_list, map_expressions
        )

        return doc, doc_train, doc_test

    @staticmethod
    def _split_train_test(doc, doc_train, doc_test, span_list, map_expressions):
        # every n-th entry is put as a test value
        n_test = 5
        n_start = 0

        for span in span_list:
            for cat_old, cat_new in map_expressions.items():
                # not all of these categories have values in every span.
                if span[cat_old]:
                    # we need to attach each span category on its own, as well as all together in "sc"

                    char_span = doc.char_span(
                        span.begin,
                        span.end,
                        label=span[cat_old],
                    )
                    if char_span:
                        doc.spans[cat_new].append(char_span)
                        doc.spans["sc"].append(char_span)
                        n_start = n_start + 1

                        if n_start % n_test != 0:
                            char_span_train = doc_train.char_span(
                                span.begin,
                                span.end,
                                label=span[cat_old],
                            )
                            doc_train.spans[cat_new].append(char_span_train)
                            doc_train.spans["sc"].append(char_span_train)
                        else:
                            char_span_test = doc_test.char_span(
                                span.begin,
                                span.end,
                                label=span[cat_old],
                            )
                            doc_test.spans[cat_new].append(char_span_test)
                            doc_test.spans["sc"].append(char_span_test)

                    # char_span returns None when the given indices do not match a token begin and end.
                    # e.G ".Ich" instead of ". Ich"
                    elif char_span is None:
                        logging_warning = f"The char span for {span.get_covered_text()} ({span}) returned None.\n"
                        logging_warning += (
                            "It might be due to a mismatch between char indices. \n"
                        )
                        if logging.root.level > logging.DEBUG:
                            logging_warning += "Skipping span! Enable Debug Logging for more information."

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

                    # create test and train set:

        return doc, doc_train, doc_test

    @staticmethod
    def files_to_docs(data_files: List or str, ts: object, language: str):
        """

        Args:
          data_files: list or str:
          ts: object:

        Returns:

        """
        doc_dict = {}
        train_dict = {}
        test_dict = {}

        for file in data_files:
            logging.info(f"Reading ./{file}")
            try:
                cas, file_type = InputOutput.read_cas_file(file, ts)
                doc, doc_train, doc_test = InputOutput.cas_to_doc(cas, ts, language)
                doc_dict[file.stem] = doc
                train_dict[file.stem] = doc_train
                test_dict[file.stem] = doc_test

            except XMLSyntaxError as e:
                logging.warning(
                    f"WARNING: skipping file '{file}' due to XMLSyntaxError: {e}"
                )

        return doc_dict, train_dict, test_dict

    @staticmethod
    def _merge_span_categories(doc_dict, merge_dict=None):
        """Take the new_dict_cat dict and add its key as a main_cat to data_dict.
        The values are the total sub_dict_entries of the given list.

        Args:
          doc_dict(dict: doc): The provided doc dict.
          new_dict_cat(dict): map new category to list of existing_categories.

        Return:
            dict: The data_dict with new span categories.
        """
        if merge_dict is None:
            merge_dict = {
                "task1": ["KAT1-Moralisierendes Segment"],
                "task2": ["KAT2-Moralwerte", "KAT2-Subjektive Ausdrücke"],
                "task3": ["KAT3-Rolle", "KAT3-Gruppe", "KAT3-own/other"],
                "task4": ["KAT4-Kommunikative Funktion"],
                "task5": ["KAT5-Forderung explizit"],
            }

        for file in doc_dict.keys():
            # initilize new span_groups
            for cat in merge_dict.keys():
                doc_dict[file].spans[cat] = []

            for new_main_cat, new_cat_entries in merge_dict.items():
                if new_cat_entries == "all":
                    for main_cat in list(doc_dict[file].spans.keys()):
                        doc_dict[file].spans[new_main_cat].extend(
                            doc_dict[file].spans[main_cat]
                        )
                else:
                    for old_main_cat in new_cat_entries:
                        doc_dict[file].spans[new_main_cat].extend(
                            doc_dict[file].spans[old_main_cat]
                        )
        return doc_dict

    @staticmethod
    def read_data(dir: str, language: str):
        """Convenience method to handle input reading in one go.

        Args:
          dir: str: Path to the data directory.

        Returns:
            doc_dict: dict: Dictionary of with all the available data in one.
            train_dict: dict: Dictionary with only the spans that are used for training.
            test_dict: dict: Dictionary with only the spans that are used for testing.
        """
        data_files, ts_file = InputOutput.get_multiple_input(dir)
        # read in the ts
        ts = InputOutput.read_typesystem(ts_file)
        doc_dict, train_dict, test_dict = InputOutput.files_to_docs(data_files, ts, language)

        for dict_ in [doc_dict, train_dict, test_dict]:
            dict_ = InputOutput._merge_span_categories(dict_)

        return doc_dict, train_dict, test_dict
