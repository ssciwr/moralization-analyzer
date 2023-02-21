"""
Module that handles input reading.
"""
from cassis import load_typesystem, load_cas_from_xmi, typesystem
import pathlib
import importlib_resources
import logging
from moralization import analyse
from lxml.etree import XMLSyntaxError
import spacy

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
        data_files = list(dir_path.glob("*.xmi"))
        if not data_files:
            raise FileNotFoundError(f"No input files found in {dir_path}")

        # look for a ts file
        ts_files = list(dir_path.glob("TypeSystem.xml"))
        if len(ts_files) == 0:
            ts_files = None
        ts_file = ts_files[0] if ts_files is not None else ts_files
        return data_files, ts_file

    @staticmethod
    def cas_to_doc(cas, ts):
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
            "Forderung": "KAT5-Forderung explizit",
            #       "KAT5Ausformulierung": "KAT5-Forderung implizit",
            #       "Kommentar": "KOMMENTAR",
        }

        nlp = spacy.blank("de")
        doc = nlp(cas.sofa_string)
        # add original cassis sentence as paragraph span
        sentence_type = ts.get_type(
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        )

        # initilize all span categories
        doc.spans["sc"] = []
        doc.spans["paragraphs"] = []
        for cat in map_expressions.values():
            doc.spans[cat] = []

        paragraph_list = cas.select(sentence_type.name)
        for paragraph in paragraph_list:
            doc.spans["paragraphs"].append(
                doc.char_span(
                    paragraph.begin,
                    paragraph.end,
                    label="paragraph",
                )
            )
        span_type = ts.get_type("custom.Span")

        span_list = cas.select(span_type.name)
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
                    doc.spans[cat_new].append(char_span)
                    doc.spans["sc"].append(char_span)

        return doc

    @staticmethod
    def files_to_docs(data_files: list or str, ts: object):
        """

        Args:
          data_files: list or str:
          ts: object:

        Returns:

        """
        doc_dict = {}
        for file in data_files:
            try:
                cas, file_type = InputOutput.read_cas_file(file, ts)
                doc = InputOutput.cas_to_doc(cas, ts)
                doc_dict[file.stem] = doc
            except XMLSyntaxError as e:
                logging.warning(
                    f"WARNING: skipping file '{file}' due to XMLSyntaxError: {e}"
                )

        return doc_dict

    @staticmethod
    def read_data(dir: str):
        """Convenience method to handle input reading in one go.

        Args:
          dir: str:

        Returns:

        """
        data_files, ts_file = InputOutput.get_multiple_input(dir)
        # read in the ts
        ts = InputOutput.read_typesystem(ts_file)
        doc_dict = InputOutput.files_to_docs(data_files, ts)
        return doc_dict


if __name__ == "__main__":
    data_dict = InputOutput.read_data("data/Test_Data/XMI_11")
    # df_instances = analyse.AnalyseOccurrence(data_dict, mode="instances").df
    # df_instances.to_csv("instances_out.csv")
    # this df can now easily be filtered.
    # print(df_instances.loc["KAT2-Subjektive Ausdrücke"])
    # df_spans = analyse.AnalyseOccurrence(data_dict, mode="spans").df
    # df_spans.to_csv("spans_out.csv")
    #
    # analyse.get_overlap_percent(
    # "Forderer:in", "Neutral", data_dict, "Gerichtsurteile-neg-AW-neu-optimiert-BB"
    #     )
    df_sentences = analyse.AnalyseSpans.report_occurrence_per_paragraph(data_dict)
    print(df_sentences)
    df_sentences.to_csv("sentences_out.csv")
    print(df_sentences)
