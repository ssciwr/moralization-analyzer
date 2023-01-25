"""
Module that handles input reading.
"""
from cassis import load_typesystem, load_cas_from_xmi, typesystem
import pathlib
import importlib_resources
import logging
from moralization import analyse
from lxml.etree import XMLSyntaxError


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
    def add_custom_instance_to_ts(
        cas,
        ts,
        custom_span_type_name="custom.Span",
        custom_span_category="KAT1MoralisierendesSegment",
        new_span_type_name="moralization.instance",
    ):
        """Make a new annotation category from the spans in custom labeled span type.

        Args:
          cas(cassis.cas): The cas object
          ts(cassis.TypeSysten): The typesystem object
          custom_span_type_name(str, optional): The name of the span category
        to be used as a base. Defaults to "custom.Span".
          custom_span_category(str, optional): The label in the custom span
        category to be filtered for. Defaults to "KAT1MoralisierendesSegment".
          new_span_type_name(str, optional): The name of the new span category.
        Defaults to 'moralization.instance'.

        Returns:
          _type_: _description_

        """
        span_type = ts.get_type(custom_span_type_name)
        try:
            instance_type = ts.create_type(name=new_span_type_name)
            ts.create_feature(
                domainType=instance_type,
                name=custom_span_category,
                rangeType=str,
            )
        except ValueError:
            instance_type = ts.get_type(new_span_type_name)
        for span in cas.select(span_type.name):
            if (
                span[custom_span_category]
                and span[custom_span_category] != "Keine Moralisierung"
            ):
                cas.add(
                    instance_type(
                        begin=span.begin,
                        end=span.end,
                        KAT1MoralisierendesSegment=span[custom_span_category],
                    )
                )
        return cas, ts

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
    def read_cas_content(data_files: list or str, ts: object):
        """

        Args:
          data_files: list or str:
          ts: object:

        Returns:

        """
        data_dict = {}
        if not isinstance(data_files, list):
            data_files = [data_files]
        for data_file in data_files:
            try:
                cas, file_type = InputOutput.read_cas_file(data_file, ts=ts)
                cas, ts = InputOutput.add_custom_instance_to_ts(cas, ts)
                data_dict[data_file.stem] = {
                    "data": analyse.get_spans(cas, ts),
                    "file_type": file_type,
                    "sofa": cas.sofa_string,
                    # note: use .sofa_string not .get_sofa()
                    # as the latter removes \n and similar markers
                    "paragraph": analyse.get_paragraphs(
                        cas, ts, span_str="moralization.instance"
                    ),
                }
            except XMLSyntaxError as e:
                logging.warning(
                    f"WARNING: skipping file '{data_file}' due to XMLSyntaxError: {e}"
                )
        return data_dict

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
        data_dict = InputOutput.read_cas_content(data_files, ts)
        return data_dict


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
