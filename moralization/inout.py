from cassis import load_typesystem, load_cas_from_xmi
import glob
import os
import importlib_resources
from moralization import analyse

pkg = importlib_resources.files("moralization")


class InputOutput:
    """Namespace class to handle input and output."""

    # this dict can be extended to contain more file formats
    input_type = {"xmi": load_cas_from_xmi}

    def __init__(self) -> None:
        None

    @staticmethod
    def get_file_type(filename):
        return filename.strip().split(".")[-1]

    @staticmethod
    def read_typesystem() -> object:
        # read in the file system types
        file = pkg / "data" / "TypeSystem.xml"
        with open(file, "rb") as f:
            ts = load_typesystem(f)
        return ts

    @staticmethod
    def get_input_file(filename: str) -> object:
        """Read in the input file. Currently only xmi file format."""
        ts = InputOutput.read_typesystem()
        file_type = InputOutput.get_file_type(filename)
        # read the actual data file
        with open(filename, "rb") as f:
            data = InputOutput.input_type[file_type](f, typesystem=ts)
        return data

    @staticmethod
    def get_input_dir(dir_path: str) -> dict:
        "Get a list of input files from a given directory. Currently only xmi files."
        ### load multiple files into a list of dictionaries
        ts = InputOutput.read_typesystem()
        data_files = glob.glob(os.path.join(dir_path, "*.xmi"))
        data_dict = {}
        for data_file in data_files:
            # get the file type dynamically
            file_type = InputOutput.get_file_type(data_file)
            # the wikipediadiskussionen file breaks as it has an invalid xmi charakter.
            # if data_file != "../data/Wikipediadiskussionen-neg-BD-neu-optimiert-CK.xmi":
            with open(data_file, "rb") as f:
                cas = InputOutput.input_type[file_type](f, typesystem=ts)
            data_dict[os.path.basename(data_file).split(".xmi")[0]] = {
                "data": analyse.sort_spans(cas, ts),
                "file_type": os.path.basename(data_file).split(".")[1],
            }
        return data_dict


if __name__ == "__main__":
    # data = InputOutput.get_input_file(
    # "moralization/data/Gerichtsurteile-pos-AW-neu-optimiert-BB.xmi"
    # )
    data_dict = InputOutput.get_input_dir("moralization/data/")
    df_instances = analyse.report_instances(data_dict)
    print(df_instances.head(10))
    # this df can now easily be filtered.
    # df_instances.loc["KAT2Subjektive_Ausdrcke"]
    # df_spans = analyse.report_spans(data_dict_list)
    # df_spans.head(10)
    # analyse.get_overlap_percent(
    # "Forderer:in", "Neutral", data_dict_list, "Gerichtsurteile-neg-AW-neu-optimiert-BB"
    #     )
