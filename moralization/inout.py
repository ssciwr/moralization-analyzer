import re
from cassis import load_typesystem, load_cas_from_xmi
from collections import defaultdict
import glob
import os
import importlib_resources

pkg = importlib_resources.files("moralization")


class InputOutput:
    def __init__(self) -> None:
        None

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
        # this dict can be extended to contain more file formats
        input_type = {"xmi": load_cas_from_xmi}
        file_ending = filename.strip().split(".")[-1]
        ts = InputOutput.read_typesystem()
        # read the actual data file
        with open(filename, "rb") as f:
            data = input_type[file_ending](f, typesystem=ts)
        return data

    @staticmethod
    def get_input_dir(dir_path: str) -> list:
        "Get a list of input files from a given directory. Only xmi files."
        ### load multiple files into a list of dictionaries
        ts = InputOutput.read_typesystem()
        data_files = glob.glob(os.path.join(dir_path, "*.xmi"))
        data_dict_list = {}
        for data_file in data_files:
            # the wikipediadiskussionen file breaks as it has an invalid xmi charakter.
            # if data_file != "../data/Wikipediadiskussionen-neg-BD-neu-optimiert-CK.xmi":
            with open(data_file, "rb") as f:
                cas = load_cas_from_xmi(f, typesystem=ts)
            data_dict_list[os.path.basename(data_file).split(".xmi")[0]] = {
                # "data": sort_spans(cas, ts),
                "file_type": os.path.basename(data_file).split(".")[1],
            }
        return data_dict_list


if __name__ == "__main__":
    data = InputOutput.get_input_file(
        "moralization/data/Gerichtsurteile-pos-AW-neu-optimiert-BB.xmi"
    )
    data_dict_list = InputOutput.get_input_dir("moralization/data/")
