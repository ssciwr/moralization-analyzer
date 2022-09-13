from cassis import load_typesystem, load_cas_from_xmi
from collections import defaultdict
import glob
import os
import importlib_resources

pkg = importlib_resources.files("moralization")


class InputOutput:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_input(filename: str) -> object:
        """Read in the input file. Currently only xmi file format."""
        # this dict can be extended to contain more file formats
        input_type = {"xmi": load_cas_from_xmi}
        file_ending = filename.strip().split(".")[-1]
        # read in the file system types
        file = pkg / "data" / "TypeSystem.xml"
        with open(file, "rb") as f:
            ts = load_typesystem(f)
        # read the actual data file
        with open(filename, "rb") as f:
            data = input_type[file_ending](f, typesystem=ts)
        return data


if __name__ == "__main__":
    data = InputOutput.get_input(
        "moralization/data/Gerichtsurteile-pos-AW-neu-optimiert-BB.xmi"
    )
