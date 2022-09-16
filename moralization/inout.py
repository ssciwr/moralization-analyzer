from cassis import load_typesystem, load_cas_from_xmi
import glob
import os
import importlib_resources
from moralization import analyse
import spacy
from spacy.tokens import DocBin
import random

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
    def read_typesystem(filename: str) -> object:
        # read in the file system types
        # file = pkg / "data" / "TypeSystem.xml"

        with open(filename, "rb") as f:
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
        ts_file = glob.glob(os.path.join(dir_path, "TypeSystem.xml"))
        if len(ts_file) == 1:
            ts = InputOutput.read_typesystem(ts_file[0])
        elif len(ts_file) == 0:
            print("No Typesystem found in given directory, trying default location")
            if not glob.glob(str(pkg / "data" / "TypeSystem.xml")):
                raise FileNotFoundError("No typesystem found in ", pkg / "data")
            ts = InputOutput.read_typesystem(pkg / "data" / "TypeSystem.xml")
        else:
            raise Warning("Multiple typesystems found. Please provide only one.")

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
                "sofa": cas.sofa_string,  # note: use .sofa_string not .get_sofa() as the latter removes \n and similar markers
            }
        return data_dict

    @staticmethod
    def prepare_spacy_dat(dir_path: str):
        data_dict = InputOutput.get_input_dir(dir_path)
        for file in data_dict.keys():

            nlp = spacy.blank("de")
            db_train = DocBin()
            db_dev = DocBin()

            doc_train = nlp(data_dict[file]["sofa"])
            doc_dev = nlp(data_dict[file]["sofa"])

            ents = []
            for main_cat_key, main_cat_value in data_dict[file]["data"].items():
                if main_cat_key != "KAT5Ausformulierung":
                    for sub_cat_label, sub_cat_span_list in main_cat_value.items():
                        for span in sub_cat_span_list:

                            spacy_span = doc_train.char_span(
                                span["begin"],
                                span["end"],
                                label=sub_cat_label,
                            )

                            ents.append(spacy_span)

        # split data in test and training
        random.shuffle(ents)
        ents_train = ents[: int(0.8 * len(ents))]
        print(f"len training: {len(ents_train)}")
        ents_test = ents[int(0.8 * len(ents)) :]
        print(f"len testing: {len(ents_test)}")

        # https://explosion.ai/blog/spancat
        # use spancat for multiple labels on the same token

        doc_train.spans["sc"] = ents_train
        db_train.add(doc_train)

        doc_dev.spans["sc"] = ents_test
        db_dev.add(doc_dev)

        db_train.to_disk("../data/Training/train.spacy")
        db_dev.to_disk("../data/Training/dev.spacy")


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
