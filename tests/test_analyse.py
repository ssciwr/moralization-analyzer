from moralization import inout, analyse
import pathlib
import cassis

data_dir = pathlib.Path("../moralization_data/Test_Data/XMI_11/")
data_dict = inout.InputOutput.get_input_dir(data_dir)


def test_AnalyseOccurence():
    df_instances = analyse.AnalyseOccurence(data_dict, mode="instances").df
    df_spans = analyse.AnalyseOccurence(data_dict, mode="spans").df
    assert len(df_instances.loc["KAT2Subjektive_Ausdrcke"]) == 6
    assert len(df_spans.loc["KAT2Subjektive_Ausdrcke"]) == 6
