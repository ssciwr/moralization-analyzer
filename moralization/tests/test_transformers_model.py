from moralization.transformers_model import TransformersSetup


def test_TransformersSetup(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_data_doc(data_dir)
    test_filenames = list(test_obj.data_doc.keys())
    reference_filenames = [
        "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW",
        "test_data-trimmed_version_of-Gerichtsurteile-neg-AW-neu-optimiert-BB",
    ]
    assert sorted(test_filenames) == sorted(reference_filenames)


def test_get_data_lists(data_dir):
    test_obj = TransformersSetup()
    test_obj.get_data_doc(data_dir)
    example_name = "test_data-trimmed_version_of-Interviews-pos-SH-neu-optimiert-AW"
    test_obj.get_data_lists(example_name=example_name)
    assert test_obj.label_list[0] == [0]
    assert test_obj.sentence_list[1][2] == "JUL.02661"
    assert test_obj.token_list[1][0].text == "T07"
