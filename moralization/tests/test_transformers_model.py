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
