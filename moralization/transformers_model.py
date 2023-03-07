from moralization import spacy_model


class TransformersSetup:
    def __init__(self) -> None:
        pass

    def get_data_doc(self, data_dir):
        # initialize spacy
        test_setup = spacy_model.SpacySetup(data_dir)
        self.data_doc = test_setup.convert_data_to_spacy_doc()


if __name__ == "__main__":
    obj = TransformersSetup()
    obj.get_data_doc("data/All_Data/XMI_11")
