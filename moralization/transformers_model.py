from moralization import spacy_model


class TransformersSetup:
    def __init__(self) -> None:
        pass

    def get_data_doc(self, data_dir):
        # initialize spacy
        test_setup = spacy_model.SpacySetup(data_dir)
        self.data_doc = test_setup.convert_data_to_spacy_doc()

    def get_data_lists(self, example_name=None):
        # for now just select one source
        if example_name is None:
            example_name = sorted(list(self.data_doc.keys()))[0]
        self.sentence_list = [
            [token.text for token in sent]
            for sent in self.data_doc[example_name]["train"].sents
        ]
        self.token_list = [
            [token for token in sent]
            for sent in self.data_doc[example_name]["train"].sents
        ]
        # initialize nested label list to 0
        self.label_list = [
            [0 for _ in sent] for sent in self.data_doc[example_name]["train"].sents
        ]
        for i in range(0, 5):
            print(self.sentence_list[i])
            print(self.label_list[i])

    def generate_labels(self, example_name=None):
        # for now just select one source
        if example_name is None:
            example_name = sorted(list(self.data_doc.keys()))[0]
        # generate the labels based on the current list of tokens
        # now set all Moralisierung, Moralisierung Kontext,
        # Moralisierung explizit, Moralisierung interpretativ, Moralisierung Weltwissen to 1
        selected_labels = [
            "Moralisierung",
            "Moralisierung Kontext",
            "Moralisierung Weltwissen",
            "Moralisierung explizit",
            "Moralisierung interpretativ",
        ]
        # create a list as long as tokens
        self.labels = [0 for _ in self.data_doc[example_name]["train"]]
        for span in self.data_doc[example_name]["train"].spans["task1"]:
            if span.label_ in selected_labels:
                self.labels[span.start + 1 : span.end] = [1] * (span.end - span.start)
                # mark the beginning of a span with 2
                self.labels[span.start] = 2

    def structure_labels(self):
        # labels now needs to be structured the same way as label_list
        # set the label at beginning of sentence to 2 if it is 1
        # also punctuation is included in the moralization label - we
        # definitely need to set those labels to -100
        j = 0
        for m in range(len(self.label_list)):
            for i in range(len(self.label_list[m])):
                self.label_list[m][i] = self.labels[j]
                if i == 0 and self.labels[j] == 1:
                    self.label_list[m][i] = 2
                if self.token_list[m][i].is_punct:
                    self.label_list[m][i] = -100
                j = j + 1
        for i in range(0, 10):
            print(self.sentence_list[i])
            print(self.label_list[i])


if __name__ == "__main__":
    obj = TransformersSetup()
    obj.get_data_doc("data/All_Data/XMI_11")
    obj.get_data_lists()
