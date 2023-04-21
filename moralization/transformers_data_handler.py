from transformers import AutoTokenizer


class TransformersDataHandler:
    """Helper class to organize and prepare data for transformer models."""

    def get_data_lists(self, doc_dict, example_name=None):
        """Convert the data from doc object to lists. Required for transformers training.

        Set the lists of tokens and labels for transformers training, with a nested list of
        sentences and tokens, and an equally nested list of labels that are initiated to zero.

        Args:
            doc_dict (dict, required): The dictionary of doc objects for each data source.
            example_name (str, optional): A single data source should be selected for conversion. Defaults to None.

        """
        # for now just select one source
        if example_name is None:
            example_name = sorted(list(doc_dict.keys()))[0]
        self.sentence_list = [
            [token.text for token in sent] for sent in doc_dict[example_name].sents
        ]
        self.token_list = [
            [token for token in sent] for sent in doc_dict[example_name].sents
        ]
        # initialize nested label list to 0
        self.label_list = [[0 for _ in sent] for sent in doc_dict[example_name].sents]

    def generate_labels(self, doc_dict, example_name=None):
        """Generate the labels from the annotated tokens in one long list. Required for transformers training.

        Args:
            doc_dict (dict, required): The dictionary of doc objects for each data source.
            example_name (str, optional): A single data source should be selected for conversion. Defaults to None.

        """
        # for now just select one source
        if example_name is None:
            example_name = sorted(list(doc_dict.keys()))[0]
        # generate the labels based on the current list of tokens
        # now set all Moralisierung, Moralisierung Kontext,
        # Moralisierung explizit, Moralisierung interpretativ, Moralisierung Weltwissen to 1
        # here we actually need to select by task
        selected_labels = [
            "Moralisierung",
            "Moralisierung Kontext",
            "Moralisierung Weltwissen",
            "Moralisierung explizit",
            "Moralisierung interpretativ",
        ]
        # create a list as long as tokens
        # we need to do this for all the data, example
        self.labels = [0 for _ in doc_dict[example_name]]
        for span in doc_dict[example_name].spans["task1"]:
            if span.label_ in selected_labels:
                self.labels[span.start + 1 : span.end + 1] = [1] * (
                    span.end - span.start
                )
                # mark the beginning of a span with 2
                self.labels[span.start] = 2

    def structure_labels(self):
        """Structure the tokens from one long list into a nested list for sentences. Required for transformers training.

        Returns:
            sentence_list (list): A nested list of the tokens (nested by sentence).
            label_list (list): A nested list of the labels (nested by sentence).
        """
        # labels now needs to be structured the same way as label_list
        # set the label at beginning of sentence to 2 if it is 1
        # also punctuation is included in the moralization label - we
        # definitely need to set those labels to -100
        j = 0
        for sent_labels, sent_tokens in zip(self.label_list, self.token_list):
            for i in range(len(sent_labels)):
                sent_labels[i] = self.labels[j]
                if i == 0 and self.labels[j] == 1:
                    sent_labels[i] = 2
                if sent_tokens[i].is_punct:
                    sent_labels[i] = -100
                j = j + 1
        return self.sentence_list, self.label_list

    def init_tokenizer(self, model_name="bert-base-cased", kwargs=None):
        if kwargs is None:
            kwargs = {}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        except OSError:
            # here we also need more exceptions for no network etc
            raise OSError("Could not initiate tokenizer - please check your model name")
        if not self.tokenizer.is_fast:
            raise ValueError(
                "Please use a different model that provices a fast tokenizer"
            )
