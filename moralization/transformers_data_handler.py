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
        """Initialize the tokenizer that goes along with the selected model.
        Only fast tokenizers can be used.

        Args:
            model_name (str, required): The name of the model that will be used for training.
            kwargs (dict, optional): Keyword arguments to pass to the tokenizer.
        """
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

    def tokenize(self, wordlist: list = None):
        if wordlist is None:
            wordlist = self.token_list
        self.inputs = self.tokenizer(
            wordlist, truncation=True, is_split_into_words=True
        )

    def _align_labels_with_tokens(self, labels: list, word_ids: list):
        """Helper method to expand the label list so that it matches the new tokens."""
        # beginning of a span needs a label of 2
        # inside a span label of 1
        # punctuation is ignored in the calculation of metrics: set to -100
        new_labels = [
            -100 if word_id is None else labels[word_id] for word_id in word_ids
        ]
        # if the beginning of a span has been split into two tokens,
        # make sure that the label "2" only appears once
        # seems to me we need to use enumerate
        new_labels = [
            1 if label == 2 and new_labels[i - 1] == 2 and i >= 1 else label
            for i, label in enumerate(new_labels)
        ]
        return new_labels

    def add_labels_to_inputs(self, labels=None):
        """Expand the label list to match the tokens after tokenization by
        selected tokenizer."""
        if labels is None:
            labels = self.label_list
        # make sure that it is a nested list to iterate over,
        # otherwise add a layer
        # do we need this?
        # maybe enough to check if labels[0] is a list?
        # does it cost us to iterate over all the data?
        labels = [labels] if not isinstance(labels[0], list) else labels
        labels = [[i] if not isinstance(i, list) else i for i in labels]
        new_labels = []
        for i, label in enumerate(labels):
            word_ids = self.inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(label, word_ids))
        # add new_labels to the tokenized data
        self.inputs["labels"] = new_labels

    def tokenize_and_align(self):
        self.init_tokenizer()
        self.tokenize()
        self.add_labels_to_inputs()
        return self.inputs

    def map_dataset(self, train_test_set):
        tokenized_datasets = train_test_set.map(
            self.tokenize_and_align,
            batched=True,
            remove_columns=train_test_set["train"].column_names,
        )
        return tokenized_datasets
