from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from typing import List


class TransformersModelManager:
    """
    Create, import, modify, train and publish transformers models.

    Models can be trained on data from a DataManager, and published to hugging face.
    """

    def __init__(
        self,
        model_name: str = "bert-base-cased",
    ) -> None:
        """
        Import an existing model from `model_name` from Hugging Face.

        Args:
            model_name (str): Name of the pretrained model
        """
        self.model_name = model_name

    def init_tokenizer(self, model_name=None, kwargs=None):
        """Initialize the tokenizer that goes along with the selected model.
        Only fast tokenizers can be used.

        Args:
            model_name (str, optional): The name of the model that will be used
            for tokenization. If not specified, the model selected at class
            instantiation will be used.
            kwargs (dict, optional): Keyword arguments to pass to the tokenizer.
        """
        if model_name is None:
            model_name = self.model_name
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

    def tokenize(self, wordlist: List = None):
        if wordlist is None:
            wordlist = self.token_list
        wordlist = self._check_is_nested(wordlist)
        self.inputs = self.tokenizer(
            wordlist, truncation=True, is_split_into_words=True
        )

    def _check_is_nested(self, list_to_check: List) -> list:
        # make sure that data is a nested list,
        # otherwise add a layer
        # do we need this?
        # maybe enough to check if list_to_check[0] is a list?
        # does it cost us to iterate over all the data?
        list_to_check = (
            [list_to_check] if not isinstance(list_to_check[0], list) else list_to_check
        )
        if not all(isinstance(i, list) for i in list_to_check):
            list_to_check = [
                [i] if not isinstance(i, list) else i for i in list_to_check
            ]
        return list_to_check

    def _align_labels_with_tokens(self, labels: List, word_ids: List):
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
            1 if label == 2 and i >= 1 and new_labels[i - 1] == 2 else label
            for i, label in enumerate(new_labels)
        ]
        return new_labels

    def add_labels_to_inputs(self, labels=None):
        """Expand the label list to match the tokens after tokenization by
        selected tokenizer."""
        if labels is None:
            labels = self.label_list
        labels = self._check_is_nested(labels)
        new_labels = []
        for i, label in enumerate(labels):
            word_ids = self.inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(label, word_ids))
        # add new_labels to the tokenized data
        self.inputs["labels"] = new_labels

    def tokenize_and_align(self, examples):
        self.init_tokenizer()
        self.tokenize(examples["word"])
        self.add_labels_to_inputs(examples["label"])
        return self.inputs

    def map_dataset(self, train_test_set):
        tokenized_datasets = train_test_set.map(
            self.tokenize_and_align,
            batched=True,
            remove_columns=train_test_set["train"].column_names,
        )
        return tokenized_datasets

    def init_data_collator(self):
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )
