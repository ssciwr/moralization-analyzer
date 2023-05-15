from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import tokenization_utils_base
from transformers import get_scheduler
from transformers import pipeline  # noqa
from datasets import DatasetDict, formatting
from typing import Union, List, Dict
import evaluate
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch import no_grad


class TransformersModelManager:
    """
    Create, import, modify, train and publish transformers models.

    Models can be trained on data from a DataManager, and published to hugging face.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_name: str = "bert-base-cased",
    ) -> None:
        """
        Import an existing model from `model_name` from Hugging Face.

        Args:
            model_path (str or Path): Folder where the model is (or will be) stored
            model_name (str): Name of the pretrained model
        """
        self._model_path = Path(model_path)
        self.model_name = model_name

    def init_tokenizer(self, model_name=None, kwargs=None) -> None:
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

    def tokenize(self, wordlist: List) -> None:
        """Tokenize the pre-tokenized inputs with the selected tokenizer.

        Args:
            wordlist (list, required): The list of words that will be tokenized.
            The list is checked for nesting, the tokenizer expects a nested list of lists.
        """
        wordlist = self._check_is_nested(wordlist)
        self.inputs = self.tokenizer(
            wordlist, truncation=True, is_split_into_words=True
        )

    def _check_is_nested(self, list_to_check: List) -> List:
        """Check the pre-tokenized list of words for nesting.

        Args:
            wordlist (list, required): The list of words that will be tokenized.
            The list is checked for nesting, the tokenizer expects a nested list of lists.
        """
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

    def _align_labels_with_tokens(self, labels: List, word_ids: List) -> List:
        """Expand the label list so that it matches the new tokens.

        Args:
            labels (list, required): The list of labels that needs to be aligned.
            word_ids (list, required): The word ids of the tokenized words, mapping
            tokenized to pre-tokenized data.
        """
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

    def add_labels_to_inputs(self, labels: List) -> None:
        """Expand the label list to match the tokens after tokenization by
        selected tokenizer. Add to inputs.

        Args:
            labels (list, required): The nested list of labels that needs to be aligned.
        """
        labels = self._check_is_nested(labels)
        new_labels = []
        for i, label in enumerate(labels):
            word_ids = self.inputs.word_ids(i)
            new_labels.append(self._align_labels_with_tokens(label, word_ids))
        # add new_labels to the tokenized data
        self.inputs["labels"] = new_labels

    def tokenize_and_align(
        self, examples: formatting.formatting.LazyBatch
    ) -> tokenization_utils_base.BatchEncoding:
        """Tokenize the word list and align the label list to the new tokens.

        Args:
            examples (batch, required): The batch of pre-tokenized words
            and labels that needs to be aligned.

        Returns:
            inputs (BatchEncoding): The encoded tokens, labels, etc, after tokenization
        """
        self.init_tokenizer()
        self.tokenize(examples[self.token_column_name])
        self.add_labels_to_inputs(examples[self.label_column_name])
        print(type(self.inputs))

        return self.inputs

    def map_dataset(
        self,
        train_test_set: DatasetDict,
        token_column_name: str = "word",
        label_column_name: str = "label",
    ) -> DatasetDict:
        """Apply the tokenization to the complete dataset using a mapping function.

        Args:
            train_test_set (DatasetDict, required): The nested list of labels that needs to be aligned.
            token_column_name (str, optional): The name of the column containing the sentences/tokens.
            Defaults to "word".
            label_column_name (str): The name of the column containing the labels.
            Defaults to "label".

        Returns:
            tokenized_datasets (DatasetDict): The tokenized and label-aligned dataset.
        """
        self.token_column_name = token_column_name
        self.label_column_name = label_column_name
        tokenized_datasets = train_test_set.map(
            self.tokenize_and_align,
            batched=True,
            remove_columns=train_test_set["train"].column_names,
        )
        return tokenized_datasets

    def init_data_collator(self, kwargs: Dict = None) -> None:
        """Initializes the Data Collator that will form batches from the data
        for the training.

        Args:
            kwargs (dict, optional): Keyword arguments to be passed to the data collator.
            Only arguments other than `tokenizer`.
        """
        if not kwargs:
            kwargs = {}
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, **kwargs
        )

    def create_batch(
        self, tokenized_datasets: DatasetDict
    ) -> tokenization_utils_base.BatchEncoding:
        """Creates the batches from the tokenized datasets using the Data Collator."""
        batch = self.data_collator([item for item in tokenized_datasets["train"]])
        return batch

    def load_evaluation_metric(
        self, label_names: List = None, eval_metric: str = "seqeval"
    ) -> None:
        """Loads the evaluation metric to determine scores for the training.

        Args:
            label_names (list, optional): Label names as strings for the labels to be trained. If
            the evaluation metric is sequential evaluation, strings are required. For other metrics,
            please consult the evaluate documentation: https://huggingface.co/docs/evaluate/choosing_a_metric
            eval_metric (str, optional): Evaluation metric to be used, defaults to `seqeval`.
        """
        # default is sequential evaluation
        # for other metrics, please see
        # https://huggingface.co/docs/evaluate/choosing_a_metric
        if not label_names:
            self.label_names = ["0", "M", "M-BEG"]
        else:
            self.label_names = label_names
        self.metric = evaluate.load(eval_metric)

    def compute_metrics(self, eval_preds: tuple) -> Dict:
        """Convenience function to compute and return the metrics.

        Args:
            eval_preds (tuple, required): The predicted and actual labels as a tuple.
        """
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        # Remove ignored index (special tokens) and convert to labels
        # we need this since seqeval operates on strings and not integers
        true_labels = [
            [self.label_names[m] for m in label if m != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, m) in zip(prediction, label) if m != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def set_id2label(self) -> None:
        """Creates a map from label id (integer) to label name (string)."""

        if not hasattr(self, "label_names"):
            raise ValueError("Please set the label names first!")
        self.id2label = {i: label for i, label in enumerate(self.label_names)}

    def set_label2id(self) -> None:
        """Creates a map from label name (string) to label id (integer)."""

        if not hasattr(self, "id2label"):
            raise ValueError("Please set id2label first!")
        self.label2id = {v: k for k, v in self.id2label.items()}

    def load_model(self, model_name: str = None) -> None:
        """Loads the model to be finetuned in the training.

        Args:
            model_name (str, optional): The name of the model to be used in the training.
            Defaults to "bert-base-cased".
        """
        if model_name is None:
            model_name = self.model_name
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        except OSError:
            # here we also need more exceptions for no network etc
            raise OSError("Could not initiate model - please check your model name")

    def load_dataloader(
        self, tokenized_datasets: DatasetDict, batch_size: int = 8
    ) -> None:
        """Loads the pytorch dataloader for the train and test data, that loads the data into batches.

        Args:
            tokenized_datasets (DatasetDict, required): The tokenized train and test datasets.
            batch_size (int, optional): Batch size that the data will be processed in. Defaults
            to 8.
        """
        self.train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=batch_size,
        )
        self.eval_dataloader = DataLoader(
            tokenized_datasets["test"],
            collate_fn=self.data_collator,
            batch_size=batch_size,
        )

    def load_optimizer(self, learning_rate: float = 2e-5, kwargs: Dict = None) -> None:
        """Load the AdamW adaptive optimizer that handles the optimization process.

        Args:
            learning_rate (float, optional): Learning rate to be used in the optimization. Defaults to
            2e-5.
            kwargs (dict, optional): Further keyword arguments other than learning rate to be passed to the
            optimizer.
        """
        if not kwargs:
            kwargs = {}
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, **kwargs)

    def load_accelerator(self) -> None:
        """Loads the accelerator that enables PyTorch to run on any distributed configuration, handles all
        cuda and device placements."""
        self.accelerator = Accelerator()
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )

    def load_scheduler(
        self, scheduler_name: str = "linear", num_train_epochs: int = 3
    ) -> None:
        """Load the scheduler that handles the adjustement of the learning rate during the training.

        Args:
            scheduler_name (string, optional): Type of scheduler to be used that adjusts the learning
            rate During the training. Defaults to "linear".
            num_train_epochs (int, optional): The number of training steps to do. Defaults to 3.
        """

        self.num_train_epochs = num_train_epochs
        try:
            self.num_update_steps_per_epoch = len(self.train_dataloader)
        except AttributeError:
            raise ValueError(
                "Dataloader not initialized. Please load the dataloader first."
            )
        if not hasattr(self, "optimizer"):
            raise ValueError(
                "Optimizer not initialized. Please load the optimizer first."
            )
        self.num_training_steps = num_train_epochs * self.num_update_steps_per_epoch
        self.lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()
        if not self.label_names:
            raise ValueError("Label names not set!")
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[m] for m in label if m != -100] for label in labels
        ]
        true_predictions = [
            [self.label_names[p] for (p, m) in zip(prediction, label) if m != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def train(self) -> None:
        """Train a model using the pre-loaded components."""

        # show a progress bar
        progress_bar = tqdm(range(self.num_training_steps))

        for epoch in range(self.num_train_epochs):
            # set the mode to training
            self.model.train()
            for batch in self.train_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            self._evaluate_model()

            self.results = self.metric.compute()
            print(
                f"epoch {epoch}:",
                {
                    key: self.results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            self.save()

    def _evaluate_model(self):
        # set mode to evaluation
        self.model.eval()
        for batch in self.eval_dataloader:
            with no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            # Necessary to pad predictions and labels for being gathered
            predictions = self.accelerator.pad_across_processes(
                predictions, dim=1, pad_index=-100
            )
            labels = self.accelerator.pad_across_processes(
                labels, dim=1, pad_index=-100
            )
            predictions_gathered = self.accelerator.gather(predictions)
            labels_gathered = self.accelerator.gather(labels)
            true_predictions, true_labels = self.postprocess(
                predictions_gathered, labels_gathered
            )
            self.metric.add_batch(predictions=true_predictions, references=true_labels)

    def save(self):
        # Save the model to model path
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            self._model_path, save_function=self.accelerator.save
        )
        # the below is executed only once in main process
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self._model_path)

    def evaluate(self, token: str, model_path: Union[str, Path] = None):
        if not model_path:
            model_checkpoint = self._model_path
        else:
            model_checkpoint = model_path
        token_classifier = pipeline(
            "token-classification",
            model=model_checkpoint,
            aggregation_strategy="simple",
        )
        return token_classifier(token)
