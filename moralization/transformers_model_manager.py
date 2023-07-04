from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import tokenization_utils_base
from transformers import get_scheduler
from transformers import pipeline  # noqa
from datasets import DatasetDict, formatting
from typing import Union, List, Dict, Optional, Any
import evaluate
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch import no_grad
from moralization.data_manager import DataManager
from moralization.model_manager import ModelManager
import json
from huggingface_hub import HfApi


IGNORED_LABEL = -100


def _update_model_meta(model_path: Path, metadata: Dict):
    """
    Update matching keys in the README.md file with values from the supplied metadata dict.
    """
    meta_file = model_path / "README.md"
    if not meta_file.is_file():
        return
    with open(meta_file) as f:
        meta = json.load(f)
        for k, v in metadata.items():
            if k in meta:
                meta[k] = v
    with open(meta_file, "w") as f:
        json.dump(meta, f)


def _import_or_create_metadata(model_path: Path) -> Dict[str, Any]:
    meta_file = model_path / "README.md"
    default_metadata = {
        "name": "pipeline",
        "version": "0.0.0",
        "description": "",
        "author": "",
        "email": "",
        "url": "",
        "license": "",
    }
    if not meta_file.is_file():
        with open(meta_file, "w") as f:
            json.dump(default_metadata, f)
    with open(meta_file) as f:
        return json.load(f)


class TransformersModelManager(ModelManager):
    """
    Create, import, modify, train and publish transformers models.

    Models can be trained on data from a DataManager, and published to hugging face.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_name: str = "bert-base-cased",
        label_names: List = ["0", "M", "M-BEG"],
    ) -> None:
        """
        Import an existing model from `model_name` from Hugging Face.

        Args:
            model_path (str or Path): Folder where the model is (or will be) stored
            model_name (str): Name of the pretrained model
        """
        super().__init__(model_path)
        self.model_name = model_name
        self.metadata = _import_or_create_metadata(self.model_path)
        self._model_is_trained = False
        # somewhere we should check that the label names length is same as number of different labels
        # this however can only be done after the `train` etc method is called with the data
        # and load model that uses the label names is already done at init
        # should be done when the tokenization is accessed and label list expanded?
        self.label_names = label_names
        # set up all the preprocessing for the dataset
        self._init_tokenizer()
        self._init_data_collator()
        self._load_model()
        # set up metadata
        # self.metadata = self._import_or_create_metadata(self.model_path)

    def _init_tokenizer(self, model_name=None, kwargs=None) -> None:
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
            IGNORED_LABEL if word_id is None else labels[word_id]
            for word_id in word_ids
        ]
        # if the beginning of a span has been split into two tokens,
        # make sure that the label "2" only appears once
        # seems to me we need to use enumerate
        new_labels = [
            1 if label == 2 and i >= 1 and new_labels[i - 1] == 2 else label
            for i, label in enumerate(new_labels)
        ]
        # find out how many unique labels
        # doesn't quite work as not every dataset entry has all labels
        # unique_labels = np.unique(new_labels)
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
        self.tokenize(examples[self.token_column_name])
        self.add_labels_to_inputs(examples[self.label_column_name])
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

    def _init_data_collator(self, kwargs: Dict = None) -> None:
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

    def _load_evaluation_metric(self, eval_metric: str = "seqeval") -> None:
        """Loads the evaluation metric to determine scores for the training.

        Args:
            eval_metric (str, optional): Evaluation metric to be used, defaults to `seqeval`. If
            the evaluation metric is sequential evaluation, strings are required for the label names. For other metrics,
            please consult the evaluate documentation: https://huggingface.co/docs/evaluate/choosing_a_metric
        """
        # default is sequential evaluation
        # for other metrics, please see
        # https://huggingface.co/docs/evaluate/choosing_a_metric
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
            [self.label_names[m] for m in label if m != IGNORED_LABEL]
            for label in labels
        ]
        true_predictions = [
            [
                self.label_names[p]
                for (p, m) in zip(prediction, label)
                if m != IGNORED_LABEL
            ]
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

    def _set_id2label(self) -> None:
        """Creates a map from label id (integer) to label name (string)."""
        self.id2label = {i: label for i, label in enumerate(self.label_names)}

    def _set_label2id(self) -> None:
        """Creates a map from label name (string) to label id (integer)."""
        self.label2id = {v: k for k, v in self.id2label.items()}

    def _load_model(self, model_name: str = None) -> None:
        """Loads the model to be finetuned in the training.

        Args:
            model_name (str, optional): The name of the model to be used in the training.
            Defaults to "bert-base-cased".
        """
        self._set_id2label()
        self._set_label2id()
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

    def _load_dataloader(
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

    def _load_optimizer(self, learning_rate: float, kwargs: Dict = None) -> None:
        """Load the AdamW adaptive optimizer that handles the optimization process.

        Args:
            learning_rate (float): Learning rate to be used in the optimization.
            kwargs (dict, optional): Further keyword arguments other than learning rate to be passed to the
            optimizer.
        """
        if not kwargs:
            kwargs = {}
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, **kwargs)

    def _load_accelerator(self) -> None:
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

    def _load_scheduler(
        self,
        num_train_epochs: int,
        scheduler_name: str = "linear",
    ) -> None:
        """Load the scheduler that handles the adjustement of the learning rate during the training.

        Args:
            num_train_epochs (int): The number of training steps to do. Defaults to 3.
            scheduler_name (string, optional): Type of scheduler to be used that adjusts the learning
            rate during the training. Defaults to "linear".
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
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [
            [self.label_names[m] for m in label if m != IGNORED_LABEL]
            for label in labels
        ]
        true_predictions = [
            [
                self.label_names[p]
                for (p, m) in zip(prediction, label)
                if m != IGNORED_LABEL
            ]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    def _prepare_data(
        self, data_manager: DataManager, token_column_name: str, label_column_name: str
    ) -> DatasetDict:
        train_test_dataset = data_manager.train_test_set
        tokenized_dataset = self.map_dataset(
            train_test_set=train_test_dataset,
            token_column_name=token_column_name,
            label_column_name=label_column_name,
        )
        return tokenized_dataset

    def _initialize_training(
        self,
        tokenized_dataset: DatasetDict,
        num_train_epochs: int,
        learning_rate: float,
    ) -> None:
        self._init_data_collator()
        self._load_evaluation_metric()
        self._load_dataloader(tokenized_dataset)
        self._load_optimizer(learning_rate)
        self._load_accelerator()
        self._load_scheduler(num_train_epochs=num_train_epochs)

    def train(
        self,
        data_manager: DataManager,
        token_column_name: str,
        label_column_name: str,
        num_train_epochs: int = 5,
        learning_rate: float = 2e-5,
    ) -> None:
        """Train a model using the pre-loaded components."""

        # initialize all components and prepare the dataset.
        tokenized_dataset = self._prepare_data(
            data_manager, token_column_name, label_column_name
        )
        self._initialize_training(tokenized_dataset, num_train_epochs, learning_rate)
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
        self._model_is_trained = True

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
                predictions, dim=1, pad_index=IGNORED_LABEL
            )
            labels = self.accelerator.pad_across_processes(
                labels, dim=1, pad_index=IGNORED_LABEL
            )
            predictions_gathered = self.accelerator.gather(predictions)
            labels_gathered = self.accelerator.gather(labels)
            true_predictions, true_labels = self.postprocess(
                predictions_gathered, labels_gathered
            )
            self.metric.add_batch(predictions=true_predictions, references=true_labels)

    def save(self):
        """Save the model to the set model path.
        If a model already exists in that path, it will be overwritten."""
        model_file = self.model_path / "pytorch_model.bin"
        if model_file.exists():
            print(
                "Model file already existing at specified model path {} - will be overwritten.".format(
                    self.model_path
                )
            )
        # save the metadata
        with open(self.model_path / "README.md", "w") as f:
            json.dump(self.metadata, f)
        _update_model_meta(self.model_path, self.metadata)
        # Save the model to model path
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            self.model_path, save_function=self.accelerator.save
        )
        # the below is executed only once in main process
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.model_path)

    def evaluate(self, token: str):
        if not hasattr(self, "_model_path"):
            raise ValueError(
                "Please initiate the class first with a path to the model that you want to evaluate"
            )
        token_classifier = pipeline(
            "token-classification",
            model=self.model_path,
            aggregation_strategy="simple",
        )
        return token_classifier(token)

    def test(self):
        pass

    def _check_model_is_trained_before_it_can_be(self, action: str = "used"):
        if not self._model_is_trained:
            raise RuntimeError(f"Model must be trained before it can be {action}.")

    def publish(self, hugging_face_token: Optional[str] = None) -> str:
        """Publish the model to Hugging Face.

        This requires a User Access Token from https://huggingface.co/

        The token can either be passed via the `hugging_face_token` argument,
        or it can be set via the `HUGGING_FACE_TOKEN` environment variable. If
        no token is provided, a command prompt will open to request the token.

        Args:
            hugging_face_token (str, optional): Hugging Face User Access Token
        Returns:
            str: The URL of the published model
        """
        # self._check_model_is_trained_before_it_can_be("published")
        for key, value in self.metadata.items():
            if value == "":
                raise RuntimeError(
                    f"Metadata '{key}' is not set - all metadata needs to be set before publishing a model."
                )
        # self.save()
        self._login_to_huggingface(hugging_face_token)
        self.model.push_to_hub("test-model")
        # also push the README metadata
        api = HfApi()
        myurl = api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id="iulusoy/test-model",
            repo_type="model",
        )
        return myurl
