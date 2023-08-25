from typing import Dict, List, Tuple, Union
import copy


class TransformersDataHandler:
    """Helper class to organize and prepare data for transformer models."""

    def get_data_lists(self, doc_dict: Dict) -> None:
        """Convert the data from doc object to lists. Required for transformers training.

        Set the lists of tokens and labels for transformers training, with a nested list of
        sentences and tokens, and an equally nested list of labels that are initiated to zero.

        Args:
            doc_dict (dict, required): The dictionary of doc objects for each data source.

        """
        # Create a list of all the sentences for all sources.
        self.sentence_list = []
        # Create a list of all the tokens.
        self.token_list = []
        # Create a list of all the labels.
        self.label_list = []
        for example_name in doc_dict.keys():
            sentence_list = [
                [token.text for token in sent] for sent in doc_dict[example_name].sents
            ]
            token_list = [
                [token for token in sent] for sent in doc_dict[example_name].sents
            ]
            # initialize nested label list to 0
            label_list = [[0 for _ in sent] for sent in doc_dict[example_name].sents]
            # extend the main lists for all the sources by the lists for the single sources
            self.sentence_list.extend(sentence_list)
            self.token_list.extend(token_list)
            self.label_list.extend(label_list)

    def generate_labels(
        self,
        doc_dict: Dict,
        selected_labels: Union[List, str] = None,
        task: str = "task1",
    ) -> None:
        """Generate the labels from the annotated tokens in one long list. Required for transformers training.

        Args:
            doc_dict (dict, required): The dictionary of doc objects for each data source.
            selected_labels (Union[list, str], optional): The labels that should be combined in the training. Default: [
                "Moralisierung Kontext", "Moralisierung Weltwissen", "Moralisierung explizit",
                "Moralisierung interpretativ",]. You can select "all" to choose all labels for a given task.
            task (string, optional): The task from which the labels are selected. Default is task 1 (category
                1 "KAT1-Moralisierendes Segment").
        """
        # generate the labels based on the current list of tokens
        # now set all Moralisierung, Moralisierung Kontext,
        # Moralisierung explizit, Moralisierung interpretativ, Moralisierung Weltwissen to 1
        # here we actually need to select by task
        if not selected_labels:
            # if not set, we select all
            selected_labels = "all"
        # create a list as long as tokens
        self.labels = []
        for example_name in doc_dict.keys():
            labels = [0 for _ in doc_dict[example_name]]
            for span in doc_dict[example_name].spans[task]:
                if selected_labels == "all" or span.label_ in selected_labels:
                    labels[span.start + 1 : span.end] = [1] * (
                        span.end - span.start - 1
                    )
                    # mark the beginning of a span with 2
                    labels[span.start] = 2
                else:
                    print(span.label_, "not in selected labels")
            self.labels.extend(labels)

    def generate_spans(
        self, doc_dict: Dict, selected_labels: Union[List, str] = None, task: str = None
    ) -> None:
        """Generate the spans from the annotated tokens in a nested list. Required for spacy training.

        This is a bit painful as the spans slices from the doc do contain the sent information,
        but there is no sentence id or so. So we need to iterate over both sents and spans
        and compare to find out if a span is inside a certain sentence. Also, the span token ids
        (span.start and span.end) are given relative per text source (doc). So we always need
        to correct this, as for the dataset splitting into test and train, this information will
        be lost. The span begin and span end token integer needs to be given relative to the
        position of the token in the sentence.

        Args:
            doc_dict (dict, required): The dictionary of doc objects for each data source.
            selected_labels (Union[list, str], optional): The labels that should be combined in the training. Default: [
                "Moralisierung Kontext", "Moralisierung Weltwissen", "Moralisierung explizit",
                "Moralisierung interpretativ",]. You can select "all" to choose all labels for a given task.
                Note that this will not produce relevant results for task1 as "Keine Moralisierung" is also a label.
            task (string, optional): The task from which the labels are selected. Default is task 1 (category
                1 "KAT1-Moralisierendes Segment").
        """
        if not selected_labels:
            # if no labels are selected per task, we just choose all
            selected_labels = "all"
        if not task:
            task = "task1"
        # generate the nested spans based on the sentences
        self.span_begin = []
        self.span_end = []
        self.span_label = []
        for example_name in doc_dict.keys():
            sentence_start_end = [
                (sent.start, sent.end) for sent in doc_dict[example_name].sents
            ]
            span_b = [[] for _ in doc_dict[example_name].sents]
            span_e = copy.deepcopy(span_b)
            span_l = [[] for _ in doc_dict[example_name].sents]
            for span in doc_dict[example_name].spans[task]:
                if selected_labels == "all" or span.label_ in selected_labels:
                    # find out which sentence the span lies in
                    sentence_boundaries = (span.sent.start, span.sent.end)
                    sentence_id = sentence_start_end.index(sentence_boundaries)
                    span_b[sentence_id].append(span.start - sentence_boundaries[0] + 1)
                    span_e[sentence_id].append(span.end - sentence_boundaries[0] + 1)
                    span_l[sentence_id].append(span.label_)
                else:
                    print(span.label_, "not in selected labels")
            # fill all the empty lists with 0,0,"" so that in the conversion to a
            # pyarrow Table, we have enough information to create the correct Arrow type
            span_b = [[0] if not i else i for i in span_b]
            span_e = [[0] if not i else i for i in span_e]
            span_l = [[""] if not i else i for i in span_l]
            self.span_begin.extend(span_b)
            self.span_end.extend(span_e)
            self.span_label.extend(span_l)

    def structure_labels(self) -> Tuple[List, List]:
        """Structure the tokens from one long list into a nested list for sentences.
        Required for transformers training.

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
        return (
            self.sentence_list,
            self.label_list,
            self.span_begin,
            self.span_end,
            self.span_label,
        )
