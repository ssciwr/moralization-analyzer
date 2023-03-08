"""
Contains statistical analysis.
"""
from collections import defaultdict
import pandas as pd
import numpy as np
from spacy_span_analyzer import (
    SpanAnalyzer,
)  # https://github.com/ljvmiranda921/spacy-span-analyzer


def _return_span_analyzer(doc_dict):
    doc_list = []

    for doc in doc_dict.values():
        # doc.spans.pop("paragraphs", None)
        doc.spans.pop("KOMMENTAR", None)
        doc.spans.pop("KAT5-Forderung implizit", None)
        doc_list.append(doc)

    return SpanAnalyzer(doc_list)


def _loop_over_files(doc_dict, file_filter=None):
    df_list = []
    if file_filter is None:
        file_filter = doc_dict.keys()
    elif isinstance(file_filter, str):
        if file_filter not in doc_dict.keys():
            raise KeyError(
                f"The filter `{file_filter}` is not in available files: {doc_dict.keys()}."
            )
        file_filter = [file_filter]

    print(file_filter)
    for file in file_filter:
        df_list.append(_summarize_span_occurences(doc_dict[file]))

    df_complete = pd.concat(
        df_list,
        keys=list(
            zip(
                file_filter,
            )
        ),
        axis=0,
    )
    df_complete = df_complete.fillna(0)

    return df_complete


def _summarize_span_occurences(doc):

    # iterate over all annotation categories and write occurence per paragraph in pandas.DataFrame
    span_categories = list(doc.spans.keys())
    span_categories = _reduce_cat_list(span_categories)

    n_paragraphs = len(doc.spans["paragraphs"])
    span_dict = defaultdict(lambda: np.zeros(n_paragraphs))
    for span_cat in span_categories:
        return_idx_list = _find_spans_in_paragraph(doc, span_cat)
        for ixd, label in return_idx_list:
            span_dict[(span_cat, label)][ixd] = +1

    df_span = pd.DataFrame(span_dict)
    df_span.index = [
        paragraph.text.replace("#", "") for paragraph in doc.spans["paragraphs"]
    ]
    return df_span


def _find_spans_in_paragraph(doc, span_cat):

    if span_cat not in list(doc.spans.keys()):
        raise KeyError(
            f"Key: `{span_cat}` not found in doc.spans, which has {list(doc.spans.keys())}"
        )

    # return a list of indeces in which paragraph each span falls.
    paragraph_list = _get_paragraphs(doc)
    return_idx_list = []
    paragraph_idx = 0

    for span in doc.spans[span_cat]:
        if (
            paragraph_list[paragraph_idx][1] > span.end
            and paragraph_list[paragraph_idx][0] < span.start
        ):
            return_idx_list.append((paragraph_idx, span.label_))
        elif paragraph_idx + 1 < len(paragraph_list):
            paragraph_idx += 1
            return_idx_list.append((paragraph_idx, span.label_))

    return return_idx_list


def _get_paragraphs(doc):
    paragraph_list = [[span.start, span.end] for span in doc.spans["paragraphs"]]
    return paragraph_list


def _reduce_cat_list(span_categories):
    # remove "sc", "paragraph", from list
    span_categories.remove("sc")
    span_categories.remove("paragraphs")

    return span_categories
