import spacy
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import pathlib
from sklearn.metrics import classification_report
import pickle
from spacy.matcher import Matcher


class Few_Shot_Classifier:
    """
    A few shot classifier based on
    https://github.com/Pandora-Intelligence/classy-classification.

    """

    def __init__(self, model="de_core_news_sm"):
        """The initilization can either directly take a spacy nlp object, a saved
        classifier file or be empty for a new default nlp oject.

        :param nlp: Either NLP-Object, path or none, defaults to None
        :type nlp: nlp,spacy.lang.de.German or str, optional
        """

        if isinstance(model, str) and pathlib.Path(model).is_file():
            model_file = pathlib.Path(model)
            if model_file.is_file():
                self.load_model(model_file)
            else:
                raise FileNotFoundError(
                    "The nlp file under {model} does not exist".format(model=model)
                )

        elif model is None:
            self.nlp = spacy.load(model)

    def load_model(self, model_file):
        f = open(model_file, "rb")
        self.nlp = pickle.load(f)

    def create_new_model(self, data: pd.DataFrame, model="spacy"):
        """Takes prepared data as a DataFrame of categories and span strings.

        :param data: Category Span DataFrame
        :type data: pd.DataFrame
        """
        self.nlp = spacy.blank("de")
        print(data)
        self.nlp.add_pipe(
            "textcat_multilabel",
            config={"data": data["data"], "model": "de_core_news_sm"},
        )

    def create_hugging_face_model(
        self, data: pd.DataFrame, model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    ):
        self.nlp = spacy.blank("de")
        self.nlp.add_pipe(
            "text_categorizer",
            config={
                "data": list(data.keys()),
                "model": model,
                "cat_type": "zero",
            },
        )

    def _split_data_dict(self, data, test_percentage=0.2):

        data_train = {}
        data_test = {}
        for cat, value in data.items():
            random.shuffle(value)

            train_list, test_list = (
                value[int(len(value) * test_percentage) :],
                value[: int(len(value) * test_percentage)],
            )
            data_train[cat] = train_list
            data_test[cat] = test_list
        return data_train, data_test

    def apply_model(self, sofa_list):
        """_summary_

        :param sofa_list: List of sofa strings or single string
        :type sofa_list: _type_
        """

        if isinstance(sofa_list, str):
            sofa_list = [sofa_list]

        result_dict = defaultdict(list)
        for sofa in sofa_list:
            result_dict["sofa"].append(sofa)
            cats_prediciton = self.nlp(sofa)._.cats
            if isinstance(cats_prediciton, list):
                cats_prediciton = cats_prediciton[0]

            for key, value in cats_prediciton.items():
                result_dict[key].append(value)

        result_df = pd.DataFrame(result_dict).set_index("sofa")

        return result_df

    def validate_model(self, validation_data=None, return_df=False):
        """
        For the moment validation data needs to have the format specified in
        https://github.com/Pandora-Intelligence/classy-classification.

        """
        if validation_data is None:
            validation_data = self.data_test

        # transform validation data into df

        validation_df = defaultdict(lambda: defaultdict(int))
        # key is the category label and value the list of strings
        for key, val in validation_data.items():
            for sofa in val:
                validation_df[sofa][f"{key}_val"] = 1

        validation_df = pd.DataFrame(validation_df).fillna(0).transpose()
        validation_df.astype("object")
        validation_df.index.name = "sofa"

        test_df = self.apply_model(validation_df.index)

        if return_df is True:
            # return validation_df
            return pd.concat([validation_df, test_df], axis="columns")

        else:
            test_df.values[
                range(len(test_df.index)), np.argmax(test_df.values, axis=1)
            ] = 1
            test_df.where(test_df == 1, 0, inplace=True)

            # add predicted class as column entry
            test_df["pred_class"] = test_df.idxmax(axis=1)
            validation_df["true_class"] = validation_df.idxmax(axis=1)

            validation_df["true_class"] = [
                string.split("_val")[0] for string in validation_df["true_class"]
            ]

            # acc = {key:accuracy_score(validation_df[f"{key}_val"],test_df[key])
            # for key in test_df.keys() if key!= "pred_class"}
            output_df = pd.DataFrame(
                classification_report(
                    validation_df["true_class"], test_df["pred_class"], output_dict=True
                )
            )

            return output_df

    def save_model(self, filename):
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self.nlp, f)


class Binary_Few_Shot_Classifier(Few_Shot_Classifier):
    def __init__(
        self,
        df_spans,
        test_percentage=0.2,
        model_origin="spacy",
        model="de_core_news_sm",
    ):
        """This model only differentiates between no moralization and moralization.

        :param df_spans: _description_
        :type df_spans: _type_
        :param model_origin: either spacy or huggingface
        :type filename: str
        """
        Few_Shot_Classifier.__init__(self)
        df_spans.loc["KAT1-Moralisierendes Segment"]

        data = defaultdict(list)
        for file_name in df_spans.loc["KAT1-Moralisierendes Segment"]:
            for cat_name, cat_value in (
                df_spans.loc["KAT1-Moralisierendes Segment"][file_name]
                .to_dict()
                .items()
            ):
                if cat_value:
                    if cat_name == "Keine Moralisierung":
                        data["Keine Moralisierung"] += cat_value.split("&")
                    else:
                        data["Moralisierung"] += cat_value.split("&")

        # split training and test data

        if model_origin == "spacy":
            data_train, self.data_test = self._split_data_dict(data)

            self.create_new_model(data_train, model=model)
        elif model_origin == "huggingface":
            self.create_hugging_face_model(data, model=model)

            self.data_test = data


class Pattern_Matching:
    def __init__(self, df_spans):
        self.df_spans = df_spans
        self.nlp = spacy.load("de_core_news_sm")
        self.matcher = Matcher(self.nlp.vocab)

        self._make_pattern_list()

    def _make_pattern_list(self):
        self.val_dict = defaultdict(list)
        for id in self.df_spans.index:
            for column in self.df_spans.columns:

                if self.df_spans[column].loc[id]:

                    for string in self.df_spans[column].loc[id].split("&"):
                        pattern = []
                        random_int = random.randint(0, 1)
                        # take 20% of the data for validation.
                        if string.strip() and random_int < 0.8:
                            # for word in string.split(" "):

                            # if word.strip():
                            pattern.append({"LOWER": string.lower()})

                            self.matcher.add(id[1], [pattern])

                        elif string.strip() and random_int >= 0.8:
                            self.val_dict[id].append(string)

    # a problem seems to be, that some words are present in many different
    def validate(self):
        correct_labels = 0
        false_labels = 0
        for key, string_list in self.val_dict.items():
            for string in string_list:
                doc = self.nlp(string)
                matches = self.matcher(doc)
                for match_id, start, end in matches:
                    string_id = self.nlp.vocab.strings[
                        match_id
                    ]  # Get string representation
                    if string_id == key[1]:
                        correct_labels += 1
                    else:
                        false_labels += 1

        print(f"Correct: {correct_labels}, incorrect: {false_labels}")
