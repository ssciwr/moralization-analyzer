from spacy.tokens import DocBin, Doc
from pathlib import Path
from tempfile import mkdtemp
import datasets
import spacy


class SpacyDataHandler:
    """Helper class to organize and prepare spacy train and test data."""

    def docbin_from_dataset(
        data_set: datasets.Dataset,
        task: str,
        data_split: str = "train",
        output_dir: Path = None,
        overwrite: bool = False,
        check_docs: bool = False,
    ) -> DocBin:
        """Create a DocBin from a Dataset.

        This uses the span begin, span end, and span label columns from the
        Dataset. The complication here is that span begin and end are given as
        token id, whereas the doc requires the character id. So we need to count
        the characters in the sentence and assign accordingly.
        Also, each sentence creates its own doc which is then appended to the
        overall doc. It is done like this to avoid having to factor in all the
        previous character counts from prior sentences. If this becomes too slow
        for large corpora, we can think about first parsing the lists and correcting
        the character count and then feeding everything into a doc at once.

        Args:
            data_set (datasets.Dataset): The dataframe to be converted into a DocBin.
            task (str): The name of the SpanGroup (task that is targeted in the training).
            data_split (str, optional): The split of the data that is exported. Defaults to "train".
            output_dir (Path, optional): Path of the output directory where the data is saved, defaults to None.
                If None the working directory is used.
            overwrite (bool, optional): Whether or not the spacy files should be written
                even if the file is already present.
            check_docs (bool, optional): Check all the spans inside the doc and print. Defaults to False.
        Returns:
            Path: The path to the spacy formatted data."""
        nlp = spacy.blank("en")

        # first create a list from the dataset "Sentences" column for train and test
        # similarly for the spans
        # TODO here we need to pass column names
        textlist = data_set[data_split]["Sentences"]
        span_begin_list = data_set[data_split]["Span_begin"]
        span_end_list = data_set[data_split]["Span_end"]
        span_label_list = data_set[data_split]["Span_label"]

        # we create one doc container each for each sentence, then concatenate them together
        # could be faster to first concat the lists and adjust the span_begin and span_end
        # token number, but is messier, so we do it the slow way first
        doclist = []
        for i in range(len(textlist)):
            # find out if there is an annotation for that sentence
            # if yes, create span
            if span_begin_list[i] != [0]:
                # join the tokens in the sentence together with whitespace and create doc
                merged_tokens = " ".join(textlist[i])
                doc = nlp(merged_tokens)
                # create a new span group "task"
                doc.spans[task] = []

                # get the character ids for each of the spans in the sentence
                for j in range(len(span_begin_list[i])):
                    (
                        char_span_begin,
                        char_span_end,
                        substring,
                    ) = SpacyDataHandler._get_character_ids(
                        merged_tokens,
                        textlist[i],
                        span_begin_list[i][j],
                        span_end_list[i][j],
                    )
                    # check that this will return the same string as merged_tokens
                    if merged_tokens[char_span_begin:char_span_end] != substring:
                        raise RuntimeError(
                            "Could not match *{}* and *{}* inside *{}*".format(
                                merged_tokens[char_span_begin:char_span_end],
                                substring,
                                merged_tokens,
                            )
                        )
                    span = doc.char_span(
                        char_span_begin, char_span_end, span_label_list[i][j]
                    )
                    # check that span text is the same as substring
                    if span.text != substring:
                        raise RuntimeError(
                            "Could not assign span *{}* for annotation *{}*".format(
                                span, substring[1::]
                            )
                        )
                    doc.spans[task].append(span)
                doclist.append(doc)
                if check_docs:
                    SpacyDataHandler._check_docs(doc, task)
        # now merge all the docs for each sentence into one DocBin
        # the file is named "train" for training and "dev" for testing
        # for now, we do not have further filenames
        outfilename = "train" if data_split == "train" else "dev"
        data_path = SpacyDataHandler.export_training_testing_data(
            doclist, outfilename, output_dir, overwrite
        )
        return data_path

    def _check_docs(doc: Doc, task):
        # go through the doc and print all spans and labels
        for span in doc.spans[task]:
            print("""Span is: "{}", with label: "{}".""".format(span, span.label_))

    def _get_character_ids(merged_tokens, text, span_begin, span_end):
        # figure out which list indices match the span tokens
        # first put the token substring together
        substring = ""
        # sometimes annotations are longer than one sentences
        # we skip these because at the moment, we cannot account for
        # multisentence lists
        # we just end the annotation at the end of the sentence
        if len(text) < span_end - 1:
            span_end = len(text) + 1
        # account for Python counting, and span_end signifies
        # first token that is not in the span
        for j in range(span_begin - 1, span_end - 1):
            substring = " ".join([substring, text[j]])
        # do not account for whitespace in beginning of span, otherwise spacy will return None
        # this whitespace is added in first iteration of join
        substring = substring.strip()
        temp = merged_tokens.split(substring)
        # now we need the character count of the beginning of sentence up to the
        # annotated span; and the character count for the span
        # sometimes no split could be found, because the complete string is replicated in the annotation
        char_span_begin = len(temp[0]) if len(temp) > 1 else 0
        # whitespace in beginning of substring
        char_span_end = len(substring) + char_span_begin
        return char_span_begin, char_span_end, substring

    def export_training_testing_data(
        doclist: list, filename: str, output_dir, overwrite
    ) -> Path:
        """Convert a list of spacy docs to a serialisable DocBin object and save it to disk.
        Automatically processes training and testing files.

        Args:
            doclist (list): List of one doc per sentence.
            filename (str): Name of the file to write.
            output_dir (Path): Path of the output directory where the data is saved.
            overwrite (bool): Whether or not the spacy files should be written
                even if the file is already present.
        Return:
            Path: the location of the written file.
        """

        if output_dir is None:
            output_dir = Path(mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        file = filename + ".spacy"
        out_filename = output_dir / file

        # check if files already exists, only if overwrite is False:
        if overwrite is False:
            if out_filename.exists():
                raise FileExistsError(
                    "The given directory already has an exported DocBin file with name {}.".format(
                        filename
                    )
                    + " Please choose a new directory or set overwrite to True."
                    + f"Given directory is: {output_dir}"
                )

        db_out = DocBin()

        for doc in doclist:
            db_out.add(doc)

        db_out.to_disk(out_filename)
        return out_filename

    def _check_files(input_dir=None, train_file=None, test_file=None):
        if input_dir is None and test_file is None and train_file is None:
            raise FileNotFoundError(
                "Please provide either a directory or the file locations."
            )

        if (train_file is not None and test_file is None) or (
            train_file is None and test_file is not None
        ):
            raise FileNotFoundError(
                "When providing a data file location, please also provide the other one."
                + f"Currently `train_file` is {train_file} and `test_file` is {test_file}"
            )

        if train_file and test_file:
            train_file = Path(train_file)
            test_file = Path(test_file)
            # check if files are spacy
            if train_file.suffix != ".spacy" or test_file.suffix != ".spacy":
                raise TypeError("The provided files are not spacy binaries.")

            # if both files exists we can exit at this point.
            if train_file.exists() and test_file.exists():
                return train_file, test_file

        # if no files are given use the default values
        else:
            train_file = Path("train.spacy")
            test_file = Path("dev.spacy")

        # if not we search in the current or given working directory
        if input_dir is None:
            input_dir = Path.cwd()
        else:
            input_dir = Path(input_dir)

        # search the directory for the files.
        # TODO I believe the below line is not required. It is already a path.
        input_dir = Path(input_dir)
        if (input_dir / train_file).exists():
            db_train = input_dir / train_file
        else:
            raise FileNotFoundError(f"No trainings file in {input_dir}.")

        if (input_dir / test_file).exists():
            db_test = input_dir / test_file
        else:
            raise FileNotFoundError(f"No test file in {input_dir}.")

        return db_train, db_test

    def import_training_testing_data(input_dir=None, train_file=None, test_file=None):
        db_train, db_test = SpacyDataHandler._check_files(
            input_dir, train_file, test_file
        )
        db_files = [db_train, db_test]
        return db_files
