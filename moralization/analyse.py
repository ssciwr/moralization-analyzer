from collections import defaultdict
import pandas as pd
import numpy as np

map_expressions = {
    "KAT1-Moralisierendes Segment": "KAT1MoralisierendesSegment",
    "KAT2-Moralwerte": "Moralwerte",
    "KAT2-Subjektive_Ausdrücke": "KAT2Subjektive_Ausdrcke",
    "KAT3-Gruppe": "Protagonistinnen2",
    "KAT3-Rolle": "Protagonistinnen",
    "KAT3-own/other": "Protagonistinnen3",
    "KAT4-Kommunikative Funktion": "KommunikativeFunktion",
    "KAT5-Forderung_explizit": "Forderung",
    "KAT5-Forderung_implizit": "KAT5Ausformulierung",
    "Kommentar": "KOMMENTAR",
}


# select all custom Spans and store them in an ordered dict,
# where the first dimension is the used inception category (Protagonistinnen, Forderung, etc...)
# and the second dimension is the corresponding value of this category ('Forderer:in', 'Adresassat:in', 'Benefizient:in')
# dict[category][entry value] = span
def sort_spans(cas: object, ts: object) -> dict:

    span_type = ts.get_type("custom.Span")
    span_dict = defaultdict(lambda: defaultdict(list))

    # list of all interesting categories
    # as there are:
    # KAT1MoralisierendesSegment - Moralisierung explizit, Moralisierung Kontext, Moralisierung Weltwissen, Moralisierung interpretativ, Keine Moralisierung
    # Moralwerte - Care, Harm, Fairness, Cheating, …
    # KAT2Subjektive_Ausdrcke - Care, Harm, Fairness, Cheating, …
    # Protagonistinnen2 - Individuum, Menschen, Institution, Soziale Gruppe, OTHER
    # Protagonistinnen - Forderer:in, Adressat:in, Benefizient:in, Kein Bezug
    # Protagonistinnen3 - Own group, Other group, Neutral
    # KommunikativeFunktion - Darstellung, Appell, Expression, Beziehung, OTHER
    # Forderung - explizit
    # KAT5Ausformulierung - implizit
    # KOMMENTAR - tag for duplicates that should be excluded

    cat_list = [
        "KAT1MoralisierendesSegment",
        "Moralwerte",
        "KAT2Subjektive_Ausdrcke",
        "Protagonistinnen2",
        "Protagonistinnen",
        "Protagonistinnen3",
        "KommunikativeFunktion",
        "Forderung",
        "KAT5Ausformulierung",
        "KOMMENTAR",
    ]

    for span in cas.select(span_type.name):

        for cat in cat_list:
            # this excludes any unwanted datapoints
            # also ignore the ones with no moralization
            if (
                span[cat]
                and span["KOMMENTAR"] != "Dopplung"
                and span[cat] != "Keine Moralisierung"
            ):
                # Here we could also exclude unnecessary information
                span_dict[cat][span[cat]].append(span)

    # for span_dict_key, span_dict_sub_kat in span_dict.items():
    #     print(f"{span_dict_key}: {[key for key in span_dict_sub_kat.keys()]}")
    return span_dict


# find the overlaying category for an second dimension cat name
def find_cat_from_str(cat_entry, span_dict):

    for span_dict_key, span_dict_sub_kat in span_dict.items():
        if cat_entry in span_dict_sub_kat.keys():
            return span_dict_key


# get overlap%
# so far this only works on a span basis and not a sentence basis.
def get_overlap_percent(cat_1, cat_2, data_dict, file_name, ret_occ=False):
    o_cat1 = find_cat_from_str(cat_1, data_dict[file_name]["data"])
    o_cat2 = find_cat_from_str(cat_2, data_dict[file_name]["data"])

    occurence = 0
    total = 0
    for span in data_dict[file_name]["data"][o_cat1][cat_1]:
        total += 1
        if span[o_cat2] == cat_2:
            occurence += 1
    if ret_occ:
        return occurence, total
    else:
        return round(occurence / total, 7)


def get_percent_matrix(data_dict, file_name, cat_list=None):
    if cat_list is None:
        cat_list = []
        for span_dict_key, span_dict_sub_kat in data_dict[file_name]["data"].items():
            [cat_list.append(key) for key in span_dict_sub_kat.keys()]

    percent_matrix = np.zeros((len(cat_list), len(cat_list)))
    for i, cat1 in enumerate(cat_list):
        for j, cat2 in enumerate(cat_list):
            percent_matrix[i, j] = get_overlap_percent(cat1, cat2, data_dict, file_name)

    df = pd.DataFrame(percent_matrix, index=cat_list)
    df.columns = cat_list
    return df

    # mode can be "instances" or "span"


# instances reports the number of occurences and span


def report_instances(data_dict, file_names=None):

    if file_names is None:
        file_names = list(data_dict.keys())
    elif isinstance(file_names, str):
        file_names = [file_names]

    # filename: main_cat: sub_cat: instances
    instance_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for file_name in file_names:
        span_dict = data_dict[file_name]["data"]
        # initilize total instances rows for easier setting later.
        instance_dict[file_name][("total instances")] = 0

        for main_cat_key, main_cat_value in span_dict.items():
            for sub_cat_key, sub_cat_value in main_cat_value.items():
                # the tuple index makes it easy to convert the dict into a pandas dataframe
                instance_dict[file_name][(main_cat_key, sub_cat_key)] = len(
                    sub_cat_value
                )
    print(instance_dict)
    df = pd.DataFrame(instance_dict)
    df.index = df.index.set_names((["Main Category", "Sub Category"]))

    # add rows for total instances
    df.loc[("total instances"), :] = df.sum(axis=0).values
    # df.loc[("total instances", "without invalid"), :] = (
    # df.loc[("total instances", "with invalid"), :].values
    # - df.loc["KAT1MoralisierendesSegment", "Keine Moralisierung"].values
    # )

    # sort by index and occurence number
    df = df.sort_values(
        by=[
            "Main Category",
            "Sub Category",
            file_names[0],
        ],
        ascending=False,
    )

    # fill NaN
    df = df.fillna(0)
    return df


def report_spans(data_dict_list, file_names=None):

    if file_names is None:
        file_names = list(data_dict_list.keys())
    elif isinstance(file_names, str):
        file_names = [file_names]

    df_spans = report_instances(data_dict_list, file_names)
    # this report_instances call makes it much easier to include the total number of spans for each columns, as well as removes the need to duplicate the pandas setup.

    df_spans[:] = df_spans[:].astype("object")
    for file_name in file_names:
        span_dict = data_dict_list[file_name]["data"]

        for main_cat_key, main_cat_value in span_dict.items():
            for sub_cat_key, sub_cat_value in main_cat_value.items():

                # multiple options for how to report the spans are available

                # first report the entire span object as a string
                span_list = [str(span) for span in span_dict[main_cat_key][sub_cat_key]]
                # this would look like this:
                # c.Span(Protagonistinnen=Forderer:in, Protagonistinnen2=Individuum, Protagonistinnen3=Own Group, begin=21822, end=21874);
                # c.Span(Protagonistinnen=Benefizient:in, Protagonistinnen2=Institution, Protagonistinnen3=Own Group, begin=21974, end=21984);
                # c.Span(Protagonistinnen=Forderer:in, Protagonistinnen2=Institution, Protagonistinnen3=Own Group, begin=66349, end=66352)
                # maybe one should remove the c.Span() but i'm not sure what exactly is wanted here.

                # second option is to report the end or beginning index for each span
                # span_list=[str(span["end"]) for span in span_dict[main_cat_key][sub_cat_key] ]

                # convert list to seperated str

                span_str = ";".join(span_list)
                span_str = span_str.replace("[", "").replace("]", "")

                df_spans.at[
                    (main_cat_key, sub_cat_key),
                    file_name,
                ] = span_str

    return df_spans
