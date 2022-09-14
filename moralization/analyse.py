from collections import defaultdict

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
def sort_spans(cas, ts):

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
            if span[cat] and span["KOMMENTAR"] != "Dopplung":

                span_dict[cat][span[cat]].append(span)

    # for span_dict_key, span_dict_sub_kat in span_dict.items():
    #     print(f"{span_dict_key}: {[key for key in span_dict_sub_kat.keys()]}")
    return span_dict
