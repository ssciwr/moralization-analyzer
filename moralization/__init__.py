import importlib_metadata as metadata
from moralization.input_data import InputOutput
from moralization.analyse import AnalyseOccurrence, AnalyseSpans
from moralization.plot import PlotSpans, InteractiveCategoryPlot
from moralization.spacy_model import SpacySetup, SpacyTraining

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata


__all__ = [
    "InputOutput",
    "AnalyseOccurrence",
    "AnalyseSpans",
    "PlotSpans",
    "InteractiveCategoryPlot",
    "SpacySetup",
    "SpacyTraining",
]
