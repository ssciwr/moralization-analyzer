import importlib_metadata as metadata

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata
