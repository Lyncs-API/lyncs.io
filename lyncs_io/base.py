"""
Base tools for saving and loading data
"""

__all__ = [
    "load",
    "save",
    "dump",
    "formats",
]

from .formats import formats


def load(filename, format=None, **kwargs):
    """
    Loads data from a file.

    Parameters
    ----------
    filename: str, file-like object
        The filename of the data file to read. It can also be a file-like object.
    kwargs: dict
        Additional options for performing the reading. The list of options depends
        on the format and the module.
    format: str, Format
        One of the implemented formats. See documentation for more details.
    """

    return formats.get_format(format, filename=filename).load(filename, **kwargs)


def save(obj, filename, format=None, **kwargs):
    """
    Saves data into a file.

    Parameters
    ----------
    filename: str, file-like object
        The filename of the data file to write. It can also be a file-like object.
    format: str, Format
        One of the implemented formats. See documentation for more details.
    """

    return formats.get_format(format, filename=filename).save(obj, filename, **kwargs)


dump = save
