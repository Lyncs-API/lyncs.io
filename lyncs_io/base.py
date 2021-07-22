"""
Base tools for saving and loading data
"""

__all__ = [
    "load",
    "head",
    "save",
    "dump",
    "formats",
]

from .formats import formats
from .utils import find_file


def load(filename, format=None, **kwargs):
    """
    Loads data from a file.

    Parameters
    ----------
    filename: str, file-like object
        The filename of the data file to read. It can also be a file-like object.
    kwargs: dict
        Additional options for performing the reading. The list of options depends
        on the format.
    format: str, Format
        One of the implemented formats. See documentation for more details.
    """

    filename = find_file(filename)

    return formats.get_format(format, filename=filename).load(filename, **kwargs)


def head(filename, format=None, **kwargs):
    """
    Returns the header of a file. Reads the information about the content of the file
    without actually loading the data. Returns either an Header class or an Archive
    accordingly if the file contains a single object or it is an archive, respectively.

    Parameters
    ----------
    filename: str, file-like object
        The filename of the data file to read. It can also be a file-like object.
    format: str, Format
        One of the implemented formats. See documentation for more details.
    kwargs: dict
        Additional options for performing the reading. The list of options depends
        on the format.
    """

    filename = find_file(filename)

    return formats.get_format(format, filename=filename).head(filename, **kwargs)


def save(obj, filename, format=None, **kwargs):
    """
    Saves data into a file.

    Parameters
    ----------
    filename: str, file-like object
        The filename of the data file to write. It can also be a file-like object.
    format: str, Format
        One of the implemented formats. See documentation for more details.
    kwargs: dict
        Additional options for performing the writing. The list of options depends
        on the format.
    """

    return formats.get_format(format, filename=filename).save(obj, filename, **kwargs)


dump = save
