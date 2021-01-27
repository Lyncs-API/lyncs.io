"""
Base tools for saving and loading data
"""

__all__ = [
    "load",
    "save",
    "dump",
    "formats",
]

from io import FileIO
from os import PathLike
from .format import Format
from .formats import formats


def load(filename, format=None, module=None, **kwargs):
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
    module: str, module, tuple, callable
        What module to use for the reading. See documentation for more details.
    """

    return get_format(format, filename=filename).load(filename, module=module, **kwargs)


def save(obj, filename, format=None, module=None, **kwargs):
    """
    Saves data into a file.

    Parameters
    ----------
    filename: str, file-like object
        The filename of the data file to write. It can also be a file-like object.
    format: str, Format
        One of the implemented formats. See documentation for more details.
    module: str, module, tuple, callable
        What module to use for the reading. See documentation for more details.
    """

    return get_format(format, filename=filename).save(
        obj, filename, module=module, **kwargs
    )


dump = save


def get_format(format=None, filename=None):
    "Return the appropriate format checking the format string or the filename extension."

    # 1. Using format
    if format:
        if isinstance(format, Format):
            return format
        if not isinstance(format, str):
            raise TypeError("Format should be a string.")

        for frmt in formats:
            if frmt == format:
                return frmt

        raise ValueError(f"Unknown format {format}")

    # 2. Using filename (checking extension)
    if isinstance(filename, PathLike):
        filename = filename.__fspath__()
    if isinstance(filename, FileIO):
        filename = filename.name
    if isinstance(filename, bytes):
        filename = filename.decode()
    if isinstance(filename, str):
        for frmt in formats:
            if frmt.check_filename(filename):
                return frmt

    raise ValueError(
        """
        Format could not be deduce from the filename. Please specify a format.
        Available formats are {", ".join((frmt.name for for frmt in formats))}.
        """
    )
