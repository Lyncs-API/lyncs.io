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

    if filename:
        from os import listdir
        from os.path import dirname, abspath, splitext

        abs_path = abspath(filename) # Absolute path of filename
        ext = splitext(filename)[1] # Get the extension of the filename
        parent_dir_path = dirname(abs_path) # Name of filename's parent directory

        if not ext: 
            dir_files = listdir(parent_dir_path)
            if filename not in dir_files:
                # A list with files matching the following pattern: filename.*
                possible_files = [f for f in dir_files if splitext(f)[0] == filename]

                # If only one such file exists, load that.
                if len(possible_files) == 1:
                    filename = possible_files[0]
                else:
                    raise Exception(f'More than one {filename}.* exist')

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
