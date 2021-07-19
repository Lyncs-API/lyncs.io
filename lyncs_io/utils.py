"""
Function utils
"""

from functools import wraps
from io import IOBase, FileIO
from pathlib import Path


def find_file(filename):
    """
    Finds a file in the directory that has the same name
    as the parameter <filename>. If the file does not exist,
    the directory is searched for <filename.*> instead, and if
    only one match is found, that particular filename is returned.

    """

    from os import listdir
    from os.path import dirname, abspath, splitext, exists, basename

    abs_path = abspath(filename)  # Absolute path of filename
    parent_dir_path = dirname(abs_path)  # Name of filename's parent directory

    if not exists(filename):
        # A list with files matching the following pattern: filename.*
        possible_files = [f for f in listdir(parent_dir_path) if splitext(f)[0] == basename(filename)]

        # If only one such file exists, load that.
        if len(possible_files) == 1:
            return possible_files[0]
        elif len(possible_files) > 1:
            raise Exception(f'More than one {filename}.* exist')
        elif len(possible_files) < 1:
            raise Exception(f'No such file: {filename}, {filename}.*')
    else:
        return filename


def swap(fnc):
    "Returns a wrapper that swaps the first two arguments of the function"
    return wraps(fnc)(
        lambda fname, data, *args, **kwargs: fnc(data, fname, *args, **kwargs)
    )


def open_file(fnc, arg=0, flag="rb"):
    "Returns a wrapper that opens the file (at position arg) if needed"

    @wraps(fnc)
    def wrapped(*args, **kwargs):
        if len(args) <= arg:
            raise ValueError(f"filename not found at position {arg}")
        if isinstance(args[arg], IOBase):
            return fnc(*args, **kwargs)
        args = list(args)
        with open(args[arg], flag) as fptr:
            args[arg] = fptr
            return fnc(*args, **kwargs)

    return wrapped


def to_path(filename):
    "Returns a Path from the filename"
    if isinstance(filename, FileIO):
        filename = filename.name
    if isinstance(filename, bytes):
        filename = filename.decode()
    return Path(filename)


def default_names(i=0):
    "Infinite generator of default names ('arrN') for entries of an archive."
    yield f"arr{i}"
    yield from default_names(i + 1)
