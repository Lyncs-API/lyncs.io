"""
Function utils
"""

from functools import wraps
from io import IOBase, FileIO
from pathlib import Path
from pathlib import Path
from lyncs_utils.io import FileLike


def find_file(filename):
    """
    Finds a file in the directory that has the same name
    as the parameter <filename>. If the file does not exist,
    the directory is searched for <filename.*> instead, and if
    only one match is found, that particular filename is returned.

    """

    if isinstance(filename, FileLike):
        return filename.name

    p = Path(filename)
    if p.exists():
        return filename

    # A list with files matching the following pattern: filename.*
    potential_files = [str(f) for f in p.parent.iterdir() if str(f).startswith(filename)]

    if len(potential_files) == 1:
        return str(potential_files[0])
    elif len(potential_files) > 1:
        raise ValueError(f'More than one {filename}.* exist')
    raise FileNotFoundError(f'No such file: {filename}, {filename}.*')


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
