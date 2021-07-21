"""
Function utils
"""

from functools import wraps
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
        return filename

    path = Path(filename)
    if path.exists():
        return filename

    # Most probably is an archive
    if not path.parent.is_dir():
        return filename

    # A list with files matching the following pattern: filename.*
    potential_files = [
        str(f) for f in path.parent.iterdir() if str(f).startswith(str(path))
    ]

    if len(potential_files) == 1:
        return str(potential_files[0])
    if len(potential_files) > 1:
        raise ValueError(f"More than one {filename}.* exist")
    raise FileNotFoundError(f"No such file: {filename}, {filename}.*")


def is_dask_array(obj):
    """
    Function for checking if passed object is a dask Array
    """
    try:
        # pylint: disable=C0415
        from dask.array import Array

        return isinstance(obj, Array)
    except ImportError:
        return False


def swap(fnc):
    "Returns a wrapper that swaps the first two arguments of the function"
    return wraps(fnc)(
        lambda fname, data, *args, **kwargs: fnc(data, fname, *args, **kwargs)
    )


def default_names(i=0):
    "Infinite generator of default names ('arrN') for entries of an archive."
    yield f"arr{i}"
    yield from default_names(i + 1)
