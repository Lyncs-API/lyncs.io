"""
Function utils
"""

from functools import wraps
from io import IOBase, FileIO
from pathlib import Path
import os


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


def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as fptr:
        os.utime(
            fptr.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd,
            **kwargs,
        )


def exists(fname):
    return os.path.exists(fname)
