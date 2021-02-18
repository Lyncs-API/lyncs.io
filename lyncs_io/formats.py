""" List of formats supported by Lyncs IO """

__all__ = [
    "register",
    "formats",
]

from functools import wraps
from io import IOBase
import pickle
import json
from .format import Format, Formats

formats = Formats()


def register(*names, **kwargs):
    "Adds a format to the list of formats"
    assert names
    fmt = Format(names[0], alias=names[1:], **kwargs)
    for name in names:
        formats[name.lower()] = fmt


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


register(
    "pickle",
    extensions=["pkl"],
    load=open_file(pickle.load, 0, "rb"),
    save=open_file(pickle.dump, 1, "wb"),
    description="Python's pickle file format. See https://docs.python.org/3/library/pickle.html.",
)

# TODO: add collection of JSON encoders, e.g. numpyencoder
register(
    "JSON",
    extensions=["json"],
    load=open_file(json.load, 0, "rb"),
    save=open_file(json.dump, 1, "wb"),
    description="Python's JSON file format. See https://docs.python.org/3/library/json.html.",
)

try:
    import dill

    register(
        "dill",
        extensions=["pkl", "dll"],
        load=open_file(dill.load, 0, "rb"),
        save=open_file(dill.dump, 1, "wb"),
        description="Alternative to Python's pickle file format. Supports lambda functions.",
    )
except ImportError:
    pass

try:
    import numpy

    register(
        "ASCII",
        extensions=["txt"],
        load=numpy.loadtxt,
        save=swap(numpy.savetxt),
        description="ASCII, human-readable format. Limited to 1D or 2D arrays.",
    )

    register(
        "Numpy",
        extensions=["npy"],
        load=numpy.load,
        save=swap(numpy.save),
        description="Numpy binary format",
    )

    register(
        "NumpyZ",
        extensions=["npz"],
        load=numpy.load,
        save=swap(numpy.save),
        description="Numpy zip format",
        archive=True,
    )
except ImportError:
    pass

try:
    from . import hdf5

    register(
        "HDF5",
        extensions=["h5", "hdf5"],
        load=hdf5.load,
        save=hdf5.save,
        description="HDF5 file format",
        archive=True,
    )
except ImportError:
    pass


"""
register(
    "lime",
    extensions=["lime"],
    modules={
        "lime": "lime",
        "clime": "lyncs_clime",
    },
    description="LQCD lime format",
    archive=True,
)
"""
