""" List of formats supported by Lyncs IO """

__all__ = [
    "register",
    "formats",
]

import pickle
import json
from .format import Format, Formats
from .utils import swap, open_file
from . import numpy

formats = Formats()


def register(*names, **kwargs):
    "Adds a format to the list of formats"
    assert names
    fmt = Format(names[0], alias=names[1:], **kwargs)
    for name in names:
        formats[name.lower()] = fmt


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

register(
    "ASCII",
    "txt",
    extensions=["txt"],
    load=numpy.loadtxt,
    save=numpy.savetxt,
    description="ASCII, human-readable format. Limited to 1D or 2D arrays.",
)

register(
    "Numpy",
    extensions=["npy"],
    head=numpy.head,
    load=numpy.load,
    save=numpy.save,
    description="Numpy binary format",
)

register(
    "NumpyZ",
    extensions=["npz"],
    head=numpy.headz,
    load=numpy.loadz,
    save=numpy.savez,
    description="Numpy zip format",
    archive=True,
)

try:
    from . import hdf5

    register(
        "HDF5",
        extensions=["h5", "hdf5"],
        head=hdf5.head,
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
