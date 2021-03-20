""" List of formats supported by Lyncs IO """

__all__ = [
    "register",
    "formats",
]

import pickle
import json
from .format import Formats
from .utils import open_file
from . import numpy

formats = Formats()
register = formats.register


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

    _ = {
        "load": open_file(dill.load, 0, "rb"),
        "save": open_file(dill.dump, 1, "wb"),
    }

except ImportError as err:
    _ = {
        "error": err,
    }

register(
    "dill",
    extensions=["pkl", "dll"],
    description="Alternative to Python's pickle file format. Supports lambda functions.",
    **_,
)

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

    _ = {
        "head": hdf5.head,
        "load": hdf5.load,
        "save": hdf5.save,
    }

except ImportError as err:
    _ = {
        "error": err,
    }

register(
    "HDF5",
    extensions=["h5", "hdf5"],
    description="HDF5 file format",
    archive=True,
    **_,
)

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
