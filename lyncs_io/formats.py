""" List of formats supported by Lyncs IO """

__all__ = [
    "register",
    "formats",
]

from functools import wraps
import pickle
import numpy
from .format import Format, Formats

formats = Formats()


def register(name, *args, **kwargs):
    "Adds a format to the list of formats"
    formats[name] = Format(name, *args, **kwargs)


def swap(fnc):
    "Returns a wrapper that swaps the first two arguments of the function"
    return wraps(fnc)(
        lambda fname, data, *args, **kwargs: fnc(data, fname, *args, **kwargs)
    )


register(
    "pickle",
    extensions=["pkl"],
    modules={
        "pickle": pickle,
        "dill": "dill",
    },
    description="Python's pickle file format",
)

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

register(
    "ascii",
    extensions=["txt"],
    modules={
        "numpy": {"load": numpy.loadtxt, "save": swap(numpy.savetxt)},
    },
    description="ASCII format. Not recommended!",
)

register(
    "hdf5",
    extensions=["h5", "hdf5"],
    modules={
        "h5py": ".hdf5",
    },
    description="HDF5 file format",
    archive=True,
)

register(
    "numpy",
    extensions=["npy"],
    modules={
        "numpy": {"load": numpy.load, "save": swap(numpy.save)},
    },
    description="Numpy binary format",
)

register(
    "numpyz",
    extensions=["npz"],
    modules={
        "numpy": {"load": numpy.load, "save": swap(numpy.save)},
    },
    description="Numpy zip format",
)
