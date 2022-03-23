"""
Converts Python data objects to an array buffer returning also a list of attributes
such that the array buffer can be converted back to the original data objects.
"""

from datetime import datetime
import numpy
from .utils import is_dask_array
from . import __version__


def get_attrs(data):
    """
    Returns the list of attributes needed for reconstructing a data object
    """
    return {
        "_lyncs_io": __version__,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": repr(type(data)),
    }


def get_array_attrs(data):
    "Returns attributes of an array"
    if is_dask_array(data):
        fortran_order = False
    else:
        fortran_order = numpy.isfortran(data)
    return {
        "shape": data.shape,
        "dtype": data.dtype,
        "fortran_order": fortran_order,
        "descr": numpy.lib.format.dtype_to_descr(data.dtype),
        "nbytes": data.nbytes,
    }


def _to_array(data):
    "Converts data to array"
    if is_dask_array(data):
        return data
    return numpy.array(data, copy=False, order="C")


def to_array(data):
    """
    Converts a data object to array. Returns also the list of attributes
    needed for reconstructing it.
    """
    attrs = get_attrs(data)
    data = _to_array(data)
    attrs.update(get_array_attrs(data))
    return data, attrs


def to_bytes(data):
    """
    Converts a data object to bytes. Returns also the list of attributes
    needed for reconstructing it.
    """
    arr, attrs = to_array(data)
    return arr.tobytes(order="F" if attrs["fortran_order"] else "C"), attrs


def from_bytes(data, attrs=None):
    """
    Converts bytes to a data object. Undoes to_bytes.
    """
    attrs = attrs or dict()
    dtype = attrs.get("dtype", None)
    shape = attrs.get("shape", None)
    fortran_order = attrs.get("fortran_order", None)
    arr = numpy.frombuffer(data, dtype=dtype).reshape(
        shape, order="F" if fortran_order else "C"
    )
    return from_array(arr, attrs)


def from_array(data, attrs=None):
    """
    Converts array to a data object. Undoes to_array.
    """
    # TODO
    return data
