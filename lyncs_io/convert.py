"""
Converts Python data objects to an array buffer returning also a list of attributes
such that the array buffer can be converted back to the original data objects.
"""

from datetime import datetime
import numpy
from dask.array.core import Array as darr
from torch import Tensor, tensor
from .utils import (
    is_dask_array,
    is_sparse_matrix,
    from_reduced,
    in_torch_nn,
    layer_to_tensor,
    tensor_to_numpy,
)
from . import __version__


def reconstruct_reduced(attrs):
    "Reconstructs an object from the tuple returned by __reduce__"
    fnc, args, kwargs = attrs
    obj = fnc(*args)

    if hasattr(obj, "__setstate__"):
        obj.__setstate__(kwargs)
    else:
        obj.__dict__.update(kwargs)

    return obj


def get_attrs(data, flag=False):
    """
    Returns the list of attributes needed for reconstructing a data object
    """
    _dict = {
        "_lyncs_io": __version__,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": repr(type(data)),
    }

    _dict["type"] = type(data) if flag else _dict["type"]

    if _dict["type"] not in (Tensor, numpy.ndarray, darr, type(None)):

        if hasattr(data, "__reduce__"):
            return data.__reduce__()
        if hasattr(data, "__getstate__"):
            return _dict["type"], data.__getstate__()

        # No need for __dict__:
        # "If the method is absent, the instance’s __dict__ is pickled as usual"

    return _dict


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

    if is_sparse_matrix(type(data)):
        return data.toarray()

    if in_torch_nn(type(data)):
        return layer_to_tensor(tensor_to_numpy(data))

    if isinstance(data, Tensor):
        return tensor_to_numpy(data)

    return numpy.array(data)


def to_array(data):
    """
    Converts a data object to array. Returns also the list of attributes
    needed for reconstructing it.
    """
    attrs = get_attrs(data, flag=True)
    data = _to_array(data)

    if isinstance(attrs, dict):
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

    if from_reduced(attrs):
        return reconstruct_reduced(attrs)

    if isinstance(attrs, dict):
        if attrs["type"] == Tensor:
            return tensor(data)

    return data
