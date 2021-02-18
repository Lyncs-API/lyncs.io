"""
Converts Python data objects to an array buffer returning also a list of attributes
such that the array buffer can be converted back to the original data objects.
"""

from datetime import datetime
from numpy import frombuffer, array
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


def to_array(data):
    """
    Converts a data object to array. Returns also the list of attributes
    needed for reconstructing it.
    """
    return array(data), get_attrs(data)


def to_bytes(data, order="C"):
    """
    Converts a data object to bytes. Returns also the list of attributes
    needed for reconstructing it.
    """
    arr, attrs = to_array(data)
    attrs.update(
        {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "bytes_order": order,
        }
    )
    return arr.tobytes(order), attrs


def from_bytes(data, attrs=None):
    """
    Converts bytes to a data object. Undoes to_bytes.
    """
    attrs = attrs or dict()
    dtype = attrs.get("dtype", None)
    shape = attrs.get("shape", None)
    order = attrs.get("bytes_order", None)
    # TODO: use order
    arr = frombuffer(data, dtype=dtype).reshape(shape)
    return from_array(arr, attrs)


def from_array(data, attrs=None):
    """
    Converts array to a data object. Undoes to_array.
    """
    # TODO
    return data
