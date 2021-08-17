"""
Converts Python data objects to an array buffer returning also a list of attributes
such that the array buffer can be converted back to the original data objects.
"""

from datetime import datetime
import numpy
from pandas import DataFrame
from scipy import sparse
from .utils import is_dask_array, is_sparse_matrix
from . import __version__


<<<<<<< HEAD
=======
def array_to_coo(data):
    return sparse.coo_matrix(data)


def array_to_csc(data):
    return sparse.csc_matrix(data)


def array_to_csr(data):
    return sparse.csr_matrix(data)


>>>>>>> 02b069d08fb1cc443fe80101102701f0e4d56b60
def array_to_df(data, attrs):
    # attrs = (
    #             <function _reconstructor>,
    #             (
    #                 <class dataframe>,
    #                 <class object>,
    #                 None,
    #             ),
    #             {_mgr : BlockManager...}
    #         )

    instance = attrs[1][0]
    index = [x for x in attrs[2]["_mgr"].items]
    df_data = dict(zip(index, data.T))
    return instance(df_data)


def get_attrs(data, flag=False):
    """
    Returns the list of attributes needed for reconstructing a data object
    """
    _dict = {
        "_lyncs_io": __version__,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": repr(type(data)),
    }

    if flag:
        _dict["type"] = type(data)

    if _dict["type"] == DataFrame:
        return data.__reduce__()

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

    return numpy.array(data)


def to_array(data):
    """
    Converts a data object to array. Returns also the list of attributes
    needed for reconstructing it.
    """
    attrs = get_attrs(data, flag=True)
    data = _to_array(data)
    if type(attrs) != dict and attrs[1][0] != DataFrame:
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
    if type(attrs) == dict and attrs.get('type') is not None:
        if is_sparse_matrix(attrs['type']):
            return attrs['type'](data)

    if attrs[1][0] == DataFrame:
        return array_to_df(data, attrs)
