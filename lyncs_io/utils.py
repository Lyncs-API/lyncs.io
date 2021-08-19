"""
Function utils
"""

from functools import wraps
from pathlib import Path
from os.path import splitext
from inspect import getmembers
from collections import defaultdict
from warnings import warn
import torch.nn
from lyncs_utils.io import FileLike
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    coo_matrix,
    bsr_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
)


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


""" !!!!!!!!!!! """


# def check_suport(obj):
#     "Checks whether the object's type is supported"


def in_torch_nn(obj):
    "Checks if an object belongs in the torch.nn module (Layers)"
    return obj in getmembers(torch.nn)


def layer_to_tensor(layer):
    "Converts a torch layer to a tensor"
    _, _, kwargs = layer.__reduce__()
    params = kwargs["_parameters"]
    items = list(params.items())
    param = items[0][1]
    return param[:]


def layers_are_equal(layer1, layer2):
    "Compare two layers. Using double equals is inappropriate"
    return layer1.__reduce__() == layer2.__reduce__()


def tensor_to_numpy(tensor):
    "Converts a tensor to a numpy array"
    return tensor.detach().numpy()


def is_sparse_matrix(obj):
    "Check whether an object is a sparse matrix"
    return obj in (
        csc_matrix,
        csr_matrix,
        coo_matrix,
        bsr_matrix,
        dia_matrix,
        dok_matrix,
        lil_matrix,
    )


def from_state(attrs):
    "Check whether an object matches the tuple's format returned by __getstate__"
    return (
        isinstance(attrs, tuple)
        and len(attrs) == 2
        and callable(attrs[0])
        and isinstance(type(attrs[1]), dict)
    )


def from_reduced(attrs):
    "Returns whether an object matches the tuple's format returned by __reduce__"
    return (
        isinstance(attrs, tuple)
        and len(attrs) == 3
        and callable(attrs[0])
        and isinstance(attrs[1], tuple)
        and isinstance(attrs[2], dict)
    )


""" !!!!!!!!!!! """


def swap(fnc):
    "Returns a wrapper that swaps the first two arguments of the function"
    return wraps(fnc)(
        lambda fname, data, *args, **kwargs: fnc(data, fname, *args, **kwargs)
    )


def default_names(i=0):
    "Infinite generator of default names ('arrN') for entries of an archive."
    yield f"arr{i}"
    yield from default_names(i + 1)


def nested_dict():
    """
    Creates a default dictionary where each value is an other default dictionary.
    """
    return defaultdict(nested_dict)


def default_to_regular(dic):
    """
    Convert a dictionary from default to regular
    """
    if isinstance(dic, defaultdict):
        dic = {k: default_to_regular(v) for k, v in dic.items()}
    return dic


def find_member(tar, key):
    """
    Similar to find_file but with members in a taball.
    """
    if splitext(key)[-1]:
        member = tar.getmember(key)
    else:
        potential_members = [
            f.name for f in tar.getmembers() if splitext(f.name)[0] == key
        ]
        num = len(potential_members)
        if num == 1:
            member = tar.getmember(potential_members[0])
        elif num > 1:
            raise KeyError(
                f"Can't omit extension when multiple files with the same name exist: {','.join(potential_members)}"
            )
        else:
            raise KeyError(f"No such file: {key}, {key}.*")
    return member


def get_depth(path, key):
    """
    Returns the depth of a key relatively to a path.
    E.g. /bar1/bar2/bar3/foo.npy is at depth=2 relatively to /bar1/bar2/
    """
    path = str(Path(path))
    key = str(Path(key)) + "/" if key != "/" else str(Path(key))
    key_depth = sum([1 for char in key if char == "/"])
    path_depth = sum([1 for char in path if char == "/"])
    diff = (
        path_depth
        - key_depth
        + sum([2 for c in key.split("/") if c == ".."])
        + sum([1 for c in key.split("/") if c == "."])
    )
    # key[0] will never be "/" since absolute paths cannot be used in a tarball
    return diff + 1 if key[0] == "/" else diff


def format_key(key):
    """
    Format the key provided for consistency.
    """
    if key:
        return key if key[-1] == "/" else key + "/"
    return "/"
