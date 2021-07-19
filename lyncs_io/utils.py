"""
Function utils
"""

from functools import wraps


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


def swap(fnc):
    "Returns a wrapper that swaps the first two arguments of the function"
    return wraps(fnc)(
        lambda fname, data, *args, **kwargs: fnc(data, fname, *args, **kwargs)
    )


def default_names(i=0):
    "Infinite generator of default names ('arrN') for entries of an archive."
    yield f"arr{i}"
    yield from default_names(i + 1)
