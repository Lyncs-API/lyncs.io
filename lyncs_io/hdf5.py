"""
Interface for HDF5 format based on h5py
"""

__all__ = [
    "head",
    "load",
    "save",
]

from h5py import File, Dataset, Group
from .archive import split_filename, Data, Loader, Archive
from .convert import to_array, from_array
from .header import Header
from .utils import default_names

from .dask_io import is_dask_array
from .decomposition import Decomposition


def _load_dataset(dts, header_only=False, comm=None, **kwargs):
    assert isinstance(dts, Dataset)

    attrs = Header(dts.attrs)
    attrs["shape"] = dts.shape
    attrs["dtype"] = dts.dtype

    if header_only:
        return attrs

    if comm is not None:
        _, subsizes, starts = Decomposition(comm=comm).decompose(dts.shape)
        slc = tuple(slice(start, start + size) for start, size in zip(starts, subsizes))
    else:
        slc = tuple(slice(size) for size in dts.shape)

    return from_array(dts[slc], attrs)


def _load(h5f, depth=1, header_only=False, comm=None, **kwargs):
    if isinstance(h5f, Group):
        return {
            key: _load(val, depth=depth - 1, comm=comm) if depth > 0 else None
            for key, val in h5f.items()
        }

    if isinstance(h5f, Dataset):
        header = _load_dataset(h5f, header_only=True, comm=comm, **kwargs)
        if header_only:
            return header
        return Data(header)

    raise TypeError(f"Unsupported {type(h5f)}")


def _load_dispatch(h5f, key, loader, comm=None, **kwargs):

    if key:
        h5f = h5f[key]

    if isinstance(h5f, Dataset):
        return _load_dataset(h5f, comm=comm, **kwargs)

    if isinstance(h5f, Group):
        return Archive(_load(h5f, comm=comm, **kwargs), loader=loader, path=key)

    raise TypeError(f"Unsupported {type(h5f)}")


def load(filename, key=None, chunks=None, comm=None, **kwargs):
    "Load function for HDF5"

    if comm is not None and chunks is not None:
        raise ValueError("chunks and comm parameters cannot be both set")

    filename, key = split_filename(filename, key)
    loader = Loader(load, filename, kwargs={"chunks": chunks, "comm": comm, **kwargs})

    if chunks is not None:
        raise NotImplementedError("DaskIO for HDF5 load not implemented yet.")

    if comm is not None:
        if not hasattr(comm, "size"):
            raise TypeError(
                "comm variable needs to be a valid MPI communicator with size attribute."
            )

        if comm.size > 1:
            with File(filename, "r", driver="mpio", comm=comm) as h5f:
                return _load_dispatch(h5f, key, loader, comm=comm, **kwargs)

    with File(filename, "r") as h5f:
        return _load_dispatch(h5f, key, loader, **kwargs)


def head(*args, comm=None, **kwargs):
    "Head function for HDF5"
    return load(*args, header_only=True, comm=comm, **kwargs)


def _write_dataset(grp, key, data, comm=None, **kwargs):
    "Writes a dataset in the group"
    if not key:
        for name in default_names():
            if name not in grp:
                key = name
                break

    if key in grp:
        # TBD: If key is a group do we want to write inside it
        #      OR overwrite it? Latter is the current behaviour
        #
        # if isinstance(grp[key], Group):
        #     return _write_dataset(grp[key], "", data, **kwargs)
        del grp[key]

    data, attrs = to_array(data)

    if comm is not None:
        global_shape, subsizes, starts = Decomposition(comm=comm).compose(data.shape)
        slc = tuple(slice(start, start + size) for start, size in zip(starts, subsizes))
        dset = grp.create_dataset(key, global_shape, dtype=data.dtype)
        dset[slc] = data
    else:
        grp.create_dataset(key, data=data)

    for attr, val in attrs.items():
        grp[key].attrs[attr] = val


def split_key(key):
    "Splits the key in group & dataset"
    tmp = key.lstrip("/").split("/")
    return "/" + "/".join(tmp[:-1]), tmp[-1]


def _save_dispatch(h5f, data, key, comm=None, **kwargs):
    group, dataset = split_key(key)
    h5f = h5f.require_group(group)
    return _write_dataset(h5f, dataset, data, comm=comm, **kwargs)


def save(data, filename, key=None, comm=None, **kwargs):
    "Save function for HDF5"
    filename, key = split_filename(filename, key)
    key = key or "/"

    if is_dask_array(data):
        raise NotImplementedError("DaskIO for HDF5 save not implemented yet.")

    if comm is not None:
        if not hasattr(comm, "size"):
            raise TypeError(
                "comm variable needs to be a valid MPI communicator with size attribute."
            )

        if comm.size > 1:
            with File(filename, "a", driver="mpio", comm=comm) as h5f:
                return _save_dispatch(h5f, data, key, comm=comm, **kwargs)

    with File(filename, "a") as h5f:
        return _save_dispatch(h5f, data, key, **kwargs)
