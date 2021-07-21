"""
Interface for HDF5 format based on h5py
"""

__all__ = [
    "head",
    "load",
    "save",
]

from collections.abc import Mapping
from h5py import File, Dataset, Group, h5
from .archive import split_filename, Data, Loader, Archive
from .convert import to_array, from_array
from .header import Header
from .utils import default_names, is_dask_array
from .mpi_io import check_comm
from .decomposition import Decomposition

mpi = h5.get_config().mpi


def _load_dataset(dts, header_only=False, comm=None, **kwargs):
    assert isinstance(dts, Dataset)
    assert not kwargs, f"Unknown parameters {kwargs}"

    attrs = Header(dts.attrs)
    attrs["shape"] = dts.shape
    attrs["dtype"] = dts.dtype

    if header_only:
        return attrs, None

    if comm is not None:
        _, subsizes, starts = Decomposition(comm=comm).decompose(dts.shape)
        slc = tuple(slice(start, start + size) for start, size in zip(starts, subsizes))
    else:
        slc = tuple(slice(size) for size in dts.shape)

    return attrs, from_array(dts[slc], attrs)


def _load(h5f, depth=1, header_only=False, all_data=False, **kwargs):
    if isinstance(h5f, Group):
        return {
            key: _load(val, depth=depth - 1, all_data=all_data, **kwargs)
            if all_data or depth > 0 or isinstance(val, Dataset)
            else None
            for key, val in h5f.items()
        }

    if isinstance(h5f, Dataset):
        header, data = _load_dataset(
            h5f, header_only=header_only or (not all_data), **kwargs
        )
        if header_only:
            return header
        return Data(header, data)

    raise TypeError(f"Unsupported {type(h5f)}")


def _load_dispatch(h5f, key, loader, header_only=False, **kwargs):

    if key:
        h5f = h5f[key]

    if isinstance(h5f, Dataset):
        header, data = _load_dataset(h5f, **kwargs)
        if header_only:
            return header

        return data

    if isinstance(h5f, Group):
        return Archive(_load(h5f, **kwargs), loader=loader, path=key)

    raise TypeError(f"Unsupported {type(h5f)}")


def load(filename, key=None, chunks=None, comm=None, **kwargs):
    "Load function for HDF5"
    if comm is not None and chunks is not None:
        raise ValueError("chunks and comm parameters cannot be both set")

    filename, key = split_filename(filename, key)
    # Append comm and chunks in kwargs
    kwargs = {"comm": comm, **kwargs}
    loader = Loader(load, filename, kwargs=kwargs)

    if chunks is not None:
        raise NotImplementedError("DaskIO for HDF5 load not implemented yet.")

    if comm is not None:
        check_comm(comm)

        if comm.size > 1:
            with File(filename, "r", driver="mpio", comm=comm) as h5f:
                return _load_dispatch(h5f, key, loader, **kwargs)

    with File(filename, "r") as h5f:
        return _load_dispatch(h5f, key, loader, **kwargs)


def head(*args, **kwargs):
    "Head function for HDF5"
    return load(*args, header_only=True, **kwargs)


def _write_dataset(grp, key, data, comm=None, **kwargs):
    "Writes a dataset in the group"
    assert not kwargs, f"Unknown parameters {kwargs}"

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
        try:
            grp[key].attrs[attr] = val
        except TypeError:
            grp[key].attrs[attr] = str(val)


def split_key(key):
    "Splits the key in group & dataset"
    tmp = key.lstrip("/").split("/")
    return "/" + "/".join(tmp[:-1]), tmp[-1]


def _write(h5f, data, key, **kwargs):
    "Ensures a group is created and the dataset is written"
    group, dataset = split_key(key)
    h5f = h5f.require_group(group)
    return _write_dataset(h5f, dataset, data, **kwargs)


def _write_dispatch(h5f, data, key, **kwargs):
    if isinstance(data, Mapping):
        for map_key, val in data.items():
            _write_dispatch(h5f, val, key + "/" + map_key, **kwargs)
    else:
        _write(h5f, data, key, **kwargs)


def save(data, filename, key=None, comm=None, **kwargs):
    "Save function for HDF5"
    filename, key = split_filename(filename, key)
    key = key or "/"
    # Append comm in kwargs
    kwargs = {"comm": comm, **kwargs}

    if is_dask_array(data):
        raise NotImplementedError("DaskIO for HDF5 save not implemented yet.")

    if comm is not None:
        check_comm(comm)

        # TODO: restore only the case with "a" when the issue in h5py is fixed
        if comm.size > 1:
            try:
                with File(filename, "a", driver="mpio", comm=comm) as h5f:
                    return _write_dispatch(h5f, data, key, **kwargs)
            except OSError:
                with File(filename, "w", driver="mpio", comm=comm) as h5f:
                    return _write_dispatch(h5f, data, key, **kwargs)

    with File(filename, "a") as h5f:
        return _write_dispatch(h5f, data, key, **kwargs)
