"""
Interface for HDF5 format based on h5py
"""

__all__ = [
    "head",
    "load",
    "save",
]

from collections.abc import Mapping
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
        return attrs, None

    if comm is not None:
        _, subsizes, starts = Decomposition(comm=comm).decompose(dts.shape)
        slc = tuple(slice(start, start + size) for start, size in zip(starts, subsizes))
    else:
        slc = tuple(slice(size) for size in dts.shape)

    return attrs, from_array(dts[slc], attrs)


def _load(h5f, depth=1, header_only=False, comm=None, all=False, **kwargs):
    if isinstance(h5f, Group):
        return {
            key: _load(val, depth=depth - 1, comm=comm, all=all)
            if all or depth > 0
            else None
            for key, val in h5f.items()
        }

    if isinstance(h5f, Dataset):
        header, data = _load_dataset(
            h5f, header_only=header_only or (not all), comm=comm, **kwargs
        )
        if header_only:
            return header
        return Data(header, data)

    raise TypeError(f"Unsupported {type(h5f)}")


def _load_dispatch(h5f, key, loader, comm=None, **kwargs):

    if key:
        h5f = h5f[key]

    if isinstance(h5f, Dataset):
        _, data = _load_dataset(h5f, comm=comm, **kwargs)
        return data

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


def _nested_mapping_iterator(mapping_obj):
    """This function accepts a nested dictionary as argument
    and iterate over all values of nested dictionaries
    """
    # Iterate over all key-value pairs of mapping argument
    for key, value in mapping_obj.items():
        # Check if value is of dict type
        if isinstance(value, Mapping):
            # If value is dict then iterate over all its values
            for pair in _nested_mapping_iterator(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            print(f"key: {key}")
            yield (key, value)


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


def join_dict_entries(pair):
    "Joins the nested mapping keys to form the group"
    return "/" + "/".join(pair[:-1]), pair[-1]


def split_key(key):
    "Splits the key in group & dataset"
    tmp = key.lstrip("/").split("/")
    return "/" + "/".join(tmp[:-1]), tmp[-1]


def _write(h5f, data, key, comm=None, all=False, **kwargs):

    if isinstance(data, Mapping):
        for pair in _nested_mapping_iterator(data):
            if all:
                group, value = join_dict_entries(pair)
                map_key = key + group
                group, dataset = split_key(map_key)
                h5f = h5f.require_group(group)
                _write_dataset(h5f, dataset, value, comm=comm, **kwargs)
            else:
                raise NotImplementedError(
                    "Writing only one entry of the dictionary not yet implemented"
                )
    else:
        group, dataset = split_key(key)
        print(key, group, dataset)
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
                return _write(h5f, data, key, comm=comm, **kwargs)

    with File(filename, "a") as h5f:
        return _write(h5f, data, key, **kwargs)
