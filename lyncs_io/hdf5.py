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


def _load_dataset(dts, header_only=False, **kwargs):
    assert isinstance(dts, Dataset)

    attrs = Header(dts.attrs)
    attrs["shape"] = dts.shape
    attrs["dtype"] = dts.dtype

    if header_only:
        return attrs

    return from_array(dts[:], attrs)


def _load(h5f, depth=1, header_only=False, **kwargs):
    if isinstance(h5f, Group):
        return {
            key: _load(val, depth=depth - 1) if depth > 0 else None
            for key, val in h5f.items()
        }

    if isinstance(h5f, Dataset):
        header = _load_dataset(h5f, header_only=True, **kwargs)
        if header_only:
            return header
        return Data(header)

    raise TypeError(f"Unsupported {type(h5f)}")


def load(filename, key=None, **kwargs):
    "Load function for HDF5"
    filename, key = split_filename(filename, key)

    loader = Loader(load, filename, kwargs=kwargs)

    with File(filename, "r") as h5f:
        if key:
            h5f = h5f[key]

        if isinstance(h5f, Dataset):
            return _load_dataset(h5f, **kwargs)

        if isinstance(h5f, Group):
            return Archive(
                _load(h5f, **kwargs),
                loader=loader,
                path=key,
            )

        raise TypeError(f"Unsupported {type(h5f)}")


def head(*args, **kwargs):
    "Head function for HDF5"
    return load(*args, header_only=True, **kwargs)


def _write_dataset(grp, key, data, **kwargs):
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
    grp.create_dataset(key, data=data)

    for attr, val in attrs.items():
        grp[key].attrs[attr] = val


def split_key(key):
    "Splits the key in group & dataset"
    tmp = key.lstrip("/").split("/")
    return "/" + "/".join(tmp[:-1]), tmp[-1]


def save(data, filename, key=None, **kwargs):
    "Save function for HDF5"
    filename, key = split_filename(filename, key)
    key = key or "/"

    with File(filename, "a") as h5f:
        group, dataset = split_key(key)
        h5f = h5f.require_group(group)
        return _write_dataset(h5f, dataset, data, **kwargs)
