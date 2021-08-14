"""
Interface for Tar format
"""
# disable import outside top-level warnings
# pylint: disable=C0415

# disable spelling warnings
# pylint: disable=C0401

import tarfile
from collections.abc import Mapping
from contextlib import contextmanager
from os import listdir
from os.path import exists, splitext, basename
from io import BytesIO
from .mpi_io import check_comm, tempdir_MPI
from .header import Header
from .archive import split_filename, Data, Archive, Loader
from .utils import (
    format_key,
    nested_dict,
    default_to_regular,
    find_member,
    get_depth,
)


_all_extensions = [
    ".tar",
    ".tar.bz2",
    ".tb2",
    ".tbz",
    ".tbz2",
    ".tz2",
    ".tar.gz",
    ".taz",
    ".tgz",
    ".tar.lz",
    ".tar.lzma",
    ".tlz",
    ".tar.lzo",
    ".tar.xz",
    ".txz",
    ".tar.Z",
    ".tZ",
    ".taZ",
    ".tar.zst",
    ".tzst"
    # Source: Wikipedia
]

# Format extensions for later use
all_extensions = [
    splitext(ext)[-1][1:] if sum([1 for x in ext if x == "."]) > 1 else ext[1:]
    for ext in _all_extensions
]

modes = {
    ":gz": [".gz", ".taz", ".tgz"],
    ":bz2": [".bz2", ".tb2", ".tbz", ".tbz2", ".tz2"],
    ":xz": [".xz", ".txz"],
    ":": [".tar"],
}


def _save(arr, tar, key, **kwargs):
    from . import base
    from .formats import formats

    _format = formats.get_format(filename=basename(key))
    key = key[1:] if key[0] == "/" else key

    if kwargs.get("comm", None) is not None:
        with tempdir_MPI(kwargs["comm"]) as temp:
            base.save(arr, temp + "/" + key, format=_format, **kwargs)
            # Only rank 0 does the writing
            if tar is not None:
                tar.add(temp + "/" + key, arcname=key)
    else:
        fptr = BytesIO()
        base.save(arr, fptr, format=_format, **kwargs)
        size = fptr.tell()  # get the size of the file object to write in the tarball
        fptr.seek(0)
        tarinfo = tarfile.TarInfo(name=key)
        tarinfo.size = size
        tar.addfile(tarinfo, fptr)


def _write_dispatch(arr, tar, key, **kwargs):
    if isinstance(arr, Mapping):
        for mkey, val in arr.items():
            _write_dispatch(val, tar, key + "/" + mkey, **kwargs)
    else:
        _save(arr, tar, key, **kwargs)


@contextmanager
def _open_for_saving(filename, mode_suffix, comm=None):
    if comm is not None and comm.rank != 0:
        tar = None
    elif exists(filename):
        if mode_suffix != ":":
            raise ValueError("Appending in a compressed tarball is not supported")
        tar = tarfile.open(filename, "a")
    else:
        tar = tarfile.open(filename, "w" + mode_suffix)

    yield tar

    if tar is not None:
        tar.close()
    if comm is not None:
        # make processes wait to avoid race conditions
        comm.Barrier()


def save(arr, filename, key=None, comm=None, **kwargs):
    """
    Save function for tar
    """
    filename, key = split_filename(filename, key)
    mode_suffix = _get_mode(filename)
    kwargs = {"comm": comm, **kwargs}

    if comm is not None:
        check_comm(kwargs["comm"])

    with _open_for_saving(filename, mode_suffix, comm=comm) as tar:
        _write_dispatch(arr, tar, key, **kwargs)


def _load_member(tar, member, header_only=False, as_data=False, **kwargs):
    from . import base
    from .formats import formats

    _format = formats.get_format(filename=basename(member.name))

    # 1. get buffer (extractfile) but causes fileno issues
    # 2. extract to a temporary file for parallel read
    # 3. read buffer (as is now)

    with _extract(tar, member, **kwargs) as fptr:

        header = Header(
            base.head(fptr, format=_format),
            **kwargs,
        )
        if header_only:
            return header

        _ = not kwargs["comm"] and fptr.seek(0)

        data = base.load(fptr, format=_format, **kwargs)

        return Data(header, data) if as_data else data


def _load(paths, tar, **kwargs):
    new_path_dict = nested_dict()
    for path in paths:
        parts = path.split("/")
        if parts:
            marcher = new_path_dict
            for key in parts[:-1]:
                marcher = marcher[key]
            header = _load_member(
                tar, find_member(tar, tar.getmember(path).name), as_data=True, **kwargs
            )
            marcher[parts[-1]] = header
    return default_to_regular(new_path_dict)


def _load_dispatch(tar, key, loader, depth=1, all_data=False, **kwargs):
    # if .h5
    if key and key.split("/")[0].endswith(".h5"):
        raise NotImplementedError(
            f"HDF5 file {key} is not supported when inside a tarball"
        )

    if is_dir(tar, format_key(key)):
        key = format_key(key)
        paths = [
            member.name
            for member in tar.getmembers()
            if (
                (member.name.startswith(key) or key == "/")
                and (get_depth(member.name, key) <= depth or all_data)
            )
        ]

        # avoid {dir : {data}}. Return {data} instead if key is given.
        _dict = (
            _load(paths, tar, **kwargs)[key[:-1]]
            if key != "/"
            else _load(paths, tar, **kwargs)
        )

        return Archive(_dict, loader=loader, path=key)

    return _load_member(tar, find_member(tar, key), **kwargs)


def load(filename, key=None, chunks=None, comm=None, **kwargs):
    """
    Load function for tar
    """

    if chunks and comm:
        raise ValueError("chunks and comm parameters cannot be both set")

    filename, key = split_filename(filename, key)
    mode_suffix = _get_mode(filename)
    kwargs = {"comm": comm, **kwargs}
    loader = Loader(load, filename, kwargs=kwargs)

    with tarfile.open(filename, "r" + mode_suffix) as tar:
        return _load_dispatch(tar, key, loader, **kwargs)


def head(*args, **kwargs):
    """
    Head function for tar
    """
    return load(*args, header_only=True, **kwargs)


def _get_mode(filename):
    """
    Returns the mode (from modes) with which the tarball should be read/written as
    """
    ext = splitext(filename)[-1]
    for key, val in modes.items():
        if ext in val:
            return key
    raise ValueError(f"{ext} is not supported.")


def is_dir(tar, key):
    """
    Check whether a member in a tarball is a directory
    """
    for member in tar.getmembers():
        if key == "/" or member.name.startswith(key):
            return True
    return False


@contextmanager
def _extract(tar, member, get_buff=False, **kwargs):
    if kwargs.get("comm", None) is not None:
        with tempdir_MPI(kwargs.get("comm")) as temp:
            check_comm(kwargs["comm"])
            if kwargs["comm"].rank == 0:
                tar.extract(member, path=temp)
            kwargs["comm"].Barrier()
            yield temp + "/" + listdir(temp)[0]
    elif get_buff:
        yield tar.extractfile(member)
    else:
        # This is what's used at the moment
        # get_buff raises errors ('fileno')
        fptr = BytesIO()
        fptr.write(tar.extractfile(member).read())
        fptr.seek(0)
        yield fptr
