"""
Interface for Tar format
"""

import tarfile
import tempfile
from .mpi_io import check_comm
from io import BytesIO
from os.path import exists, splitext, basename
from .header import Header
from .archive import split_filename, Data, Archive, Loader
from .utils import (
    format_key,
    nested_dict,
    default_to_regular,
    default_names,
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

# format extension to be used by formats.get_format
all_extensions = [
    splitext(ext)[-1][1:] if sum([1 for x in ext if x == "."]) > 1 else ext[1:]
    for ext in _all_extensions
]

modes = {
    # .gz, .bz2 and .xz should start with .tar
    # .tar was removed because of splitext()
    ":gz": [".gz", ".taz", ".tgz"],
    ":bz2": [".bz2", ".tb2", ".tbz", ".tbz2", ".tz2"],
    ":xz": [".xz", ".txz"],
    ":": [".tar"],
}

# TODO: fix issues with compression in append mode
# TODO: issues with h5 files
# TODO: deduce data format from the data argument (for saving with a default key)
# TODO: test parallel save/load


def _save(arr, tar, key, **kwargs):
    from . import base
    from .formats import formats

    # if comm -> filename
    # else -> bytesio

    fptr = BytesIO()
    base.save(arr, fptr, format=formats.get_format(filename=basename(key)), **kwargs)
    size = fptr.tell()  # get the size of the file object to write in the tarball
    fptr.seek(0)
    tarinfo = tarfile.TarInfo(name=key)
    tarinfo.size = size

    # addfile v add (comm(one process extracts, the rest of the processes read)) depends on fptr
    tar.addfile(tarinfo, fptr)
    tar.close()


def save(arr, filename, key=None, comm=None, **kwargs):
    """
    Save function for tar
    """
    filename, key = split_filename(filename, key)
    mode_suffix = _get_mode(filename)
    kwargs = {"comm": comm, **kwargs}

    # create Tar if doesn't exist - append if it does
    if exists(filename):
        tar = tarfile.open(filename, "a")
    else:
        tar = tarfile.open(filename, "w" + mode_suffix)

    if not key:
        for name in default_names():
            if name + ".npy" not in [m.name for m in tar.getmembers()]:
                key = name + ".npy"
                break

    _save(arr, tar, key, **kwargs)


def _load_member(tar, member, header_only=False, as_data=False, **kwargs):
    from . import base
    from .formats import formats

    # 1. get buffer (extractfile (fileno issues)),
    # 2. extract to a temp file,
    # 3. read buffer (as it's now),
    # 4. do nothing/wait (one process extracts, the rest of the processes read)

    temp = None
    if kwargs['comm']:
        temp = tempfile.TemporaryDirectory()

    fptr = _extract(tar, member, temp=temp, **kwargs)
    
    header = Header(
        base.head(fptr, format=formats.get_format(filename=basename(member.name))),
        **kwargs,
    )
    if header_only:
        return header

    not kwargs['comm'] and fptr.seek(0)

    data = base.load(
        fptr, format=formats.get_format(filename=basename(member.name)), **kwargs
    )
        
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

def _extract(tar, member, get_buff=False, wait=False, temp=None, **kwargs):
    import os
    from . import base

    if get_buff:
        raise NotImplementedError

    if wait:
        raise NotImplementedError

    if kwargs['comm']:
        check_comm(kwargs['comm'])
        tar.extract(member, path=temp.name)
        return temp.name +'/'+ os.listdir(temp.name)[0]
    
    fptr = BytesIO()
    fptr.write(tar.extractfile(member).read())
    fptr.seek(0)
    return fptr




