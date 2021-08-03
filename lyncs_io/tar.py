"""
Interface for Tar format
"""

from os.path import exists, splitext, basename
from .archive import split_filename
from io import BytesIO
import tarfile
import tempfile
from .utils import (format_key,
                    is_dir,
                    nested_dict,
                    default_to_regular,
                    default_names,
                    find_member,
                    get_depth)


_all_extensions = [
    '.tar',
    '.tar.bz2', '.tb2', '.tbz', '.tbz2', '.tz2',
    '.tar.gz', '.taz', '.tgz',
    '.tar.lz',
    '.tar.lzma', '.tlz',
    '.tar.lzo',
    '.tar.xz', '.txz',
    '.tar.Z', '.tZ', '.taZ',
    '.tar.zst', '.tzst'
    # source: wikipedia
]

# format extension to be used by formats.get_format
all_extensions = [splitext(ext)[-1][1:]
                  if sum([1 for x in ext if x == '.']) > 1
                  else ext[1:]
                  for ext in _all_extensions]

modes = {
    # .gz, .bz2 and .xz should start with .tar
    # .tar was removed because of splitext()
    ':gz': ['.gz', '.taz', '.tgz'],
    ':bz2': ['.bz2', '.tb2', '.tbz', '.tbz2', '.tz2'],
    ':xz': ['.xz', '.txz'],
    ':': ['.tar'],
}

# TODO: fix issues with compression in append mode
# TODO: issues with h5 files
# TODO: deduce data format from the data argument (for saving with a default key)
# TODO: change test_tar.test_serial_tar, use ['arr0'] instead (omit ext)
# TODO: test parallel save/load
# TODO: clean up redundant code/ make code more readable


def _save(arr, tar, key, **kwargs):
    from .base import save as bsave
    from .formats import formats

    f = BytesIO()
    bsave(arr, f, format=formats.get_format(filename=basename(key)))
    size = f.tell()  # get the size of the file object to write in the tarball
    f.seek(0)
    tarinfo = tarfile.TarInfo(name=key)
    tarinfo.size = size
    tar.addfile(tarinfo, f)
    tar.close()


def save(arr, filename, key=None, comm=None, **kwargs):
    from lyncs_utils.io import IOBase
    from sys import getsizeof

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
            if name + '.npy' not in [m.name for m in tar.getmembers()]:
                key = name + '.npy'
                break

    _save(arr, tar, key, **kwargs)


def _load_member(tar, member, header_only=False, as_data=False, comm=None, **kwargs):
    from .base import head as bhead
    from .base import load as bload
    from .archive import Data
    from .formats import formats
    from .header import Header

    f = BytesIO()
    f.write(tar.extractfile(member).read())
    f.seek(0)
    header = Header(bhead(f, format=formats.get_format(
                        filename=basename(member.name))))

    if header_only:
        return header

    f.seek(0)
    data = bload(f, format=formats.get_format(filename=basename(member.name)))
    return Data(header, data) if as_data else data


def _load(paths, tar, **kwargs):
    new_path_dict = nested_dict()
    for path in paths:
        parts = path.split('/')
        if parts:
            marcher = new_path_dict
            for key in parts[:-1]:
                marcher = marcher[key]
            header = _load_member(tar, find_member(tar, tar.getmember(path).name),
                                  as_data=True, **kwargs)
            marcher[parts[-1]] = header
    return default_to_regular(new_path_dict)


def _load_dispatch(tar, key, loader, depth=1, all_data=False, **kwargs):
    from .archive import Archive

    # if .h5
    if key and key.split('/')[0].endswith('.h5'):
        raise NotImplementedError(
            f"HDF5 file {key} is not supported when inside a tarball")

    if is_dir(tar, format_key(key)):
        key = format_key(key)
        paths = [member.name for member in tar.getmembers()
                if ((member.name.startswith(key) or key == '/')
                and (get_depth(member.name, key) <= depth or all_data))]

        # avoid {dir : {data}}. Return {data} instead if key is given.
        _dict = _load(paths, tar, **kwargs)[key[:-1]]\
                    if key != '/'\
                    else _load(paths, tar, **kwargs)
        return Archive(_dict, loader=loader, path=key)

    return _load_member(tar, find_member(tar, key), **kwargs)


def load(filename, key=None, chunks=None, comm=None, **kwargs):
    from .archive import Loader

    filename, key = split_filename(filename, key)
    mode_suffix = _get_mode(filename)
    kwargs = {"comm": comm, **kwargs}
    loader = Loader(load, filename, kwargs=kwargs)

    with tarfile.open(filename, "r" + mode_suffix) as tar:
        return _load_dispatch(tar, key, loader, **kwargs)


def head(*args, **kwargs):
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
