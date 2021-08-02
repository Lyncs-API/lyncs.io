"""
Interface for Tar format
"""

from os.path import exists, splitext, basename
from .archive import split_filename
from io import BytesIO
from .utils import nested_dict, default_to_regular
import tarfile
import tempfile


all_extensions = [
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


def save(arr, filename):
    from .base import save as b_save
    from .formats import formats
    from lyncs_utils.io import IOBase
    from sys import getsizeof

    tarball_path, leaf = split_filename(filename)
    mode_suffix = _get_mode(tarball_path)

    # create Tar if doesn't exist - append if it does
    if exists(tarball_path):
        tar = tarfile.open(tarball_path, "a")
    else:
        tar = tarfile.open(tarball_path, "w" + mode_suffix)

    f = BytesIO()

    b_save(arr, f, format=formats.get_format(filename=basename(leaf)))
    size = f.tell()  # get the size of the file object to write in the tarball
    f.seek(0)
    tarinfo = tarfile.TarInfo(name=leaf)
    tarinfo.size = size
    tar.addfile(tarinfo, f)
    tar.close()

    return f


def _format_key(key):
    if key:
        return key if key[-1] == '/' else key + '/'
    return '/'


def _get_depth(path, key):
    key_depth = sum([1 for char in key if char == '/'])
    path_depth = sum([1 for char in path if char == '/'])
    diff = path_depth - key_depth
    return diff + 1 if key != '/' else diff + 2



def _is_dir(tar, key):
    for member in tar.getmembers():
        if key == '/' or member.name.startswith(key):
            return True
    return False


def _load_member(tar, member, header_only=False, as_data=False):
    from .base import head as bhead
    from .base import load as bload
    from .archive import Data
    from .formats import formats

    f = BytesIO()
    f.name = 'foo.h5'
    f.write(tar.extractfile(member).read())
    f.seek(0)
    header = bhead(f, format=formats.get_format(filename=basename(member.name)))

    if header_only:
        return Data(header, None) if as_data else header

    f.seek(0)
    data = bload(f, format=formats.get_format(filename=basename(member.name)))
    return Data(header, data) if as_data else data


def _load(paths, tar):
    new_path_dict = nested_dict()
    for path in paths:
        parts = path.split('/')
        if parts:
            marcher = new_path_dict
            for key in parts[:-1]:
                marcher = marcher[key]
            header = _load_member(tar, tar.getmember(path), header_only=True, as_data=True)
            marcher[parts[-1]] = header
    return default_to_regular(new_path_dict)


def _load_dispatch(tar, key, loader, header_only, depth=1, all_data=False, ** kwargs):
    from .base import head as bhead
    from .base import load as bload
    from .formats import formats
    from .archive import Archive

    # if .h5
    if key and key.split('/')[0].endswith('.h5'):
        raise NotImplementedError(f"HDF5 file {key} is not supported when inside a tarball")

    if _is_dir(tar, _format_key(key)):
        key = _format_key(key)
        paths = [member.name for member in tar.getmembers()
                if ((member.name.startswith(key) or key=='/')
                and (_get_depth(member.name, key) <= depth or all_data))]
        # avoid {dir : {data}}. Return {data} instead if key is given.
        _dict = _load(paths, tar)[key[:-1]] if key != '/' else _load(paths, tar)
        return Archive(_dict, loader=loader, path=key)

    else:
        member = tar.getmember(key)
        return _load_member(tar, member)


def load(filename, key=None, header_only=False, **kwargs):
    from .archive import Loader

    filename, key = split_filename(filename, key)
    mode_suffix = _get_mode(filename)
    loader = Loader(load, filename, kwargs=kwargs)

    with tarfile.open(filename, "r" + mode_suffix) as tar:
        return _load_dispatch(tar, key, loader, header_only, **kwargs)


def head(filename):
    return load(filename, header_only=True)


def _get_mode(filename):
    """
    Returns the mode (from modes) with which the tarball should be read/written as
    """
    ext = splitext(filename)[-1]
    for key, val in modes.items():
        if ext in val:
            return key
    raise ValueError(f"{ext} is not supported.")
