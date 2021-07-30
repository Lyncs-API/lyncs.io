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
    # f.name = 'temp.h5'

    b_save(arr, f, format=formats.get_format(filename=basename(leaf)))
    size = f.tell()  # get the size of the file object to write in the tarball
    f.seek(0)
    tarinfo = tarfile.TarInfo(name=leaf)
    tarinfo.size = size
    tar.addfile(tarinfo, f)

    tar.close()

    return f


def _is_dir(tar, key):
    for member in tar.getmembers():
        if not key or member.name.startswith(key + '/'):
            return True
    return False


def _get_content(tar, key):
    from .archive import Data
    from .base import head as bhead
    from .formats import formats

    cont = dict()
    for member in tar.getmembers():
        if not key or member.name.startswith(key + '/'):
            data_obj = _load_member(tar, member, header_only=False)
            cont[basename(member.name)] = data_obj
    return cont


def _load_member(tar, member, header_only=False):
    from .base import head as bhead
    from .base import load as bload
    from .archive import Data
    from .formats import formats

    f = BytesIO()
    f.write(tar.extractfile(member).read())
    f.seek(0)
    header = bhead(f, format=formats.get_format(filename=basename(member.name)))
    f.seek(0)

    if not header_only:
        data = bload(f, format=formats.get_format(filename=basename(member.name)))
        return Data(header, data)

    return Data(header, None)


def _load(paths, tar):
    new_path_dict = nested_dict()
    for path in paths:
        parts = path.split('/')
        if parts:
            marcher = new_path_dict
            for key in parts[:-1]:
                marcher = marcher[key]
            header = _load_member(tar, tar.getmember(path), header_only=True)
            marcher[parts[-1]] = header
    return default_to_regular(new_path_dict)


def _load_dispatch(tar, key, loader, header_only, **kwargs):
    from .base import head as bhead
    from .base import load as bload
    from .formats import formats
    from .archive import Archive

    if _is_dir(tar, key):
        paths = [member.name for member in tar.getmembers()]
        return Archive(_load(paths, tar), loader=loader, path=key)

    else:
        member = tar.getmember(key)
        if member.isfile():
            f = BytesIO()
            f.write(tar.extractfile(member).read())
            f.seek(0)
            if header_only:
                return bhead(f, format=formats.get_format(filename=basename(key)))

            return bload(f, format=formats.get_format(filename=basename(key)))


def load(filename, key=None, header_only=False, **kwargs):
    from .archive import Loader
    from .base import load as bload
    from .base import head as bhead
    from .formats import formats

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
