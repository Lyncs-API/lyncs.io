"""
Interface for Tar format
"""

from os.path import exists, splitext, basename
from .archive import split_filename
from io import BytesIO
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
    tarinfo = tarfile.TarInfo(name=basename(leaf))
    tarinfo.size = size
    tar.addfile(tarinfo, f)

    tar.close()

    return f


def load(filename, header_only=False):

    from .base import load as b_load
    from .base import head as b_head
    from .formats import formats

    tarball_path, leaf = split_filename(filename)
    mode_suffix = _get_mode(tarball_path)

    with tarfile.open(tarball_path, "r" + mode_suffix) as tar:
        member = tar.getmember(leaf)
        f = BytesIO()
        f.write(tar.extractfile(member).read())
        f.seek(0)

        if header_only:
            return b_head(f, format=formats.get_format(filename=basename(leaf)))

        return b_load(f, format=formats.get_format(filename=basename(leaf)))


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
