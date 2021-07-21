"""
Interface for Tar format
"""

from os.path import exists, splitext
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
    ':gz': ['.gz', '.taz', '.tgz'],
    ':bz2': ['.bz2', '.tb2', '.tbz', '.tbz2', '.tz2'],
    ':xz': ['.xz', '.txz'],
    ':': ['.tar'],
}

# Given the tarball name, return the compression mode using modes


def get_mode(ext):
    for key, val in modes.items():
        if ext in val:
            return key
    raise ValueError(f"{ext} is not supported.")


def get_extension(path):
    return splitext(path)[1]


def save(arr, tarball_name, filename):
    """
    Save data from an array to a file in a tarball.

    Params:
    -------
    arr: The array to save the data from.
    tarball_name: The name of the tarball.
    filename: The name of the file to save the data to.
    """

    mode_suffix = get_mode(get_extension(tarball_name))

    # create Tar if doesn't exist - append if it does
    if exists(tarball_name):
        tar = tarfile.open(tarball_name, "a")
    else:
        tar = tarfile.open(tarball_name, "w" + mode_suffix)

    # A temporary directory is created to write a new file in.
    # The new file is then added to the tarball before being deleted.
    with tempfile.TemporaryDirectory() as tmpf:
        dat = open(tmpf + "/data.txt", 'w')
        for elt in arr:
            dat.write(str(elt) + '\n')
        dat.flush()  # flush data before adding the file to the archive
        tar.add(dat.name, arcname=filename)
        dat.close()
        print("Success!")

    tar.close()
