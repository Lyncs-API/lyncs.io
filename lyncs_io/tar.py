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
def save(arr, filename):
    """
    Save data from an array to a file in a tarball.

    Params:
    -------
    arr: The array to save the data from.
    tarball_name: The name of the tarball.
    filename: The name of the file to save the data to.
    """
    pre, tarball_name, leaf = split_filename(filename)
    mode_suffix = get_mode(get_extension(tarball_name))

    # create Tar if doesn't exist - append if it does
    if exists(pre + tarball_name):
        tar = tarfile.open(pre + tarball_name, "a")
    else:
        tar = tarfile.open(pre + tarball_name, "w" + mode_suffix)

    # A temporary directory is created to write a new file in.
    # The new file is then added to the tarball before being deleted.
    with tempfile.TemporaryDirectory() as tmpf:
        dat = open(tmpf + "/data.txt", 'w')
        for elt in arr:
            dat.write(str(elt) + '\n')
        dat.flush()  # flush data before adding the file to the archive
        tar.add(dat.name, arcname=leaf)
        dat.close()

    tar.close()



# def load(tarball_name, filename):
def load(filename):
    """
    Return an array from the data of a file in a tarball.
    """

    pre, tarball_name, leaf = split_filename(filename)
    mode_suffix = get_mode(get_extension(tarball_name))

    arr = [] # store data in

    if exists(pre + tarball_name):
        tar = tarfile.open(pre + tarball_name, "r" + mode_suffix)
    else:
        raise FileNotFoundError(f"{tarball_name} does not exist.")

    with tempfile.TemporaryDirectory() as tmpf:
        member = tar.getmember(leaf)
        tar.extract(member, path=tmpf) # extract to a temp location for reading
        data_file = f"{tmpf}/{leaf}"
        with open(data_file, "r") as dat:
            for line in dat.readlines():
                line = line.replace('\n','') # remove the newline character
                arr.append(line)
    tar.close()
    return arr


def get_tarball_from_path(path):
    """
    Return the name of the first tarball from a path and its parent directory.
    e.g. /home/user/tarball.tar/data/arr.npy --> /home/user/, tarball.tar/data/arr.npy
    """
    
    path = path.split('/')
    
    for i, f in enumerate(path):
        if '.tar' in f:
            return '/'.join(path[:i]) + '/' ,'/'.join(path[i:])
    raise ValueError("No tarball found in the path")


def get_mode(ext):
    """
    Returns the mode (from modes) with which the tarball should be read/written as
    """
    for key, val in modes.items():
        if ext in val:
            return key
    raise ValueError(f"{ext} is not supported.")


def get_extension(filename):
    """
    Returns the extension from the filename
    e.g. 
    """
    return splitext(filename)[1]


def split_filename(filename):
    """
    Splits the filename into 3 parts:
    - The parent directory of the tarball
    - The name of the tarball
    - The file to read/write in the tarball
    """
    pre, filename = get_tarball_from_path(filename)
    cont = filename.split('/')
    tarball_name = cont[0]
    leaf = '/'.join(cont[1:])
    return pre, tarball_name, leaf

# Demo:
# --------------
# save an array to a location and load the array to x
# 
# save([0,0,1,0,1,1,0], '/home/alexandros/tarball.tar/dir1/datafile.txt')
# x = load('/home/alexandros/tarball.tar/dir1/datafile.txt')
# print(x)