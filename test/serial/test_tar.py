from lyncs_io.tar import load, save
from lyncs_io.testing import tempdir
import pytest
import tempfile
import tarfile


def test_save_load(tempdir):
    arr_a = [0,3,4,65]
    save(arr_a, tempdir + '/tarball.tar.gz/data.txt')
    assert load(tempdir + '/tarball.tar.gz/data.txt') == [str(elt) for elt in arr_a]


def test_save(tempdir):
    arr_a = [1,2,3,0]
    
    save(arr_a, tempdir + '/temptar.tar/datafile.txt')
    with tarfile.open(tempdir + '/temptar.tar', "r:") as tar:
        member = tar.getmember('datafile.txt')
        assert tar.getmembers()[0] == member
        assert len(tar.getmembers()) == 1

        # extract for reading
        tar.extract(tar.getmember('datafile.txt'), path=tempdir)

        with open(tempdir + '/datafile.txt', "r") as dat:
            for i, line in enumerate(dat.readlines()):
                line = line.replace('\n', '')
                # assert that the lines from the file match the elements in the arr
                assert line == str(arr_a[i])


def test_load(tempdir):
    arr_a = ['a', 'b', 'z']
    with tarfile.open(tempdir + '/tarball.tar', "w:") as tar:
        with open(tempdir + "/datafile.txt", "w") as dat:
            for elt in arr_a:
                dat.write(str(elt) + '\n')
            dat.flush()
            tar.add(dat.name, arcname='datafile.txt')
        
    assert load(tempdir + '/tarball.tar/datafile.txt') == arr_a

