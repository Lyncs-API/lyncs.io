from lyncs_io.tar import *
from lyncs_io.testing import tempdir
import pytest
import tempfile
import tarfile


def test_save_load():
    arr_a = [0,3,4,65]
    with tempfile.TemporaryDirectory() as tempd:
        save(arr_a, tempd + '/tarball.tar.gz/data.txt')
        assert load(tempd + '/tarball.tar.gz/data.txt') == [str(elt) for elt in arr_a]


def test_save():
    arr_a = [1,2,3,0]
    
    with tempfile.TemporaryDirectory() as tempd:
        save(arr_a, tempd + '/temptar.tar/datafile.txt')
        with tarfile.open(tempd + '/temptar.tar', "r:") as tar:
            member = tar.getmember('datafile.txt')
            assert tar.getmembers()[0] == member
            assert len(tar.getmembers()) == 1

            # extract for reading
            tar.extract(tar.getmember('datafile.txt'), path=tempd)

            with open(tempd + '/datafile.txt', "r") as dat:
                for i, line in enumerate(dat.readlines()):
                    line = line.replace('\n', '')
                    # assert that the lines from the file match the elements in the arr
                    assert line == str(arr_a[i])


def test_load():
    arr_a = ['a', 'b', 'z']
    with tempfile.TemporaryDirectory() as tempd:
        with tarfile.open(tempd + '/tarball.tar', "w:") as tar:
            with open(tempd + "/datafile.txt", "w") as dat:
                for elt in arr_a:
                    dat.write(str(elt) + '\n')
                dat.flush()
                tar.add(dat.name, arcname='datafile.txt')
            
        assert load(tempd + '/tarball.tar/datafile.txt') == arr_a


def test_split_parent_tarball():
    path_a = '/home/user/tarball.tar.gz/datafile'
    assert split_parent_tarball(path_a)[0] == '/home/user/'
    assert split_parent_tarball(path_a)[1] == 'tarball.tar.gz/datafile'

    path_b = 'dir/tarball.txz'
    assert split_parent_tarball(path_b)[0] == 'dir/'
    assert split_parent_tarball(path_b)[1] == 'tarball.txz'

    path_c = '/home/user/datafile'
    with pytest.raises(ValueError):
        split_parent_tarball(path_c)


def test_get_mode():
    assert get_mode('tarball.tar.gz') == ':gz'
    assert get_mode('tarball.tb2') == ':bz2'
    assert get_mode('tarball.tar.xz') == ':xz'
    assert get_mode('tarball.tar') == ':'
    
    with pytest.raises(ValueError):
        get_mode('tarball.txt')


def test_get_extension():
    assert get_extension('tarball.tar.gz') == '.gz'
    assert get_extension('') == ''


def test_split_filename():
    path_a = '/home/user/tarball.tar/dir/data'
    assert split_filename(path_a)[0] == '/home/user/'
    assert split_filename(path_a)[1] == 'tarball.tar'
    assert split_filename(path_a)[2] == 'dir/data'
    assert len(split_filename(path_a)) == 3

    path_b = 'tarball.tar.gz/data'
    assert split_filename(path_b)[0] == ''
    assert split_filename(path_b)[1] == 'tarball.tar.gz'
    assert split_filename(path_b)[2] == 'data'
    assert len(split_filename(path_b)) == 3

    path_c = 'tarball.tar.gz'
    assert split_filename(path_c)[0] == ''
    assert split_filename(path_c)[1] == 'tarball.tar.gz'
    assert split_filename(path_c)[2] == ''
    assert len(split_filename(path_c)) == 3

