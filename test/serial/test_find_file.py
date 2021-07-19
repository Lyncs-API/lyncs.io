import sys
import os
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lyncs_io.utils import find_file

class TestFunction(unittest.TestCase):
    def test_should_find_file_with_omitted_extension_and_only_one_match(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
            with open(tempdir + "/data.npy", "w") as data:
                self.assertEqual(find_file(tempdir + '/data'), os.path.abspath("data.npy"))

    def test_should_raise_exception_when_more_than_one_match_is_found(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
            with open(tempdir + "/data.npy", "w") as data_npy:
                with open(tempdir + "/data.h5", "w") as data_h5:
                    self.assertRaises(Exception, find_file, tempdir + '/data')

    def test_should_raise_exception_when_no_match_is_found(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as tempdir:
            with open(tempdir + "/data.npy", "w") as data:
                self.assertRaises(Exception, find_file, tempdir + '/d_data')


if __name__ == "__main__":
    unittest.main()
