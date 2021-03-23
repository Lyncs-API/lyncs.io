from lyncs_io.archive import Data, Archive, split_filename


def test_split():
    for fname, key in [
        ("foo", "bar"),
        ("../foo", "bar"),
        ("/foo", "bar"),
    ]:
        assert split_filename(fname + "/" + key) == (fname, key)
        assert split_filename(fname, key=key) == (fname, key)
