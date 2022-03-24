from numpy import dtype
from lyncs_io import load, head
from lyncs_io.lib import lib
from lyncs_io.testing import skip_openqcd


@skip_openqcd
def test_reference():

    arr = load("test/conf.oqcd")
    assert arr.shape == (16, 8, 8, 8, 4, 3, 3)
    assert arr.dtype == "<c16"

    attrs = head("test/conf.oqcd")
    assert attrs["shape"] == (16, 8, 8, 8, 4, 3, 3)
    assert dtype(attrs["dtype"]) == "<c16"
    assert "plaq" in attrs
