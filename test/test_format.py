from lyncs_io.format import Formats
import pytest


@pytest.mark.mpi_skip()
def test_error():
    formats = Formats()
    formats.register(
        "broken", extensions=["bar"], error=ImportError("Not a real format")
    )

    with pytest.raises(ImportError):
        formats.get_format(format="broken")

    with pytest.raises(ImportError):
        formats.get_format(filename="foo.bar")
