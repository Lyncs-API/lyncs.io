from pytest import raises
from lyncs_io.format import Formats


def test_error():
    formats = Formats()
    formats.register(
        "broken", extensions=["bar"], error=ImportError("Not a real format")
    )

    with raises(ImportError):
        formats.get_format(format="broken")

    with raises(ImportError):
        formats.get_format(filename="foo.bar")
