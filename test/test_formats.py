from pytest import raises
from lyncs_io.formats import *


def test_register():
    f0 = formats[0]
    with raises(ValueError):
        register(f0.name, f0.modules, f0.extensions)
