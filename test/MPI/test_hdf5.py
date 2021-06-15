from pathlib import Path
import lyncs_io as io
import numpy as np
import tempfile

from lyncs_io.testing import dtype_loop, shape_loop
from lyncs_utils import prod


# @dtype_loop
# @shape_loop
def test_numpy():
    raise ValueError