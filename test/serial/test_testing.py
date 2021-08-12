"""
Tests for testing.py
"""

from lyncs_io.testing import generate_rand_arr, dtype_loop, shape_loop


@shape_loop
@dtype_loop
def test_generate_rand_arr(shape, dtype):
    arr = generate_rand_arr(shape, dtype)
    assert arr.dtype == dtype
    assert arr.shape == shape
