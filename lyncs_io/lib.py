"""
Includes functions defined in include using lyncs_cppyy
"""

__all__ = ["lib"]

import os
from lyncs_utils import RaiseOnUse

headers = [
    "openqcd.h",
]

try:
    from lyncs_cppyy import Lib

    lib = Lib(
        path=os.path.dirname(os.path.abspath(__file__)),
        header=headers,
        namespace="lyncs_io",
    )

except ImportError as _err:
    error = RuntimeError("lyncs_cppyy not available. Please install lyncs_cppyy")
    error.__cause__ = _err
    lib = RaiseOnUse(error)
