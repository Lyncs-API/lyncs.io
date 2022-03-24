from glob import glob
from lyncs_setuptools import setup

setup(
    "lyncs_io",
    install_requires=[
        "lyncs_utils>=0.3.5",
        "numpy",
        "dataclasses",
        "xmltodict",
        "filelock",
    ],
    extras_require={
        "tree": ["ipython"],
        "dill": ["dill"],
        "hdf5": ["h5py"],
        "mpi": ["mpi4py"],
        "dask": ["dask", "distributed"],
        "openqcd": ["lyncs_cppyy"],
        "test": ["pytest", "pytest-cov", "ipython", "pytest-mpi"],
    },
    data_files=[
        ("lyncs_io/include", glob("lyncs_io/include/*.h")),
    ],
    package_data={"lyncs_cppyy": ["include/*.h"]},
    include_package_data=True,
)
