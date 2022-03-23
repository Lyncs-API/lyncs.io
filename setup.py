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
)
