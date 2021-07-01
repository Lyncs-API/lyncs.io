from lyncs_setuptools import setup

setup(
    "lyncs_io",
    install_requires=[
        "lyncs_utils",
        "numpy",
        "dataclasses",
    ],
    extras_require={
        "tree": ["ipython"],
        "dill": ["dill"],
        "hdf5": ["h5py"],
        "mpi": [
            'mpi4py==3.0.0; python_version < "3.8"',
            'mpi4py==3.0.3; python_version >= "3.8"',
        ],
        "dask": ["dask", "distributed", "filelock"],
        "test": ["pytest", "pytest-cov", "ipython", "pytest-mpi"],
    },
)
