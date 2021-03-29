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
        "mpi": ["mpi4py"],
        "test": ["pytest", "pytest-cov", "ipython", "pytest-mpi"],
    },
)
