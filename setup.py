from lyncs_setuptools import setup

setup(
    "lyncs_io",
    install_requires=[
        "lyncs_utils",
        "numpy",
        "dataclasses",
    ],
    extras_require={
        "dill": ["dill"],
        "hdf5": ["h5py"],
        "test": ["pytest", "pytest-cov", "ipython"],
    },
)
