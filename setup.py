from lyncs_setuptools import setup

setup(
    "lyncs_io",
    install_requires=[
        "lyncs_utils",
        "numpy",
    ],
    extras_require={
        "dill": ["dill"],
        "hdf5": ["h5py"],
        "numpy": ["numpy"],
        "test": ["pytest", "pytest-cov", "ipython"],
    },
)
