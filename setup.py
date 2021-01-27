from lyncs_setuptools import setup

setup(
    "lyncs_io",
    install_requires=[
        "lyncs_utils",
        "numpy",
    ],
    extras_require={"test": ["pytest", "pytest-cov", "ipython"]},
)
