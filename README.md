## I/O functions for Python and LQCD file formats

[![python](https://img.shields.io/pypi/pyversions/lyncs_io.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_io/)
[![pypi](https://img.shields.io/pypi/v/lyncs_io.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_io/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.io?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.io/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.io/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.io/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.io?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.io)
[![pylint](https://img.shields.io/badge/pylint%20score-9.2%2F10-green?logo=python&logoColor=white)](http://pylint.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)

Lyncs IO offers two high-level functions `load` and `save` (or `dump` as alias of `save`).

The main features of this module are

- **Seamlessly IO**, reading and writing made simple.
  In most of the cases, after saving `save(obj, filename)`,
  loading `obj=load(filename)` returns the original Python object.
  This feature is already ensured by formats like `pickle`, but we
  try to ensure it as much as possible also for other formats.

- **Many formats supported**. The file format can be specified either
  via the filename's extension or with the option `format` passed to
  `load/save`. The structure of the package is flexible enough to
  easily accomodate new/customized file formats as these arise.
  See [Adding a file format] for guidelines.

- **Support for archives**. In case of archives, e.g. HDF5, zip etc.,
  the content can be accessed directly by specifying it in the path.
  For instance with `directory/file.h5/content`, `directory/file.h5`
  is the file path, and the remaining is content to be accessed that
  will be searched inside the file.

- **Support for Parallel IO**. Where possible, the option `chunks`
  can be used for enabling parallel IO via `Dask`.

- **Omission of extension**. When saving, if the extension is omitted,
  the optimal file format is deduced from the data type and the extension
  is added to the filename. When loading, any extension is considered,
  i.e. `filename.*`, and if only one match is available, the file is loaded.

### Example

```python
import numpy as np
import lyncs_io as io

arr1 = np.random.rand((10,10,10))
io.save(arr, "data.h5/random")

arr2 = np.zeros_like(arr)
io.save(arr, "data.h5/zeros")

arrs = io.load("data.h5")
assert (arr1 == arrs["random"]).all()
assert (arr2 == arrs["zeros"]).all()
```

NOTE: for `save` we use the order `data, filename`. This is the opposite
of what done in `numpy` but consistent with `pickle`'s `dump`. This order
is preferred because the function can be used directly as a method
for a class since `self`, i.e. the `data`, would be passed as the first
argument of `save`.

### File formats

### Adding a file format