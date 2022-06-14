## I/O functions for Python and LQCD file formats

[![python](https://img.shields.io/pypi/pyversions/lyncs_io.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_io/)
[![pypi](https://img.shields.io/pypi/v/lyncs_io.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_io/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.io?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.io/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.io/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.io/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.io?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.io)
[![pylint](https://img.shields.io/badge/pylint%20score-8.9%2F10-yellowgreen?logo=python&logoColor=white)](http://pylint.pycqa.org/)
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
  will be searched inside the file (this is inspired by `h5py`).

- **Support for Parallel IO**. Where possible and implemented,
  parallel IO is supported. This is enabled either via MPI providing
  a valid communicator with the option `comm`, or via [Dask](https://dask.org)
  providing the option `chunks` (see
  [Dask's Array](https://docs.dask.org/en/latest/array-api.html)).

- **Omission of extension**. When saving, if the extension is omitted,
  the optimal file format is deduced from the data type and the extension
  is added to the filename. When loading, any extension is considered,
  i.e. `filename.*`, and if only one match is available, the file is loaded.

## Installation

The package can be installed via `pip`:

```bash
pip install [--user] lyncs_io
```

**NOTE**: for enabling parallel IO, lyncs_io requires a working MPI installation.
This can be installed via `apt-get`:

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

OR using `conda`:

```bash
conda install -c anaconda mpi4py
```

Parallel IO can then be enabled via

```bash
pip install [--user] lyncs_io[mpi]
```

## Documentation

The package provides three high-level functions:
- `head`: loads the metadata of a file (e.g. `shape`, `dtype`, etc)
- `load`: loads the content of a file
- `save` or `dump`: saves data to file

```python
import numpy as np
import lyncs_io as io

arr1 = np.random.rand(10,10,10)
io.save(arr1, "data.npy")

arr2 = io.load("data.npy")

assert (arr1 == arr2).all()
```

NOTE: for `save` we use the order `data, filename`. This is the opposite
of what done in `numpy` but consistent with `pickle`'s `dump`. This order
is preferred because the function can be used directly as a method
for a class since `self`, i.e. the `data`, would be passed as the first
argument of `save`.

### Supported file formats

Format  | Extensions | Binary | Archive | Parallel MPI | Parallel Dask
--------|------------|--------|---------|--------------|--------------
pickle  | pkl        | yes    | no      | no           | no
dill    | dll        | yes    | no      | no           | no
JSON    | json       | no     | no      | no           | no
ASCII   | txt        | no     | no      | no           | no
Numpy   | npy        | yes    | no      | yes          | yes
Numpyz  | npz        | yes    | yes     | TODO         | TODO
HDF5    | hdf5,h5    | yes    | yes     | yes          | TODO
lime    | lime       | yes    | TODO    | yes          | yes
Tar     | tar, tar.* |    -   | yes     | yes          | no
openqcd | oqcd       | yes    | no      | TODO         | TODO

### IO with HDF5

```python
import numpy as np
import lyncs_io as io

arr1 = np.random.rand(10,10,10)
io.save(arr1, "data.h5/random")

arr2 = np.zeros_like(arr1)
io.save(arr2, "data.h5/zeros")

arrs = io.load("data.h5")
assert (arr1 == arrs["random"]).all()
assert (arr2 == arrs["zeros"]).all()
```

Also the content of nested dictionary can be stored with HDF5:

```python
import numpy as np
import lyncs_io as io

mydict = {
        "random": {
            "arr0": np.random.rand(10,10,10),
            "arr1": np.random.rand(5,5),
        },
        "zeros":  np.zeros((4, 4, 4, 4)),
    }
# once a dictionary (or mapping) is passed it is written as a group
io.save(mydict, "data.h5")

# all the datasets in the .h5 file are loaded here using all_data argument
loaded_dict = io.load("data.h5", all_data=True)

assert (mydict["random"]["arr0"] == loaded_dict["random"]["arr0"]).all()
assert (mydict["random"]["arr1"] == loaded_dict["random"]["arr1"]).all()
assert (mydict["zeros"] == loaded_dict["zeros"]).all()
```

Parallel IO via MPI can be enabled with a parallel installation of HDF5.
For doing so, please refer to https://docs.h5py.org/en/stable/mpi.html.

### IO with MPI

```python
import numpy as np
import lyncs_io as io
from mpi4py import MPI

# Assume 2D cartesian topology
comm = MPI.COMM_WORLD
dims = (2,2) # e.g. 4 procs
cartesian2d = comm.Create_cart(dims=dims)

oarr = np.random.rand(6, 4, 2, 2)
io.save(oarr, "pario.npy", comm=cartesian2d)
iarr = io.load("pario.npy", comm=cartesian2d)

assert (iarr == oarr).all()
```

NOTE: Parallel IO is enabled once a valid cartesian communicator is passed to `load` or `save` routines, otherwise Serial IO is performed. Currently only `numpy` format supports this functionality.

### IO with Dask

```python
import lyncs_io as io
import dask
from distributed import Client, progress

client = Client(n_workers=2, threads_per_worker=1)

x = da.arange(0,128).reshape((16, 8)).rechunk(chunks=(8,4))

xout_lazy = io.save(x, "pario.npy")
xin_lazy = io.load("pario.npy", chunks=(8,4))

assert (x.compute() == xin_lazy.compute()).all()
client.shutdown()
```

NOTE: Parallel IO with Dask is enabled once a valid chunk size is passed to `load` routine using `chunks` parameter. For `save` routine, the DaskIO is enabled only if the array passed is a Dask Array. Currently only `numpy` format supports this functionality.

### IO with Tar

```python
import numpy as np
import lyncs_io as io

arr1 = np.random.rand(10,10,10)
io.save(arr1, "data.tar/random.npy")

arr2 = np.zeros_like(arr1)
io.save(arr2, "data.tar/zeros.npy")

arrs = io.load("data.tar")

assert (arr1 == arrs["random.npy"]).all()
assert (arr2 == arrs["zeros.npy"]).all()
```

Also the content of nested dictionary can be stored with Tar:

```python
mydict = {
  "random": {
		"arr0.npy": np.random.rand(10,10,10),
		"arr1.npy": np.random.rand(5,5),
	},
	"zeros.npy": np.zeros((4, 4, 4, 4)),
}

io.save(mydict, 'data.npy')

loaded_dict = io.load('data.npy', all_data=True)

assert (mydict["random"]["arr0.npy"] == loaded_dict["random"]["arr0.npy"]).all()
assert (mydict["random"]["arr1.npy"] == loaded_dict["random"]["arr1.npy"]).all()
assert (mydict["zeros.npy"] == loaded_dict["zeros.npy"]).all()
```
#### Note:
- Some formats inside a Tar are not currently supported. (See [Issues](https://github.com/Lyncs-API/lyncs.io/issues))
- When loading/saving a file in series, it's done directly on the memory. When in parallel, files are first written on the secondary storage before being saved/loaded.

### Adding a file format

New file formats can be registered providing providing where possible the respective `head`, `load` and `save` functions.
For example the `pickle` file format can be registered as follow:

```python
import pickle
from lyncs_io.formats import register

register(
    "pickle",                         # Name of the format
    extensions=["pkl"],               # List of extensions
    head=None,                        # Function called by head (omitted)
    load=pickle.load,                 # Function called by load
    save=pickle.dump,                 # Function called by save
    description="Pickle file format", # Short description
)
```

## Acknowledgments

### Authors
- Simone Bacchio (sbacchio)
- Christodoulos Stylianou (cstyl)
- Alexandros Angeli (alexandrosangeli)

### Fundings
- PRACE-6IP, Grant agreement ID: 823767, Project name: LyNcs.
