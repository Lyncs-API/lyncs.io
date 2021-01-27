## I/O functions for common Python and LQCD file formats

Lyncs IO offers two high-level functions `load` and `save/dump`.

The main features/objectives of this module are

- **Seamlessly IO**, reading and writing made simple.
  In most of the cases, after saving `save(obj, filename)`,
  loading `obj=load(filename)` returns the original Python object. 

- **Many formats supported**. The file format can be specified either
  via the filename's extension or with the option `format` passed to
  `load/save`.

- **Support for encapsulation and archives**. When formats support
  encapsuplation, e.g. HDF5, or are archives, e.g. tar, parts of the
  content can be accessed directly by specifying it in the path.
  For instance with `directory/file.h5/content`, `directory/file.h5`
  is the file path, and the remaining is content to be accessed that
  will be searched inside the file.

- **Support for Parallel IO**. Where possible, the option `chunks`
  can be used for enabling parallel IO via `Dask`.

- **Omission of extension**. When saving, the optimal file format is
  deduced from the data type and the extension is added to the filename.
  When loading, any extension is considered, `filename.*`, and if only
  one match is available, that file is loaded.

## Supported file formats

## Supported objects