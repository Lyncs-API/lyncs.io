"""
Archive class and utils
"""

from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass
from lyncs_utils import to_path
from .header import Header


@dataclass
class Loader:
    """
    Loader class

    Attributes
    ----------
    - load: function to be called for loading.
    - filename: name of the file to load from.
    - kwargs: list of options to pass to the load function.
    """

    load: callable
    filename: str
    kwargs: dict

    def __call__(self, key=None, **kwargs):
        "Load the given key from the file"
        return self.load(self.filename, key=key, **self.kwargs, **kwargs)


@dataclass
class Data:
    """
    Data class

    Attributes
    ----------
    - header: a dictionary that describes the content
    - value: the value of the data (None if not loaded)
    """

    header: Header
    value: Any = None

    def __repr__(self):
        header = ", ".join(
            (f"{key}={val}" for key, val in self.header.items() if key[0] != "_")
        )
        return f"Data({header})"


@dataclass
class Archive(Mapping):
    """
    Archive class (mapping)

    Attributes
    ----------
    - _dict: backend map, usually a dictionary
    - loader: instance of the Loader class
    - path: current path inside the file
    - parent: parent archive
    """

    _dict: dict
    loader: Loader = None
    path: str = None
    parent: Any = None

    def __post_init__(self):
        if not self.path:
            self.path = ""

    def load(self, key, **kwargs):
        "Loads key from the file"
        if not self.loader:
            raise ValueError("Loader not given")
        return self.loader(f"{self.path}/{key}", **kwargs)

    def data(self):
        """
        Returns keys and values of entries that are data objects.
        Note: commonly the data has not be loaded yet. For loading it
              you need to access the key in the archive.
        """
        return {
            key: self._dict[key]
            for key in self
            if isinstance(self._dict[key], (Data, Header))
        }

    @property
    def key(self):
        "Returns the current key of the archive"
        return self.path.split("/")[-1]

    def tree(self):
        "Tree representation of the archive"
        try:
            # pylint: disable=import-outside-toplevel
            from IPython.lib.pretty import pretty

            return pretty(self)
        except:
            raise ImportError("tree requires IPython")

    def _repr_pretty_(self, printer=None, cycle=False):
        if cycle:
            keys = self.keys()
            keys = repr(next(keys)) + ", ..." if next(keys) else ""
            printer.text(f"Archive({keys})")
            return

        ind = 4
        for key in self:
            if len(self) > 1:
                printer.break_()
            else:
                printer.breakable("")
            val = self._dict[key]

            if isinstance(val, (Data, Header)):
                printer.text(f"{key} -> {val}")
                continue
            if val is None:
                printer.text(f"{key}/...")
                continue
            if not isinstance(val, Mapping):
                raise TypeError(f"Unexpected type {type(val)} for key {key}")
            printer.begin_group(ind, key + "/")
            self[key]._repr_pretty_(printer)
            printer.end_group(ind)

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, key):
        return key in self._dict

    @classmethod
    def _split_key(cls, key):
        if not key:
            return key, ()

        if isinstance(key, tuple):
            this, others = cls._split_key(key[0])
            return this, others + key[1:]

        if isinstance(key, str):
            key = key.lstrip("/").split("/")
            return key[0], tuple(key[1:])

        return key, ()

    def __getitem__(self, key):
        this, others = self._split_key(key)
        if not isinstance(this, str):
            raise TypeError(f"Expected a str for key not {type(this)}")

        val = None
        if not this or this == ".":
            val = self

        elif this == "..":
            if not self.parent:
                raise KeyError("'..' no parent available")
            val = self.parent

        elif this in self:
            val = self._dict[this]
            if val is None:
                self._dict[this] = self.load(this)
                val = self._dict[this]

            if isinstance(val, Data):
                if val.value is None:
                    val.value = self.load(this)
                val = val.value
            elif isinstance(val, Header):
                pass
            else:
                val = Archive(
                    val,
                    loader=self.loader,
                    path=f"{self.path}/{this}",
                    parent=self,
                )

        else:
            raise KeyError(f"Unexpected key: {this}")

        if others:
            return val[others]
        return val

    def __repr__(self):
        return repr(self._dict)


def split_filename(filename, key=None):
    "Splits the actual filename from the content to be accessed."
    key = key or ""
    if not isinstance(key, str):
        raise TypeError("Key is not a string")

    try:
        path = to_path(filename)
    except TypeError:
        return filename, key

    while not path.parent.is_dir():
        if key:
            key = f"{path.name}/{key}"
        else:
            key = path.name
        path = path.parent

    return str(path), key.lstrip("/")
