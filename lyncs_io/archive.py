"""
Archive class and utils
"""

from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from os import PathLike


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
    - attrs: a dictionary that describes the content
    - value: the value of the data (None if not loaded)
    """

    attrs: dict
    value: Any = None


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
    loader: Loader
    path: str
    parent: Any = None

    def load(self, key, **kwargs):
        "Loads key from the file"
        return self.loader(f"{self.path}/{key}", **kwargs)

    def data(self):
        """
        Returns keys and values of entries that are data objects.
        Note: commonly the data has not be loaded yet. For loading it
              you need to access the key in the archive.
        """
        return {
            key: self._dict[key] for key in self if isinstance(self._dict[key], Data)
        }

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
            if not val:
                self._dict[this] = self.load(this)
                val = self._dict[this]

            if isinstance(val, Data):
                if val.value is None:
                    val.value = self.load(this)
                val = val.value
            else:
                val = Archive(
                    val, loader=self.loader, path=f"{self.path}/{this}", parent=self
                )

        else:
            raise KeyError(f"Unexpected key: {this}")

        if others:
            return val[others]
        return val


def split_filename(filename, key=None):
    "Splits the actual filename from the content to be accessed."
    if key:
        return filename, key

    if not isinstance(filename, (PathLike, bytes, str)):
        return filename, key

    path = Path(filename)
    return filename, key


def default_names(i=0):
    "Infinite generator of default names ('arrN') for entries of an archive."
    yield f"arr{i}"
    yield from default_names(i + 1)
