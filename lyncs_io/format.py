"""
Base class for file formats
"""

from dataclasses import dataclass
from collections import OrderedDict
from lyncs_utils import to_path
from .archive import split_filename


def not_implemented(*args, **kwargs):
    "Raises not implemented error"
    raise NotImplementedError("This function is not avaiable")


@dataclass
class Format:
    """
    Format class

    Attributes
    ----------
    - name: name of the format
    - alias: alternative names of the format
    - load: function for loading
    - save: function for saving
    - extensions: list of extensions used by the format
    - archive: whether the format is used for archiving
    - binary: whether the format stores the data as binary
    - description: a description of the format
    """

    name: str
    alias: list
    extensions: list
    load: callable = not_implemented
    save: callable = not_implemented
    head: callable = not_implemented
    error: Exception = None
    archive: bool = False
    description: str = ""

    def __eq__(self, other):
        if isinstance(other, Format):
            return super().__eq__(other)
        if self.name == other:
            return True
        try:
            return other in self.alias
        except TypeError:
            pass
        return False

    def __str__(self):
        return str(self.name)

    def check(self):
        "Checks the format and return itself. Otherwise raises error."
        if self.error:
            raise self.error
        return self

    @property
    def names(self):
        "Names of the format (name and alias)"
        return (self.name,) + tuple(self.alias)


class Formats(OrderedDict):
    "Collection of formats"

    def from_format(self, format):
        "Returns a format from the given format"
        if isinstance(format, Format):
            return format
        if not isinstance(format, str):
            raise TypeError("Format should be a string.")

        format = format.lower()
        if format in self:
            return self[format]

        raise ValueError(f"Unknown format {format}")

    def from_suffix(self, *suffixes):
        "Returns a format from the given suffix"
        suffix = ""
        for part in reversed(suffixes):
            suffix += part
            for format in self.values():
                if suffix[1:] in format.extensions:
                    return format
        raise ValueError(f"Could not deduce the format from the suffix: {suffix}")

    def from_path(self, path):
        "Returns a format from the given path"
        path = to_path(path)
        err = "Could not deduce the format from the path"

        if path.parent.is_dir():
            if path.suffix:
                return self.from_suffix(*path.suffixes)
            if path.exists():
                raise ValueError(err)
            matches = tuple(path.parent.glob(path.name + ".*"))
            if not matches or len(matches) > 1:
                raise ValueError(err)
            return self.from_path(matches[0])

        path, _ = split_filename(path)
        return self.from_path(path)

    def get_format(self, format=None, filename=None):
        "Return the appropriate format checking the format string or the filename extension."

        # 1. Using format
        if format:
            return self.from_format(format).check()

        # 2. Using filename
        if filename:
            return self.from_path(filename).check()

        raise ValueError(
            """
            Format could not be deduce from the filename. Please specify a format.
            Available formats are {self}.
            """
        )

    def register(self, *names, **kwargs):
        "Adds a format to the list of formats"
        if not names:
            raise ValueError("Name not given")
        fmt = Format(names[0], alias=names[1:], **kwargs)
        for name in names:
            self[name.lower()] = fmt

    def __str__(self):
        return ", ".join(self.keys())

    def doc(self):
        "Returns a documentation describing the available formats"
        # TODO
        return str(self)
