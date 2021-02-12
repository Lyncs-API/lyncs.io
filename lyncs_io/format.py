"""
Base class for file formats
"""

from dataclasses import dataclass
from importlib import import_module
from collections import OrderedDict
from .archive import to_path, split_filename


@dataclass
class Format:
    """
    Format class

    Attributes
    ----------
    - name: name of the format
    - modules: list of modules that implement the format
    - extensions: list of extensions used by the format
    - archive: whether the format is used for archiving
    - binary: whether the format stores the data as binary
    - description: a description of the format
    """

    name: str
    modules: dict
    extensions: list
    archive: bool = False
    binary: bool = True
    description: str = ""

    def load(self, filename, module=None, **kwargs):
        "Calls the load function of the format. The module used can be changed."
        fnc = self.get_func(module, "load")
        assert callable(fnc), "Got a non-callable function"

        try:
            return fnc(filename, **kwargs)
        except TypeError:
            return fnc(self.ropen(filename), **kwargs)

    def save(self, data, filename, module=None, **kwargs):
        "Calls the save function of the format. The module used can be changed."
        fnc = self.get_func(module, "save", "dump")
        assert callable(fnc), "Got a non-callable function"

        try:
            return fnc(data, filename, **kwargs)
        except TypeError:
            return fnc(data, self.wopen(filename), **kwargs)

    def ropen(self, filename):
        "Opens the file for reading"
        return open(filename, "r" + "b" if self.binary else "")

    def wopen(self, filename):
        "Opens the file for writing"
        return open(filename, "a" + "b" if self.binary else "")

    def get_module(self, module):
        "Translates the given module (e.g. str) to an effective module"
        if module is None:
            # Returning the first working module
            for mod in self.modules:
                try:
                    return self.get_module(mod)
                except ImportError:
                    continue
                raise ImportError(f"No module available for {self.name}")

        if module in self.modules:
            module = self.modules[module]

        if isinstance(module, str):
            return import_module(module, "lyncs_io")

        return module

    def get_func(self, module, *names):
        """
        Gets the required function from the module.
        Multiple aliases of the function name can be given.
        """
        module = self.get_module(module)

        if callable(module):
            return module

        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
            if name in module:
                return module[name]

        raise ValueError(f"Function(s) {names} not found in {module}")

    def __eq__(self, other):
        if isinstance(other, Format):
            return super().__eq__(other)
        if isinstance(other, str):
            return self.name == other
        return False

    def __str__(self):
        return str(self.name)

    def check_filename(self, filename):
        """
        Bool, check if the filename extension is appropriate for the format.
        """
        assert isinstance(filename, str)

        ext = filename.split("/")[-1].split(".")[-1]
        if ext in self.extensions:
            return True

        return False


class Formats(OrderedDict):
    "Collection of formats"

    def from_format(self, format):
        "Returns a format from the given format"
        if isinstance(format, Format):
            return format
        if not isinstance(format, str):
            raise TypeError("Format should be a string.")

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
            return self.from_format(format)

        # 2. Using filename
        if filename:
            return self.from_path(filename)

        raise ValueError(
            """
            Format could not be deduce from the filename. Please specify a format.
            Available formats are {self}.
            """
        )

    def __str__(self):
        return ", ".join(self.keys())

    def doc(self):
        "Returns a documentation describing the available formats"
        # TODO
        return str(self)
