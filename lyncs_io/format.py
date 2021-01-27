"""
Base class for file formats
"""

from dataclasses import dataclass
from importlib import import_module


@dataclass
class Format:
    name: str
    modules: dict
    extensions: list
    encapsulated: bool = False
    binary: bool = True
    description: str = ""

    def load(self, filename, module=None, **kwargs):
        fnc = self.get_func(module, "load")
        assert callable(fnc), "Got a non-callable function"

        try:
            return fnc(filename, **kwargs)
        except TypeError:
            return fnc(self.ropen(filename), **kwargs)

    def save(self, data, filename, module=None, **kwargs):
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
            return import_module(self)

        return module

    def get_func(self, module, *names):
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
        assert isinstance(filename, str)

        ext = filename.split("/")[-1].split(".")[-1]
        if ext in self.extensions:
            return True

        return False
