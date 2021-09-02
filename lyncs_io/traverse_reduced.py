import numpy
import torch
from collections import OrderedDict
from typing import Iterable
from torch.nn import Conv1d
from pprint import pprint


class Dummy(list):
    pass


def from_dummy(ds):
    for i, elt in enumerate(ds):
        if isinstance(elt, Dummy):
            ds[i] = from_dummy(elt)
    if isinstance(ds, Dummy):
        ds = tuple(ds)
    return ds


def to_dummy(ds):
    if isinstance(ds, tuple):
        ds = Dummy(ds)
    for i, elt in enumerate(ds):
        if isinstance(elt, tuple):
            ds[i] = to_dummy(elt)
    return ds


def gen():
    num = 0
    while True:
        yield num
        num += 1


gen = gen()
global_dict = {}


def fnc(s, reverse=False, gen=gen):
    global global_dict

    if isinstance(s, (dict, OrderedDict)):
        x = {key: fnc(value, reverse) for key, value in s.items()}
        return OrderedDict(x) if isinstance(s, OrderedDict) else x
    elif isinstance(s, (list, tuple)):
        x = [fnc(e, reverse) for e in s]
        return tuple(x) if isinstance(s, tuple) else x
    elif isinstance(s, torch.nn.Parameter):
        placeholder_no = str(next(gen))
        global_dict["placeholder" + placeholder_no] = s
        return "placeholder" + str(placeholder_no)
    elif isinstance(s, str) and reverse and s in global_dict.keys():
        return global_dict[s]
    return s


c = Conv1d(4, 4, 3)
reduced = c.__reduce__()

result = fnc(reduced)
after = fnc(result, reverse=True)

test = reduced == after and reduced != result
print(test)
