"Interface for lime files"

__all__ = [
    "head",
    "load",
    "save",
]

import pickle
import re
from io import SEEK_CUR, BytesIO
import numpy
import xmltodict
from lyncs_utils import (
    prod,
    open_file,
    read_struct,
    write_struct,
    file_size,
)
from .convert import from_array, to_array
from .header import Header
from .utils import is_dask_array
from .mpi_io import MpiIO, check_comm
from .dask_io import DaskIO

# Constants
HEADER_SIZE = 144
MAGIC_NUMBER = 1164413355
FILE_VERSION_NUMBER = 1
TYPE_LENGTH = 128


def parse_num(val):
    "Returns number where possible"
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def parse_ildg(info):
    "Parser for ildg metadata"
    info = xmltodict.parse(info)["ildgFormat"]
    info = {k: parse_num(v) for k, v in info.items()}
    info["dtype"] = ">c%d" % (info["precision"] / 4)
    info["shape"] = tuple(info[key] for key in ("lt", "lz", "ly", "lx"))
    if info["field"] == "su3gauge":
        info["shape"] += (4, 3, 3)
    return info


def write_ildg(info):
    "Writer for ildg metadata"
    # TODO
    raise ValueError()


def parse_xlf(info):
    "Parser for xlf metadata"
    if isinstance(info, bytes):
        info = info.decode()
    info = (line.split("=") for line in re.split("\n|,", info) if "=" in line)
    return {k.strip(): parse_num(v.strip()) for k, v in info}


def write_xlf(info):
    "Writer for xlf metadata"
    return b"\n".join((f"{key} = {val}".encode() for key, val in info.items()))


# lime_type for records with metadata and respective parser
parse_metadatas = {
    "ildg-format": parse_ildg,
    "xlf-info": parse_xlf,
    "lyncs-io-info": pickle.loads,
}

write_metadatas = {
    "ildg-format": write_ildg,
    "xlf-info": write_xlf,
    "lyncs-io-info": pickle.dumps,
}

# lime_type for records with data
datas = [
    "lyncs-io-data",
    "ildg-binary-data",
]


def is_lime_file(_fp):
    "Whether is a lime file"
    magic_number = read_struct(_fp, ">l")[0]
    return magic_number == MAGIC_NUMBER


@open_file
def read_record_header(_fp):
    "Reads the header of a record"
    # pylint: disable=possibly-unused-variable
    magic_number, version, msg_bits, nbytes, lime_type = read_struct(
        _fp, ">lHHQ%ds" % (TYPE_LENGTH,)
    )
    offset = _fp.tell()
    if magic_number != MAGIC_NUMBER:
        raise TypeError(f"Not a valid lime file. magic_number = {magic_number}")
    lime_type = lime_type.decode().split("\0")[0]
    begin = bool(msg_bits >> 15 & 0b1)
    end = bool(msg_bits >> 14 & 0b1)
    return locals()


@open_file
def read_record(_fp, maxsize=1000):
    "Reads a record"
    rec = read_record_header(_fp)
    if (rec["lime_type"] not in datas) and (
        (rec["lime_type"] in parse_metadatas) or (rec["nbytes"] < maxsize)
    ):
        rec["data"] = read_struct(_fp, "%ds" % (rec["nbytes"],))[0]
    else:
        rec["data"] = None
    return rec


@open_file
def read_records(_fp, maxsize=1000):
    "Scans the content of a lime file and returns the list of records"

    fsize = file_size(_fp)

    records = []
    offset = 0
    while offset + HEADER_SIZE < fsize:
        _fp.seek(offset)
        records.append(read_record(_fp, maxsize=maxsize))
        offset = (
            records[-1]["offset"]
            + records[-1]["nbytes"]
            + ((8 - records[-1]["nbytes"] % 8) % 8)
        )

    return records


def write_record_header(_fp, lime_type, nbytes, begin=False, end=False):
    "Writes the header of a record to file"
    if isinstance(lime_type, str):
        lime_type = bytes(lime_type, "utf-8")
    if not isinstance(lime_type, bytes):
        raise TypeError("lime_type neither a string or bytes")
    if len(lime_type) > TYPE_LENGTH:
        raise ValueError(f"Length of lime_type exceeds {TYPE_LENGTH}")

    msg_bits = 0
    if begin:
        msg_bits |= 0b1 << 15
    if end:
        msg_bits |= 0b1 << 14

    write_struct(
        _fp,
        ">lHHQ%ds" % TYPE_LENGTH,
        MAGIC_NUMBER,
        FILE_VERSION_NUMBER,  # Version number
        msg_bits,
        nbytes,
        lime_type,
    )


@open_file(flag="wb")
def write_record(_fp, lime_type, data, begin=False, end=False):
    "Writes a record to file"
    if isinstance(data, bytes):
        nbytes = len(data)
    elif isinstance(data, numpy.ndarray):
        nbytes = data.nbytes
    elif isinstance(data, int):
        # Considering as number of bytes to skip
        nbytes = data
    else:
        raise TypeError(f"expected bytes or int, got {type(data)}")
    size = nbytes + (8 - nbytes % 8) % 8
    write_record_header(_fp, lime_type, nbytes, begin=begin, end=end)
    if isinstance(data, bytes):
        write_struct(_fp, "%ds" % size, data)
    elif isinstance(data, numpy.ndarray):
        data.tofile(_fp)
    else:
        _fp.seek(size, SEEK_CUR)


@open_file(flag="wb")
def write_records(_fp, records):
    "Writes a list of records to file"
    if not isinstance(records, dict):
        records = dict(records)
    last = len(records) - 1
    for idx, (key, val) in enumerate(records.items()):
        write_record(_fp, key, val, begin=(idx == 0), end=(idx == last))


def write_data(_fp, data, metadata=None, lime_type=None):
    "Writes a data object to file and its metadata"
    if lime_type is None:
        lime_type = datas[-1]
    records = {}
    if metadata:
        for key, fnc in write_metadatas.items():
            try:
                records[key] = fnc(metadata)
            except ValueError:
                pass
    records[lime_type] = data
    write_records(_fp, records)


def get_header_bytes(metadata, lime_type=None):
    "Returns the bytes of the header of metadata"
    out = BytesIO()
    write_data(out, metadata["nbytes"], metadata=metadata, lime_type=lime_type)
    return out.getvalue()


@open_file
def head(_fp):
    "Returns metadata of a lime file"

    records = read_records(_fp)
    keys = [rec["lime_type"] for rec in records]

    # Archive not allowed for now
    if not len(set(keys)) == len(keys):
        raise ValueError("Repeated records in file")

    records = {rec["lime_type"]: rec for rec in records}

    data = []
    for key in datas:
        if key in records:
            data.append(key)
    if not data:
        raise ValueError("No data record in file")
    if len(data) > 1:
        raise ValueError("More than one data record in file")
    data = data[0]

    header = Header()
    for key in parse_metadatas:
        if key in records:
            header.update(parse_metadatas[key](records[key]["data"]))

    header.update(
        nbytes=records[data]["nbytes"],
        _offset=records[data]["offset"],
        fortran_order=False,
    )

    if (
        header["nbytes"]
        != prod(header["shape"]) * numpy.dtype(header["dtype"]).itemsize
    ):
        raise ValueError("Size does not match shape and dtype")

    return header


def load(filename, chunks=None, comm=None, **kwargs):
    """
    High level interface function for lime load.
    Loads a numpy array from file either in serial or parallel.
    The parallelism is enabled by providing a valid communicator.

    Parameters
    ----------
    filename : str
        Filename of the numpy array to be loaded.
    chunks: list
        How to divide the data domain. This enables the Dask API.
    comm: MPI.Cartcomm
        A valid cartesian MPI Communicator.
    kwargs: dict
        Additional parameters can be passed to override metadata.
        E.g. shape, dtype, etc.

    Returns:
    --------
    local_array : list
        Returns a numpy array representing the local elements of the domain.
    """

    if comm is not None and chunks is not None:
        raise ValueError("chunks and comm parameters cannot be both set")

    metadata = head(filename)
    shape = metadata["shape"]
    dtype = metadata["dtype"]
    offset = metadata["_offset"]
    order = "F" if metadata["fortran_order"] else "C"

    if chunks is not None:

        daskio = DaskIO(filename)

        return daskio.load(
            shape, dtype, offset, chunks=chunks, order=order, metadata=metadata
        )

    if comm is not None:
        check_comm(comm)

        with MpiIO(comm, filename, mode="r") as mpiio:
            return from_array(mpiio.load(shape, dtype, order, offset), attrs=metadata)

    return from_array(
        numpy.fromfile(filename, dtype=dtype, count=prod(shape), offset=offset).reshape(
            shape
        ),
        attrs=metadata,
    )


def save(array, filename, comm=None, metadata=None):
    """
    High level interface function for lime load.
    Loads a numpy array from file either in serial or parallel.
    The parallelism is enabled by providing a valid communicator.

    Parameters
    ----------
    filename : str
        Filename of the numpy array to be loaded.
    chunks: list
        How to divide the data domain. This enables the Dask API.
    comm: MPI.Cartcomm
        A valid cartesian MPI Communicator.
    metadata: dict
        Additional metadata to write in the header
    """

    array, attrs = to_array(array)
    if not array.dtype.byteorder == ">":
        attrs["dtype"] = array.dtype.newbyteorder(">")
        array = array.astype(attrs["dtype"])

    if metadata:
        attrs.update(metadata)

    if is_dask_array(array):
        daskio = DaskIO(filename)
        header = get_header_bytes(attrs)
        return daskio.save(array, header=header)

    if comm is not None:
        check_comm(comm)

        with MpiIO(comm, filename, mode="w") as mpiio:
            global_shape, _, _ = mpiio.decomposition.compose(array.shape)
            attrs["shape"] = tuple(global_shape)
            attrs["nbytes"] = prod(global_shape) * attrs["dtype"].itemsize
            header = get_header_bytes(attrs)
            return mpiio.save(array, header=header)

    write_data(filename, array, attrs)
