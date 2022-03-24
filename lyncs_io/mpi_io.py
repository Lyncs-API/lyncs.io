"""
Parallel IO using MPI
"""

__all__ = ["MpiIO"]

from contextlib import contextmanager
import tempfile
import os
import numpy

# pylint: disable=C0103
try:
    import mpi4py

    with_mpi = True
except ImportError:
    with_mpi = False


from .decomposition import Decomposition


def check_comm(comm):
    "Raises error if comm is not valid"
    if not hasattr(comm, "size"):
        raise TypeError(
            "comm variable needs to be a valid MPI communicator with size attribute."
        )


def _tempdir_MPI(comm=None):
    """
    Creates a temporary directory to be used during testing
    """

    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD
    if comm.rank == 0:
        tmp = tempfile.TemporaryDirectory()
        name = tmp.__enter__()
    else:
        name = ""
    path = comm.bcast(name, root=0)

    # test path exists for all
    has_access = os.path.exists(path) and os.access(path, os.R_OK | os.W_OK)
    all_access = comm.allreduce(has_access, op=MPI.LAND)
    if not all_access:
        raise ValueError(
            "Some processes are unable to access the temporary directory. \n\
                Set TMPDIR, TEMP or TMP environment variables with the temporary \n\
                directory to be used across processes. "
        )

    yield path + "/"

    # make sure file exists until everyone is done
    comm.Barrier()
    if comm.rank == 0:
        tmp.__exit__(None, None, None)


tempdir_MPI = contextmanager(_tempdir_MPI)


class MpiIO:
    """
    Class for handling file handling routines and Parallel IO using MPI
    """

    # pylint: disable=C0103
    @property
    def MPI(self):
        """
        Property for importing MPI wherever necessary
        """
        # pylint: disable=C0415
        try:
            from mpi4py import MPI
        except ImportError as err:
            raise ImportError(
                "MPI not available. Consider installing `lyncs_io[mpi]`."
            ) from err

        return MPI

    def __init__(self, comm, filename, mode="r"):

        self.decomposition = Decomposition(comm=comm)

        self.comm = self.decomposition.comm
        self.rank = self.decomposition.rank
        self.size = self.decomposition.size
        self.filename = filename
        self.handler = None
        self.mode = mode

    def __enter__(self):
        self._file_open(mode=self.mode)
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self._file_close()

    def load(self, domain, dtype, order, header_offset):
        """
        Reads the local domain from a file and loads it in a numpy array

        Parameters
        ----------
        domain : list
            Global data domain.
        dtype: data-type
            numpy data-type for the array
        order: str
            whether data are stored in row/column
            major ('C', 'F') order in memory
        header_offset: int
            offset in bytes to where the
            data start in the file.

        Returns:
        --------
        local_array : numpy array
            Local data to the process
        """

        # skip header
        pos = self.handler.Get_position() + header_offset

        self._set_view(domain, dtype, order, pos)

        # allocate space for local_array to hold data read from file
        _, subsizes, _ = self.decomposition.decompose(domain)
        local_array = numpy.empty(subsizes, dtype=dtype, order=order.upper())

        self.handler.Read_all(self._array_view(local_array))

        return local_array

    def save(self, array, header=None, offset=None):
        """
        Writes the local array in a file in parallel

        Parameters
        ----------
        local_array : numpy array
            Local data to the process
        """
        if not numpy.isfortran(array):
            # ensure data are contiguous
            array = numpy.ascontiguousarray(array)
        else:
            raise NotImplementedError("Currently noy supporting FORTRAN ordering")

        if offset is None:
            if header:
                offset = len(header)
            else:
                offset = 0

        if self.rank == 0 and header:
            self.handler.Write(header)

        self._set_view(array.shape, array.dtype, "C", offset, compose=True)

        # collectively write the array to file
        self.handler.Write_all(self._array_view(array))

    def _file_open(self, mode=None):
        if mode is None:
            mode = self._to_mpi_file_mode(self.mode)
        else:
            mode = self._to_mpi_file_mode(mode)

        self.handler = self._FileWrapper(
            self.MPI.File.Open(self.comm, self.filename, amode=mode)
        )

    def _file_close(self):
        self.handler.Close()

    def _set_view(self, domain, dtype, order, pos, compose=None):

        if compose is True:
            sizes, subsizes, starts = self.decomposition.compose(domain)
        else:
            sizes, subsizes, starts = self.decomposition.decompose(domain)

        # assumes numpy valid type
        etype = self._dtype_to_mpi(dtype)

        if order.upper() == "C":
            mpi_order = self.MPI.ORDER_C
        else:
            raise NotImplementedError("Currently noy supporting FORTRAN ordering")

        # use fixed data-type
        filetype = etype.Create_subarray(sizes, subsizes, starts, order=mpi_order)
        filetype.Commit()

        self.handler.Set_view(pos, etype, filetype, datarep="native")

    def _to_mpi_file_mode(self, mode):
        MPI = self.MPI

        switcher = {
            "r": MPI.MODE_RDONLY,
            "w": MPI.MODE_CREATE | MPI.MODE_WRONLY,
            "a": MPI.MODE_APPEND,
            "r+": MPI.MODE_RDWR,
            "w+": MPI.MODE_CREATE | MPI.MODE_RDWR,
        }

        if switcher.get(mode) is None:
            raise ValueError("File access mode value is invalid.")

        return switcher.get(mode)

    def _array_view(self, array):
        "Array view for MPI functions"
        return array.view(self._dtype_to_mpi(array.dtype, get_key=True))

    def _dtype_to_mpi(self, np_type, get_key=False):
        """
        Convert Numpy data type to MPI type

        Parameters
        ----------
        np_type : type
            Numpy data type.

        Returns:
        --------
        mpi_type : mpi4py.MPI.Datatype
            MPI data type corresponding to `np_type`.
        """

        np_type = numpy.dtype(np_type)
        if np_type.byteorder == ">":
            np_type = np_type.newbyteorder("<")

        if hasattr(self.MPI, "_typedict"):
            types = self.MPI._typedict
        elif hasattr(self.MPI, "__TypeDict__"):
            types = self.MPI.__TypeDict__
        else:
            raise RuntimeError("Types dict not found")

        for key, val in types.items():
            try:
                if numpy.dtype(key) == np_type:
                    return key if get_key else val
            # for keys that are not understood
            except TypeError:
                continue

        raise TypeError(f"{np_type} is not supported")

    class _FileWrapper:
        """
        File Wrapper for using MPI Write with numpy write
        """

        def __init__(self, handler):
            self.handler = handler

        def __getattr__(self, key):
            if key == "write":
                key = "Write"
            return self.handler.__getattribute__(key)
