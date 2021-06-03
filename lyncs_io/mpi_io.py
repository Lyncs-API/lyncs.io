"""
Parallel IO using MPI
"""

__all__ = ["MpiIO"]

import numpy

from .decomposition import Decomposition


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
        from mpi4py import MPI

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

        self.handler.Read_all(local_array)

        return local_array

    def save(self, array):
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

        if self.rank == 0:
            pos = self.handler.Get_position()
        else:
            pos = 0

        pos = self.comm.bcast(pos, root=0)

        self._set_view(array.shape, numpy.array(array).dtype, "C", pos, compose=True)

        # collectively write the array to file
        self.handler.Write_all(array)

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

    def _dtype_to_mpi(self, np_type):
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

        if hasattr(self.MPI, "_typedict"):
            mpi_type = self.MPI._typedict[numpy.dtype(np_type).char]
        elif hasattr(self.MPI, "__TypeDict__"):
            mpi_type = self.MPI.__TypeDict__[numpy.dtype(np_type).char]
        else:
            raise ValueError("cannot convert type")
        return mpi_type

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
