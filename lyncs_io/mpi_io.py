"""
Parallel IO using MPI
"""

__all__ = ["MpiIO", "Decomposition"]

import numpy


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


class Decomposition:
    """
    Decompose data using Cartesian/Domain Decomposition
    of arbitrary dimensions
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

    def __init__(self, comm=None):
        if (comm is None) or (not isinstance(comm, self.MPI.Comm)):
            raise TypeError("Expected an MPI communicator")

        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        if comm.topology is self.MPI.GRAPH or comm.topology is self.MPI.DIST_GRAPH:
            raise TypeError(f"comm is of unsupported type: {type(comm)}")
        if comm.topology is self.MPI.CART:
            self.dims = self.MPI.Compute_dims(self.size, comm.Get_dim())
            self.coords = self.comm.Get_coords(self.rank)
        elif isinstance(comm, self.MPI.Intracomm):
            self.dims = [self.size]
            self.coords = [self.rank]

    def decompose(self, domain):
        """
        Decompose data over a cartesian/normal communicator.
        Data domains of higher order compared to communicators order
        only decomposed on the slow moving indexes.

        Parameters
        ----------
        domain : list
            Contains the global size domain we are decomposing.

        Returns:
        --------
        sizes : list
            global size of the domain
        sub_sizes : list
            local size of the domain
        starts : list
            global starting position
        """

        sizes = list(domain)
        sub_sizes = list(domain)
        starts = [0] * len(domain)

        # Iterating over the dimensions of the topology
        # allows for decomposition of higher order data domains
        for dim in range(len(self.dims)):
            workers, proc_id = self.dims[dim], self.coords[dim]
            if domain[dim] < workers:
                raise ValueError(
                    "Domain size ({}) must be larger than the amount of workers({})".format(
                        domain[dim], workers
                    )
                )

            low = _split_work(domain[dim], workers, proc_id)
            high = _split_work(domain[dim], workers, proc_id + 1)

            sub_sizes[dim] = high - low
            starts[dim] = low

        return sizes, sub_sizes, starts

    def compose(self, domain):
        """
        Reconstruct global data domain and position of the
        local array relatively to the global domain.

        Parameters
        ----------
        domain : list
            Contains the local size domain of the array.

        Returns:
        --------
        sizes : list
            global size of the domain
        sub_sizes : list
            local size of the domain
        starts : list
            global starting position
        """
        sizes = list(domain)
        sub_sizes = list(domain)
        starts = [0] * len(domain)

        # Iterating over the dimensions of the topology
        # allows for composition of higher order data domains
        for dim in range(len(self.dims)):
            rdims = [False] * len(self.dims)
            rdims[dim] = True

            # sub-communicator for collective communications over
            # a single dimension in the topology
            if self.comm.topology is self.MPI.CART:
                subcomm = self.comm.Sub(remain_dims=rdims)
            else:
                subcomm = self.comm

            sub_size = subcomm.allgather(sub_sizes[dim])
            sizes[dim] = sum(sub_size)
            starts[dim] = numpy.cumsum([0] + sub_size)[self.coords[dim]]

        return sizes, sub_sizes, starts


def _split_work(load, workers, proc_id):
    """
    Uniformly distributes load over the dimension.
    Remaining load is assigned in reverse round robin manner

    Parameters
    ----------
    load : int
        load size to be assigned
    workers : int
        total processing elements work will be assigned to
    proc_id : int
        processing element for which the bound is calculated for

    Returns:
    --------
    bound : int
        low bound of the workload assigned to the worker
    """
    unifload = load // workers  # uniform distribution
    rem = load - unifload * workers

    # round-robin assignment of the remaining work
    if proc_id <= rem:
        bound = (unifload + 1) * proc_id
    else:
        bound = (unifload + 1) * rem + unifload * (proc_id - rem)

    return bound
