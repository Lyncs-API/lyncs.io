class MpiIO:
    def __init__(self, comm, filename, dims=None):

        from mpi4py import MPI

        if isinstance(comm, MPI.Cartcomm):
            if dims is None:
                raise ValueError("Dims must be set to use Cartesian topology")
            self.decomposition = Cartesian(comm=comm, dims=dims)
        else:
            self.decomposition = Decomposition(comm=comm)

        self.comm = self.decomposition.comm
        self.rank = self.decomposition.rank
        self.size = self.decomposition.size
        self.filename = filename
        self.handler = None

    def file_open(self, mode):

        amode = __check_file_mode(mode)

        self.handler = MPI.File.Open(self.comm, self.filename, amode=amode)

    def load(self, domain, dtype, order, header_offset):

        if not isinstance(dtype, str):
            raise ValueError("dtype must be a string")

        sizes, subsizes, starts = self.decomposition.decompose(domain)

        # make sure we are receiving a numpy valid type
        if type(dtype).__module__ == np.__name__:
            etype = self.__dtype_to_mpi(dtype)
        else:
            raise NotImplementedError("Currently not supporting any other types")

        if order.upper == "C":
            mpi_order = MPI.ORDER_C
        else:
            raise ValueError("Currently noy supporting FORTRAN ordering")

        # construct the filetype, use fixed data-type
        filetype = etype.Create_subarray(sizes, subsizes, starts, order=mpi_order)
        filetype.Commit()

        # set the file view - skip header
        pos = self.handler.Get_position() + header_offset
        # move file pointer to beginning of array data
        self.handler.Set_view(pos, etype, filetype, datarep="native")

        # allocate space for local_array to hold data read from file
        # FIXME: Depending on how future formats will be used (eg HDF5) this might has to go
        local_array = numpy.empty(subsizes, dtype=dtype, order=order.upper)

        # collectively read the array from file
        self.handler.Read_all(local_array)

        return local_array

    def save(self):
        raise NotImplementedError("MPI-IO Save not implemented yet")

    def file_close():
        self.hanlder.Close()

    def __check_file_mode(self, mode):

        from mpi4py import MPI

        switcher = {
            MPI.MODE_RDONLY: "MPI.MODE_RDONLY - read only",
            MPI.MODE_RDWR: "MPI.MODE_RDWR - reading and writing",
            MPI.MODE_WRONLY: "MPI.MODE_WRONLY - write only",
            MPI.MODE_CREATE: "MPI.MODE_CREATE - create the file if it does not exist",
            MPI.MODE_EXCL: "MPI.MODE_EXCL - error if creating file that already exists",
            MPI.MODE_DELETE_ON_CLOSE: "MPI.MODE_DELETE_ON_CLOSE - delete file on close",
            MPI.MODE_UNIQUE_OPEN: "MPI.MODE_UNIQUE_OPEN - file will not be concurrently opened elsewhere",
            MPI.MODE_SEQUENTIAL: "MPI.MODE_SEQUENTIAL - file will only be accessed sequentially",
            MPI.MODE_APPEND: "MPI.MODE_APPEND - set initial position of all file pointers to end of file",
        }

        if switcher.get(mode) is None:
            raise ValueError("File access mode value is invalid.")

        return mode

    def __dtype_to_mpi(self, t):
        """
        Convert Numpy data type to MPI type
        Parameters
        ----------
        t : type
            Numpy data type.

        Returns:
        --------
        m : mpi4py.MPI.Datatype
            MPI data type corresponding to `t`.
        """
        if hasattr(MPI, "_typedict"):
            mpi_type = MPI._typedict[np.dtype(t).char]
        elif hasattr(MPI, "__TypeDict__"):
            mpi_type = MPI.__TypeDict__[np.dtype(t).char]
        else:
            raise ValueError("cannot convert type")
        return mpi_type


class Decomposition:
    """
    One dimensional domain decomposition
    """

    def __init__(self, comm=None):

        from mpi4py import MPI

        if not isinstance(comm, MPI.Comm):
            raise ValueError("Expected an MPI communicator")

        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def decompose(self, domain, workers=None, id=None):

        if not isinstance(domain, int):
            raise ValueError(
                "Domain Decomposition in 1D requires domain to be of integral type"
            )

        if (workers is not None) and (id is not None):
            low, hi = self.__split_work(domain, workers, id)
        else:
            low, hi = self.__split_work(domain, self.size, self.rank)

        sizes = domain
        subsizes = hi - low
        starts = low

        return sizes, subsizes, starts

    def __split_work(self, load, workers, id):
        """
        Uniformly distributes load over the dimension.
        Remaining load is assigned in reverse round robbin manner
        """

        part = int(load / workers)  # uniform distribution
        rem = load - part * workers

        # reverse round robbin assignment of the remaining work
        if id >= (workers - rem):
            part += 1
            low = part * id - (workers - rem)
            hi = part * (1 + id) - (workers - rem)
        else:
            low = part * id
            hi = part * (1 + id)

        return low, hi


class Cartesian(Decomposition):
    """
    Decompose data using Cartesian Decomposition
    of arbitrary dimensionality
    """

    def __init__(self, comm=None, dims=None):
        super().__init__(comm=comm)

        from mpi4py import MPI

        if not isinstance(comm, MPI.Cartcomm):
            raise ValueError("Expected a Cartesian MPI communicator")

        self.dims = MPI.Compute_dims(self.size, dims)
        self.coords = self.comm.Get_coords(self.rank)

    def decompose(self, domain):

        # TODO: Check that the data domain can be decomposed based on the topology's shape

        subsizes = list(domain)
        starts = [0] * len(domain)
        sizes = domain

        # Iterate over the dimensionality of the topology
        for dim in range(len(self.dims)):
            _, subsizes[dim], starts[dim] = super(Cartesian, self).decompose(
                domain[dim], workers=self.dims[dim], id=self.coords[dim]
            )

        return sizes, subsizes, starts
