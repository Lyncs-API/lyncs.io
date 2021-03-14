import numpy


class FileWrapper:
    def __init__(self, handler):
        self.handler = handler

    def __getattr__(self, key):
        if key == "write":
            key = "Write"
        return self.handler.__getattribute__(key)


class MpiIO:
    """
    Parallel IO using MPI
    """

    @property
    def MPI(self):
        from mpi4py import MPI

        return MPI

    def __init__(self, comm, filename, mode=None):

        if isinstance(comm, self.MPI.Cartcomm):
            self.decomposition = Cartesian(comm=comm)
        else:
            self.decomposition = Decomposition(comm=comm)

        self.comm = self.decomposition.comm
        self.rank = self.decomposition.rank
        self.size = self.decomposition.size
        self.filename = filename
        self.handler = None

        if mode is None:
            mode = "r"

        # TODO: Remove this into a convert routine
        if mode == "r":
            self.mode = self.MPI.MODE_RDONLY
        elif mode == "w":
            self.mode = self.MPI.MODE_CREATE | self.MPI.MODE_WRONLY

    def __enter__(self):
        self.file_open(mode=self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_close()
        # TODO: Check how to treat exc_error flags

    def file_open(self, mode=None):

        # amode = self.__check_file_mode(mode)
        if mode is None:
            mode = self.mode

        self.handler = FileWrapper(
            self.MPI.File.Open(self.comm, self.filename, amode=mode)
        )

    def load(self, domain, dtype, order, header_offset):

        if self.handler is None:
            self.file_open(self.MPI.MODE_RDONLY)

        # if not isinstance(dtype, str):
        #     raise ValueError("dtype must be a string")

        sizes, subsizes, starts = self.decomposition.decompose(domain)

        # skip header
        pos = self.handler.Get_position() + header_offset

        self.set_view(domain, dtype, order, pos, sizes, subsizes, starts)

        # allocate space for local_array to hold data read from file
        # FIXME: Depending on how future formats will be used (eg HDF5) this might has to go
        local_array = numpy.empty(subsizes, dtype=dtype, order=order.upper())

        self.handler.Read_all(local_array)

        return local_array

    def save(self, array, header=None):
        # Assumes header is a dict with "shape" entry

        if self.handler is None:
            self.file_open(self.MPI.MODE_CREATE | self.MPI.MODE_WRONLY)

        if self.rank == 0:
            pos = self.handler.Get_position()
        else:
            pos = 0

        pos = self.comm.bcast(pos, root=0)

        global_shape, local_shape, local_start = self.decomposition.compose(array.shape)

        self.set_view(
            array.shape,
            numpy.array(array).dtype,
            "C",
            pos,
            global_shape,
            local_shape,
            local_start,
        )

        # collectively write the array to file
        self.handler.Write_all(array)

    def set_view(self, domain, dtype, order, pos, sizes, subsizes, starts):
        # make sure we are receiving a numpy valid type
        # FIXME: Need to find a way of checking is dtype is a valid string and then pass that through numpy.dtype
        if type(dtype).__module__ == numpy.__name__:
            etype = self.__dtype_to_mpi(dtype)
        else:
            raise NotImplementedError("Currently not supporting any other types")

        if order.upper() == "C":
            mpi_order = self.MPI.ORDER_C
        else:
            raise ValueError("Currently noy supporting FORTRAN ordering")

        # use fixed data-type
        filetype = etype.Create_subarray(sizes, subsizes, starts, order=mpi_order)
        filetype.Commit()

        self.handler.Set_view(pos, etype, filetype, datarep="native")

    def file_close(self):
        self.handler.Close()

    def file_handler(self):
        return self.handler

    def __check_file_mode(self, mode):

        MPI = self.MPI

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
        MPI = self.MPI

        if hasattr(MPI, "_typedict"):
            mpi_type = MPI._typedict[numpy.dtype(t).char]
        elif hasattr(MPI, "__TypeDict__"):
            mpi_type = MPI.__TypeDict__[numpy.dtype(t).char]
        else:
            raise ValueError("cannot convert type")
        return mpi_type


class Decomposition:
    """
    One dimensional domain decomposition on data domains
    of arbitrary dimensionality
    """

    from mpi4py import MPI

    def __init__(self, comm=None):

        if not isinstance(comm, self.MPI.Comm):
            raise ValueError("Expected an MPI communicator")

        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def decompose(self, domain, workers=None, id=None):
        """
        Decompose data over a 1D communicatior.
        Only the slowest moving index is decomposed.

        Parameters
        ----------
        domain : list or int
            Contains the size of domain we are decomposing.
        workers : int
            Size of available processing elements.
        id : int
            Index of the current processing element.

        Returns:
        --------
        size : list
            global size of the domain
        subsize : list
            local size of the domain
        start : list
            global starting position
        """
        size = list(domain)
        subsize = list(domain)
        start = [0] * len(domain)

        if (workers is None) or (id is None):
            workers, id = self.size, self.rank

        if size[0] < workers:
            raise ValueError(
                "Domain size ({}) must be larger than the amount of workers({})".format(
                    size[0], workers
                )
            )

        low = self.__split_work(size[0], workers, id)
        hi = self.__split_work(size[0], workers, id + 1)

        subsize[0] = hi - low
        start[0] = low

        return size, subsize, start

    def compose(self, domain, comm=None, id=None):

        size = list(domain)
        subsize = list(domain)
        start = [0] * len(domain)

        if comm is None or id is None:
            comm, id = self.comm, self.rank

        subsizes = comm.allgather(subsize[0])
        size[0] = sum(subsizes)
        start[0] = numpy.cumsum([0] + subsizes)[id]

        return size, subsize, start

    def __split_work(self, load, workers, id):
        """
        Uniformly distributes load over the dimension.
        Remaining load is assigned in reverse round robbin manner

        Parameters
        ----------
        load : int
            load size to be assigned
        workers : int
            total processing elements work will be assigned to
        id : int
            processing element for which the bound is calculated for

        Returns:
        --------
        bound : int
            low bound of the wokrload assigned to the worker
        """

        if not isinstance(load, int):
            raise ValueError("load expected to be an integer")
        if not isinstance(workers, int):
            raise ValueError("workers expected to be an integer")
        if not isinstance(id, int):
            raise ValueError("id expected to be an integer")

        n = int(load / workers)  # uniform distribution
        r = load - n * workers

        # round robbin assignment of the remaining work
        if id <= r:
            bound = (n + 1) * id
        else:
            bound = (n + 1) * r + n * (id - r)

        return bound


class Cartesian(Decomposition):
    """
    Decompose data using Cartesian Decomposition
    of arbitrary dimensionality
    """

    def __init__(self, comm=None):
        super().__init__(comm=comm)

        if not isinstance(comm, self.MPI.Cartcomm):
            raise ValueError("Expected a Cartesian MPI communicator")

        self.dims = self.MPI.Compute_dims(self.size, comm.Get_dim())
        self.coords = self.comm.Get_coords(self.rank)

    def decompose(self, domain):
        """
        Decompose data over a cartesian communicatior.
        Data domains of higher order compared to communicators order
        only decomposed on the slow moving indexes.

        Parameters
        ----------
        domain : list
            Contains the global size domain we are decomposing.

        Returns:
        --------
        size : one-dimensional list
            global size of the dimension
        subsize : one-dimensional list
            local size of the dimension
        start : one-dimensional list
            global starting position
        """

        sizes = list(domain)
        subsizes = list(domain)
        starts = [0] * len(domain)

        # Iterating over the dimensionality of the topology
        # allows for decomposition of higher order data domains
        for dim in range(len(self.dims)):
            _, ssz, st = super(Cartesian, self).decompose(
                [domain[dim]], workers=self.dims[dim], id=self.coords[dim]
            )
            subsizes[dim], starts[dim] = ssz[0], st[0]

        return sizes, subsizes, starts

    def compose(self, domain):

        sizes = list(domain)
        subsizes = list(domain)
        starts = [0] * len(domain)

        # Iterating over the dimensionality of the topology
        # allows for composition of higher order data domains
        for dim in range(len(self.dims)):
            rdims = [False] * len(self.dims)
            rdims[dim] = True
            # sub-communicator for collective communications over
            # a single dimension in the topology
            subcomm = self.comm.Sub(remain_dims=rdims)

            sz, ssz, st = super(Cartesian, self).compose(
                [domain[dim]], comm=subcomm, id=self.coords[dim]
            )
            sizes[dim], subsizes[dim], starts[dim] = sz[0], ssz[0], st[0]

        return sizes, subsizes, starts
