class Decomposition():
    """
    One dimensional domain decomposition
    """
    def __init__(self, comm=None):

        from mpi4py import MPI
        
        if not isinstance(comm, MPI.Comm):
            raise ValueError('Expected an MPI communicator') 
        
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
    
    def decompose(self, domain, workers=None, id=None):

        if not isinstance(domain, int):
            raise ValueError('Domain Decomposition in 1D requires domain to be of integral type')

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
            raise ValueError('Expected a Cartesian MPI communicator')
        
        self.dims = MPI.Compute_dims(self.size, dims)
        self.coords = self.comm.Get_coords(self.rank)

    def decompose(self, domain):

        # TODO: Check that the data domain can be decomposed based on the topology's shape
            
        subsizes = list(domain)
        starts = [0] * len(domain)
        sizes = domain

        # Iterate over the dimensionality of the topology
        for dim in range(len(self.dims)):
            _, subsizes[dim], starts[dim] = super(Cartesian, self).decompose(domain[dim], 
                                                                            workers=self.dims[dim],
                                                                            id=self.coords[dim])
        
        return sizes, subsizes, starts

