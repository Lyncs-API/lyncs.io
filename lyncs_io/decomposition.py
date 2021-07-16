"""
Domain Decomposition
"""

__all__ = ["Decomposition"]

import numpy


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
            self.dims, _, self.coords = self.comm.Get_topo()
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
        if len(domain) < len(self.dims):
            raise ValueError(
                "Dimensionality of the domain ({}) must be larger than \
                    the dimensionality of the topology ({})".format(
                    len(domain), len(self.dims)
                )
            )
        sizes = list(domain)
        sub_sizes = list(domain)
        starts = [0] * len(domain)

        # Iterating over the dimensions of the topology
        # allows for decomposition of higher order data domains
        for dim in range(len(self.dims)):
            workers, proc_id = self.dims[dim], self.coords[dim]
            if domain[dim] < workers:
                raise ValueError(
                    f"Domain size ({domain[dim]}) for dimension {dim} must be \
                        larger than the amount of workers({workers})"
                )

            low = _split_work(domain[dim], workers, proc_id)
            high = _split_work(domain[dim], workers, proc_id + 1)

            sub_sizes[dim] = high - low
            starts[dim] = low

        return tuple(sizes), tuple(sub_sizes), tuple(starts)

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

        if len(domain) < len(self.dims):
            raise ValueError(
                "Dimensionality of the domain ({}) must be larger than \
                    the dimensionality of the topology ({})".format(
                    len(domain), len(self.dims)
                )
            )

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

        return tuple(sizes), tuple(sub_sizes), tuple(starts)


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
