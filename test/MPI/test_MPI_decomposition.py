import pytest
from lyncs_io.decomposition import Decomposition
from lyncs_io.testing import (
    mark_mpi,
    tempdir_MPI,
    shape_loop,
    parallel_loop,
)


@mark_mpi
def test_MPI_decomposition_comm_types():
    from mpi4py import MPI

    # Requires a communicator
    with pytest.raises(TypeError):
        Decomposition()

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    # Check Graph topology
    index, edges = [0], []
    for i in range(size):
        pos = index[-1]
        index.append(pos + 2)
        edges.append((i - 1) % size)
        edges.append((i + 1) % size)
    topo = comm.Create_graph(index[1:], edges)

    with pytest.raises(TypeError):
        Decomposition(comm=topo)

    topo.Free()

    # Check DistGraph topology
    sources = [rank]
    degrees = [3]
    destinations = [(rank - 1) % size, rank, (rank + 1) % size]
    topo = comm.Create_dist_graph(sources, degrees, destinations, MPI.UNWEIGHTED)

    with pytest.raises(TypeError):
        Decomposition(comm=topo)

    topo.Free()

    # Check Cartesian topology
    ndims = 2
    dims = MPI.Compute_dims(size, [0] * ndims)
    topo = comm.Create_cart(dims=dims, periods=[False] * ndims, reorder=False)
    decomp = Decomposition(comm=topo)

    assert dims == decomp.dims
    assert topo.Get_coords(rank) == decomp.coords

    topo.Free()

    # Check COMM_WORLD
    decomp = Decomposition(comm=comm)
    assert [size] == decomp.dims
    assert [rank] == decomp.coords


@mark_mpi
def test_MPI_decomposition_mpi_property():
    from mpi4py import MPI

    assert hasattr(Decomposition(MPI.COMM_WORLD), "MPI")


@mark_mpi
@shape_loop
def test_MPI_decomposition_Decomposition_Comm(tempdir_MPI, shape):
    from mpi4py import MPI

    topo = MPI.COMM_WORLD
    dec = Decomposition(comm=topo)

    if len(shape) < len(dec.dims):
        with pytest.raises(ValueError):
            dglobalsz, dlocalsz, dstart = dec.decompose(shape)
        with pytest.raises(ValueError):
            cglobalsz, clocalsz, cstart = dec.compose(shape)
    elif any(x < y for x, y in zip(shape, dec.dims)):
        with pytest.raises(ValueError):
            dglobalsz, dlocalsz, dstart = dec.decompose(shape)
    else:
        dglobalsz, dlocalsz, dstart = dec.decompose(shape)
        cglobalsz, clocalsz, cstart = dec.compose(dlocalsz)

        assert dglobalsz == cglobalsz
        assert dlocalsz == clocalsz
        assert dstart == cstart


@mark_mpi
@parallel_loop
@shape_loop
def test_MPI_decomposition_Decomposition_Cartesian(tempdir_MPI, procs, shape):
    from mpi4py import MPI

    topo = MPI.COMM_WORLD.Create_cart(dims=procs)
    dec = Decomposition(comm=topo)

    if len(shape) < len(dec.dims):
        with pytest.raises(ValueError):
            dglobalsz, dlocalsz, dstart = dec.decompose(shape)
        with pytest.raises(ValueError):
            cglobalsz, clocalsz, cstart = dec.compose(shape)
    elif any(x < y for x, y in zip(shape, dec.dims)):
        with pytest.raises(ValueError):
            dglobalsz, dlocalsz, dstart = dec.decompose(shape)
    else:
        dglobalsz, dlocalsz, dstart = dec.decompose(shape)
        cglobalsz, clocalsz, cstart = dec.compose(dlocalsz)

        assert dglobalsz == cglobalsz
        assert dlocalsz == clocalsz
        assert dstart == cstart
