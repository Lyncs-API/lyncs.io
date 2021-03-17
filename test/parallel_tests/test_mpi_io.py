from lyncs_io import mpi_io as io
import pytest


@pytest.mark.mpi(min_size=2)
def test_comm_types():
    from mpi4py import MPI

    # Requires a communicator
    with pytest.raises(TypeError):
        io.Decomposition()

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
        io.Decomposition(comm=topo)

    topo.Free()

    # Check DistGraph topology
    sources = [rank]
    degrees = [3]
    destinations = [(rank - 1) % size, rank, (rank + 1) % size]
    topo = comm.Create_dist_graph(sources, degrees, destinations, MPI.UNWEIGHTED)

    with pytest.raises(TypeError):
        io.Decomposition(comm=topo)

    topo.Free()

    # Check Cartesian topology
    dims = MPI.Compute_dims(size, [0] * 2)
    topo = comm.Create_cart(dims=dims, periods=[False, False], reorder=False)
    decomp = io.Decomposition(comm=topo)

    assert dims == decomp.dims
    assert topo.Get_coords(rank) == decomp.coords

    topo.Free()

    # Check COMM_WORLD
    decomp = io.Decomposition(comm=comm)
    assert [size] == decomp.dims
    assert [rank] == decomp.coords


@pytest.mark.mpi(min_size=2)
def test_mpi_property():
    from mpi4py import MPI

    assert hasattr(io.Decomposition(MPI.COMM_WORLD), "MPI")


@pytest.mark.mpi(min_size=2)
def test_comm_Decomposition():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    dec = io.Decomposition(comm=comm)

    # No remainder
    domain = [8 * size, 12]
    globalsz, localsz, start = dec.decompose(domain=domain)

    assert [8 * size, 12] == globalsz
    assert [8, 12] == localsz
    if rank == 0:
        assert [0, 12] == start
    elif rank == 1:
        assert [8, 12] == start

    # Remainder=1
    domain = [8 * size + 1, 12]
    globalsz, localsz, start = dec.decompose(domain=domain)

    assert domain == globalsz
    if rank == 0:
        # First process takes the remainder
        assert [9, 12] == localsz
        assert [0, 12] == start
    elif rank == size - 1:
        assert [8, 12] == localsz
        assert [8 * (size - 1) + 1, 12] == start

    # More workers than data
    with pytest.raises(ValueError):
        dec.decompose(domain=[1])

    with pytest.raises(ValueError):
        dec.decompose(domain=[1, 8])


@pytest.mark.mpi(min_size=2)
def test_cart_decomposition():
    from mpi4py import MPI

    pass


@pytest.mark.mpi(min_size=2)
def test_comm_composition():
    from mpi4py import MPI

    pass


@pytest.mark.mpi(min_size=2)
def test_cart_composition():
    from mpi4py import MPI

    pass