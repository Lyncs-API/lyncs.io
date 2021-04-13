from lyncs_io import mpi_io as io
import pytest

from lyncs_io.testing import mark_mpi

# TODO: Generalize on higher dimensions (Currently tested for cart_dim<=2)


@mark_mpi
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
    ndims = 2
    dims = MPI.Compute_dims(size, [0] * ndims)
    topo = comm.Create_cart(dims=dims, periods=[False] * ndims, reorder=False)
    decomp = io.Decomposition(comm=topo)

    assert dims == decomp.dims
    assert topo.Get_coords(rank) == decomp.coords

    topo.Free()

    # Check COMM_WORLD
    decomp = io.Decomposition(comm=comm)
    assert [size] == decomp.dims
    assert [rank] == decomp.coords


@mark_mpi
def test_mpi_property():
    from mpi4py import MPI

    assert hasattr(io.Decomposition(MPI.COMM_WORLD), "MPI")


@mark_mpi
def test_comm_Decomposition():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    dec = io.Decomposition(comm=comm)

    # No remainder
    domain = [8 * size, 12]
    globalsz, localsz, start = dec.decompose(domain=domain)

    assert domain == globalsz
    assert [8, 12] == localsz
    if rank == 0:
        assert [0, 0] == start
    elif rank == size - 1:
        assert [8 * (size - 1), 0] == start

    # Remainder=1
    domain = [8 * size + 1, 12]
    globalsz, localsz, start = dec.decompose(domain=domain)

    assert domain == globalsz
    if rank == 0:
        # First process takes the remainder
        assert [9, 12] == localsz
        assert [0, 0] == start
    elif rank == size - 1:
        assert [8, 12] == localsz
        assert [8 * (size - 1) + 1, 0] == start

    # More workers than data
    with pytest.raises(ValueError):
        dec.decompose(domain=[0] * len(domain))


@mark_mpi
def test_cart_decomposition():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    # TODO: Ensure testing generalizes in arbitrary dimension
    ndims = 2
    dims = MPI.Compute_dims(size, [0] * ndims)
    topo = comm.Create_cart(dims=dims, periods=[False] * ndims, reorder=False)
    coords = topo.Get_coords(rank)
    dec = io.Decomposition(comm=topo)

    # No remainder
    domain = [8 * dims[0], 8 * dims[1], 4, 4]
    globalsz, localsz, start = dec.decompose(domain=domain)

    assert domain == globalsz
    assert [8, 8, 4, 4] == localsz
    if coords[0] == 0 and coords[1] == 0:
        assert [0, 0, 0, 0] == start
    elif coords[0] == dims[0] and coords[1] == dims[1]:
        assert [8 * (dims[0] - 1), 8 * (dims[1] - 1), 0, 0] == start

    # Remainder=1 in each dimension
    domain = [8 * dims[0] + 1, 8 * dims[1] + 1, 4, 4]
    globalsz, localsz, start = dec.decompose(domain=domain)

    assert domain == globalsz
    if coords[0] == 0 and coords[1] == 0:
        assert [9, 9, 4, 4] == localsz
        assert [0, 0, 0, 0] == start
    elif coords[0] == dims[0] and coords[1] == dims[1]:
        assert [8 * (dims[0] - 1) + 1, 8 * (dims[1] - 1) + 1, 0, 0] == start

    # More workers than data
    with pytest.raises(ValueError):
        dec.decompose(domain=[0] * len(domain))


@mark_mpi
def test_comm_composition():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    dec = io.Decomposition(comm=comm)

    # No remainder
    local_size = [8, 8]
    globalsz, localsz, start = dec.compose(local_size)

    assert [size * 8, 8] == globalsz
    assert local_size == localsz
    assert [rank * 8, 0] == start

    # Remainder=1
    if rank == 0:
        local_size = [9, 8]

    globalsz, localsz, start = dec.compose(local_size)

    assert [size * 8 + 1, 8] == globalsz
    assert local_size == localsz
    if rank == 0:
        assert [rank * 8, 0] == start
    else:
        assert [rank * 8 + 1, 0] == start


@mark_mpi
def test_cart_composition():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    # TODO: Ensure testing generalizes in arbitrary dimension
    ndims = 2
    dims = MPI.Compute_dims(size, [0] * ndims)
    topo = comm.Create_cart(dims=dims, periods=[False] * ndims, reorder=False)
    coords = topo.Get_coords(rank)
    dec = io.Decomposition(comm=topo)

    # No remainder
    local_size = [8, 8, 4, 4]
    globalsz, localsz, start = dec.compose(domain=local_size)

    assert [dims[0] * 8, dims[1] * 8, 4, 4] == globalsz
    assert local_size == localsz
    assert [coords[0] * 8, coords[1] * 8, 0, 0] == start

    # Remainder=1 in horizontal dimension
    local_size = [8, 8, 4, 4]
    if coords[0] == 0:
        local_size = [9, 8, 4, 4]

    globalsz, localsz, start = dec.compose(local_size)

    assert [dims[0] * 8 + 1, dims[1] * 8, 4, 4] == globalsz
    assert local_size == localsz

    if coords[0] > 0:
        assert [coords[0] * 8 + 1, coords[1] * 8, 0, 0] == start
    else:
        assert [0, coords[1] * 8, 0, 0] == start

    # Remainder=1 in vertical dimension
    local_size = [8, 8, 4, 4]
    if coords[1] == 0:
        local_size = [8, 9, 4, 4]

    globalsz, localsz, start = dec.compose(local_size)

    assert [dims[0] * 8, dims[1] * 8 + 1, 4, 4] == globalsz
    assert local_size == localsz

    if coords[1] > 0:
        assert [coords[0] * 8, coords[1] * 8 + 1, 0, 0] == start
    else:
        assert [coords[0] * 8, 0, 0, 0] == start

    # Remainder=1 in each dimension
    local_size = [8, 8, 4, 4]
    if coords[0] == 0 and coords[1] == 0:
        local_size = [9, 9, 4, 4]
    elif coords[1] == 0:
        local_size = [8, 9, 4, 4]
    elif coords[0] == 0:
        local_size = [9, 8, 4, 4]

    globalsz, localsz, start = dec.compose(local_size)

    assert [dims[0] * 8 + 1, dims[1] * 8 + 1, 4, 4] == globalsz
    assert local_size == localsz

    if coords[0] == 0 and coords[1] == 0:
        assert [0, 0, 0, 0] == start
    elif coords[0] == 0 and coords[1] > 0:
        assert [0, coords[1] * 8 + 1, 0, 0] == start
    elif coords[0] > 0 and coords[1] == 0:
        assert [coords[0] * 8 + 1, 0, 0, 0] == start
    else:
        assert [coords[0] * 8 + 1, coords[1] * 8 + 1, 0, 0] == start
