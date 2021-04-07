import numpy


class DaskIO:
    """
    Class for handling file handling routines and Parallel IO using Dask
    """

    @property
    def dask(self):
        import dask

        return dask

    def __init__(self, chunks, filename, mode="r"):

        self.filename = filename
        self.handler = None
        self.mode = mode

    def load(self, domain, dtype, header_offset, chunks=None, order="C"):

        array = self._memmap(
            self.filename,
            mode="r",
            shape=domain,
            dtype=dtype,
            offset=header_offset,
            order=order,
        )

        return self.dask.array.from_array(array, chunks=chunks)

    def save(self, array, chunks=None):

        if not isinstance(array, self.dask.array.Array):
            array = self.dask.array.from_array(array, chunks=chunks)

        coords, chunksizes = self.pack_coords_chunks(array)
        starts = self.get_starts(coords, chunksizes)
        sl = self.get_slices(starts, chunksizes)

        tasks = []
        for block in range(self.prod(array.numblocks)):
            # FIXME: Each task writes to the same position
            task = self.dask.delayed(self._write_chunk)(
                self.filename,
                array,
                shape=array.shape,
                dtype=array.dtype,
                offset=0,
                sl=sl[block],
            )

            tasks.append(task)

        return tasks

    def prod(self, val):
        res = 1
        for ele in val:
            res *= ele
        return res

    def get_starts(self, coords, chunksizes):
        lststarts = []
        # FIXME: For uneven chunk-sizes last chunks do not much the starts
        for a, b in zip(coords, chunksizes):
            lststarts.append([(x * y) for x, y in zip(a, b)])

        starts = [tuple(x) for x in lststarts]
        return starts

    def get_slices(self, starts, chunksizes):
        lstsl = []
        for a, b in zip(starts, chunksizes):
            lstsl.append([slice(ai, ai + bi) for ai, bi in zip(a, b)])

        sl = [tuple(x) for x in lstsl]
        return sl

    def pack_coords_chunks(self, array):
        coords = []
        chunksizes = []

        for idx in range(self.prod(array.numblocks)):
            blockid = numpy.unravel_index(idx, array.numblocks)
            coords.append(blockid)
            chunksizes.append(array.blocks[blockid].chunksize)
        return coords, chunksizes

    def _memmap(
        self, filename, mode="r", shape=None, dtype=None, offset=None, order=None
    ):
        return numpy.memmap(
            filename, mode=mode, shape=shape, dtype=dtype, offset=offset, order=order
        )

    def _write_chunk(
        self, filename, array, shape=None, dtype=None, offset=None, order=None, sl=None
    ):
        data = numpy.memmap(
            filename, mode="w+", shape=shape, dtype=dtype, offset=offset, order=order
        )
        data[sl] = array[sl]
        return data
