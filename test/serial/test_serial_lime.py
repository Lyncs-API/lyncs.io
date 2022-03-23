import numpy
from lyncs_utils import read
from lyncs_io.convert import to_bytes, from_bytes
from lyncs_io.testing import tempdir, shape_loop, dtype_loop, generate_rand_arr
from lyncs_io.lime import (
    write_record_header,
    read_record_header,
    write_record,
    read_record,
    write_records,
    read_records,
    head,
    load,
    save,
    write_data,
    get_header_bytes,
)


def test_record_header(tempdir):
    filename = tempdir + "header"
    write_record_header(filename, "foo", 1234, begin=False, end=True)
    rec = read_record_header(filename)
    assert rec["lime_type"] == "foo"
    assert rec["nbytes"] == 1234
    assert rec["begin"] == False
    assert rec["end"] == True

    write_record_header(filename, "bar", 4567, begin=True, end=False)
    rec = read_record_header(filename)
    assert rec["lime_type"] == "bar"
    assert rec["nbytes"] == 4567
    assert rec["begin"] == True
    assert rec["end"] == False


def test_record(tempdir):
    filename = tempdir + "record"
    data = b"A very random string 123456"
    write_record(filename, "foo", data, begin=False, end=True)
    rec = read_record(filename)
    assert rec["lime_type"] == "foo"
    assert rec["nbytes"] == len(data)
    assert rec["data"] == data
    assert rec["begin"] == False
    assert rec["end"] == True


@shape_loop
@dtype_loop
def test_records(tempdir, shape, dtype):
    filename = tempdir + "records"
    records = {
        "foo": numpy.random.rand(100).tobytes(),
        "bar": numpy.random.rand(100).tobytes(),
        "size_only": 1234,
    }

    write_records(filename, records)
    recs = read_records(filename)
    last = len(records) - 1

    for i, (rec, (key, val)) in enumerate(zip(recs, records.items())):
        assert rec["lime_type"] == key
        assert rec["begin"] == (i == 0)
        assert rec["end"] == (i == last)
        if isinstance(val, int):
            assert rec["nbytes"] == val
        else:
            assert rec["nbytes"] == len(val)
            assert rec["data"] == val


@shape_loop
@dtype_loop
def test_get_header_bytes(tempdir, shape, dtype):
    arr = generate_rand_arr(shape, dtype)
    arr, attrs = to_bytes(arr)
    filename = tempdir + "header"
    write_data(filename, len(arr), metadata=attrs)
    header = get_header_bytes(attrs)
    ref = read(filename, len(header))
    assert header == ref


def test_reference():
    arr = load("test/conf.unity")
    assert arr.shape == (4, 4, 4, 4, 4, 3, 3)
    assert (arr == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]).all()
    assert arr.dtype == ">c16"


@shape_loop
@dtype_loop
def test_save(tempdir, shape, dtype):
    filename = tempdir + "rand"
    arr = generate_rand_arr(shape, dtype)
    save(arr, filename)
    header = head(filename)
    assert header["shape"] == shape
    assert header["nbytes"] == arr.nbytes
    assert header["dtype"] == numpy.dtype(dtype).newbyteorder(">")
    assert (arr == load(filename)).all()
