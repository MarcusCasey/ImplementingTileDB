import numpy as np
import sys
import tiledb

array_name = "step_2"


def create_array():
    ctx = tiledb.Ctx()

    dom = tiledb.Domain(ctx,
                        tiledb.Dim(ctx, name="rows", domain=(1, 4), tile=4, dtype=np.int32),
                        tiledb.Dim(ctx, name="cols", domain=(1, 4), tile=4, dtype=np.int32))

    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.int32)])

    tiledb.SparseArray.create(array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    with tiledb.SparseArray(ctx, array_name, mode='w') as A:
        I, J = [1, 2, 2], [1, 4, 3]
        data = np.array(([1, 2, 3]))
        A[I, J] = data

        I, J = [4, 2], [1, 4]
        data = np.array(([4, 20]))
        A[I, J] = data

def read_array():
    ctx = tiledb.Ctx()
    with tiledb.SparseArray(ctx, array_name, mode='r') as A:
        # Slice entire array
        data = A[1:5, 1:5]
        a_vals = data["a"]
        for i, coord in enumerate(data["coords"]):
            print("Cell (%d, %d) has data %d" % (coord[0], coord[1], a_vals[i]))


ctx = tiledb.Ctx()
if tiledb.object_type(ctx, array_name) != "array":
    create_array()
    write_array()

read_array()
