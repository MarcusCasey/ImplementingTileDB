import numpy as np
import sys
import tiledb

array_name = "research_project_step_1"


def create_array():
    ctx = tiledb.Ctx()

    dom = tiledb.Domain(ctx,
                        tiledb.Dim(ctx, name="rows", domain=(1, 10), tile=10, dtype=np.int32),
                        tiledb.Dim(ctx, name="cols", domain=(1, 10), tile=10, dtype=np.int32))

    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.int32)])

    tiledb.SparseArray.create(array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    with tiledb.SparseArray(ctx, array_name, mode='w') as A:
        I, J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = np.array(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        A[I, J] = data


def read_array():
    ctx = tiledb.Ctx()
    with tiledb.SparseArray(ctx, array_name, mode='r') as A:
        data = A[1:11]
        a_vals = data["a"]
        for i, coord in enumerate(data["coords"]):
            print("Cell (%d, %d) has data %d" % (coord[0], coord[1], a_vals[i]))


ctx = tiledb.Ctx()
if tiledb.object_type(ctx, array_name) != "array":
    create_array()
    write_array()

read_array()
