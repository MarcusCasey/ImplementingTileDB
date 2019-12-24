import numpy as np
import sys
import tiledb

array_name = "research_project_step_3"


def create_array():
    ctx = tiledb.Ctx()

    dom = tiledb.Domain(ctx,
                        tiledb.Dim(ctx, name="rows", domain=(1, 10), tile=2, dtype=np.int32),
                        tiledb.Dim(ctx, name="cols", domain=(1, 10), tile=2, dtype=np.int32))

    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.int32)])

    tiledb.SparseArray.create(array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    with tiledb.SparseArray(ctx, array_name, mode='w') as A:
        I, J = [1, 1, 2, 1, 2, 2], [1, 2, 2, 4, 3, 4]
        data = np.array(([1, 2, 3, 4, 5, 6]));
        A[I, J] = data

def read_array(order):
    ctx = tiledb.Ctx()
    with tiledb.SparseArray(ctx, array_name, mode='r') as A:
        print("Non-empty domain: {}".format(A.nonempty_domain()))

  
        data = A.query(attrs=["a"], order=order, coords=True)[1:100]
        a_vals = data["a"]
        coords = data["coords"]

        for i in range(coords.shape[0]):
            print("Cell {} has data {}".format(str(coords[i]), str(a_vals[i])))


ctx = tiledb.Ctx()
if tiledb.object_type(ctx, array_name) != "array":
    create_array()
    write_array()

layout = ""
if len(sys.argv) > 1:
    layout = sys.argv[1]

order = 'C'
if layout == "col":
    order = 'F'
elif layout == "global":
    order = 'G'
else:
    order = 'C'

read_array(order)
