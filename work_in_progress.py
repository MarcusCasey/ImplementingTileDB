#note this class is based off of tileDB's test case with minor modifications. This test case is to be implemented with research project but is unfinished, with reasons detailed in the research paper.

class SparseArray(DiskTestCase):

    @unittest.expectedFailure
    def test_simple_1d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int))
        att = tiledb.Attr(ctx, dtype=int)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4])
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[1, 2]], values)

    @unittest.expectedFailure
    def test_simple_2d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                            tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int),
                            tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4], dtype=float)
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[1, 2], [1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[1, 2], [1, 2]], values)

    @unittest.expectedFailure
    def test_simple3d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                            tiledb.Dim(ctx, "x", domain=(0, 3), tile=4, dtype=int),
                            tiledb.Dim(ctx, "y", domain=(0, 3), tile=4, dtype=int),
                            tiledb.Dim(ctx, "z", domain=(0, 3), tile=4, dtype=int))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4], dtype=float)
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[1, 2], [1, 2], [1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[1, 2], [1, 2], [1, 2]], values)

    @unittest.expectedFailure
    def test_sparse_ordered_fp_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, "x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[2.5, 4.2]] = values
        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[2.5, 4.2]], values)

    @unittest.expectedFailure
    def test_sparse_unordered_fp_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, "x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)
        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[4.2, 2.5]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[2.5, 4.2]], values[::-1])

    @unittest.expectedFailure
    def test_multiple_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
            tiledb.Dim(ctx, domain=(1, 10), tile=10, dtype=int),
            tiledb.Dim(ctx, domain=(1, 10), tile=10, dtype=int))
        attr_int = tiledb.Attr(ctx, "ints", dtype=int)
        attr_float = tiledb.Attr(ctx, "floats", dtype="float")
        schema = tiledb.ArraySchema(ctx,
                                domain=dom,
                                attrs=(attr_int, attr_float,),
                                sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        I = np.array([1, 1, 1, 2, 3, 3, 3, 4])
        J = np.array([1, 2, 4, 3, 1, 6, 7, 5])

        V_ints = np.array([0, 1, 2, 3, 4, 6, 7, 5])
        V_floats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 5.0])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[I, J] = V
        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            R = T[I, J]
        assert_array_equal(V["ints"], R["ints"])
        assert_array_equal(V["floats"], R["floats"])

        # check error attribute does not exist
        # TODO: should this be an attribute error?
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            V["foo"] = V["ints"].astype(np.int8)
            with self.assertRaises(tiledb.TileDBError):
                T[I, J] = V

            # check error ncells length
            V["ints"] = V["ints"][1:2].copy()
            with self.assertRaises(AttributeError):
                T[I, J] = V

    def test_subarray(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                tiledb.Dim(ctx, "x", domain=(1, 10000), tile=100, dtype=int))
        att = tiledb.Attr(ctx, "", dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            self.assertIsNone(T.nonempty_domain())

        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[50, 60, 100]] = [1.0, 2.0, 3.0]

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            self.assertEqual(((50, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["coords"]["x"], [50, 60])

            #TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], [1.0, 2.0])
            self.assertEqual(("coords" in res), False)