import unittest
import ROOT
import numpy as np


class TTreeAsMatrix(unittest.TestCase):
    # Helpers
    def make_tree(self, *dtypes):
        tree = ROOT.TTree("test", "description")
        col_names = ["col{}".format(i) for i in range(len(dtypes))]

        col_vars = []
        for dtype in dtypes:
            if "F" in dtype:
                var = np.empty(1, dtype=np.float32)
            elif "D" in dtype:
                var = np.empty(1, dtype=np.float64)
            elif "I" in dtype:
                var = np.empty(1, dtype=np.int32)
            elif "i" in dtype:
                var = np.empty(1, dtype=np.uint32)
            elif "L" in dtype:
                var = np.empty(1, dtype=np.int64)
            elif "l" in dtype:
                var = np.empty(1, dtype=np.uint64)
            elif "S" in dtype:
                var = np.empty(1, dtype=np.int16)
            else:
                raise Exception(
                    "Type {} not known to create branch.".format(dtype))
            col_vars.append(var)

        for dtype, name, var in zip(dtypes, col_names, col_vars):
            tree.Branch(name, var, name + "/" + dtype)

        reference = []
        for i in range(10):
            row = []
            for i_var, var in enumerate(col_vars):
                var[0] = i + i_var
                row.append(var[0])
            reference.append(row)
            tree.Fill()

        return tree, reference, dtypes, col_names, col_vars

    def make_example(self, *dtypes):
        tree, reference, dtypes, col_names, col_vars = self.make_tree(*dtypes)
        matrix_ttree = tree.AsMatrix(col_names)
        matrix_ref = np.asarray(reference)
        return matrix_ttree, matrix_ref

    # Tests
    def test_instance(self):
        tree, _, _, col_names, _ = self.make_tree("F", "F")
        tree.AsMatrix(col_names)

    def test_not_supported_dtype(self):
        tree, _, _, col_names, _ = self.make_tree("F", "S")
        try:
            tree.AsMatrix(col_names)
            self.assertFail()
        except Exception as exception:
            self.assertEqual(
                "Branch col1 of tree test has not supported data-type Short_t.",
                exception.args[0])

    def test_not_existent_column(self):
        tree, _, _, col_names, _ = self.make_tree("F", "F")
        try:
            tree.AsMatrix(["foo"])
            self.assertFail()
        except Exception as exception:
            self.assertEqual("Tree test has no branch foo.", exception.args[0])

    def test_shape(self):
        matrix_ttree, matrix_ref = self.make_example("F", "F")
        for i in range(2):
            self.assertEqual(matrix_ttree.shape[i], matrix_ref.shape[i])

    def test_take_all_columns(self):
        tree, reference, _, _, _ = self.make_tree("F", "F")
        matrix_ttree = tree.AsMatrix()
        matrix_ref = np.asarray(reference)
        for i in range(matrix_ref.shape[0]):
            for j in range(matrix_ref.shape[1]):
                self.assertEqual(matrix_ttree[i, j], matrix_ref[i, j])

    def test_values(self):
        matrix_ttree, matrix_ref = self.make_example("F", "F")
        for i in range(matrix_ref.shape[0]):
            for j in range(matrix_ref.shape[1]):
                self.assertEqual(matrix_ttree[i, j], matrix_ref[i, j])

    def test_zero_entries(self):
        tree = ROOT.TTree("tree", "tree")
        var = np.empty(1, np.float32)
        tree.Branch("col", var, "col/F")
        try:
            tree.AsMatrix(["col"])
            self.assertFail()
        except Exception as exception:
            self.assertEqual("Tree has no entries.", exception.args[0])

    def test_allowed_dtypes(self):
        matrix_ttree, matrix_ref = self.make_example("F", "D", "I", "i", "L",
                                                     "l")
        for i in range(matrix_ref.shape[0]):
            for j in range(matrix_ref.shape[1]):
                self.assertEqual(matrix_ttree[i, j], matrix_ref[i, j])

    def test_dtype_conversion(self):
        # all -> double
        matrix_ttree, matrix_ref = self.make_example("F", "D", "I", "i", "L",
                                                     "l")
        self.assertEqual(matrix_ttree.dtype, np.float64)

        # float, float -> float
        matrix_ttree, matrix_ref = self.make_example("F", "F")
        self.assertEqual(matrix_ttree.dtype, np.float32)

        # unsigned int, unsigned int -> unsigned int
        matrix_ttree, matrix_ref = self.make_example("i", "i")
        self.assertEqual(matrix_ttree.dtype, np.uint32)

        # float, int -> float
        matrix_ttree, matrix_ref = self.make_example("F", "I")
        self.assertEqual(matrix_ttree.dtype, np.float32)

        # long, int -> long
        matrix_ttree, matrix_ref = self.make_example("L", "I")
        self.assertEqual(matrix_ttree.dtype, np.int64)

        # unsigned long, int -> long
        matrix_ttree, matrix_ref = self.make_example("l", "I")
        self.assertEqual(matrix_ttree.dtype, np.int64)

        # unsigned int, unsigned long -> unsigned long
        matrix_ttree, matrix_ref = self.make_example("i", "l")
        self.assertEqual(matrix_ttree.dtype, np.uint64)

        # unsigned int, unsigned long, float -> float
        matrix_ttree, matrix_ref = self.make_example("i", "l", "F")
        self.assertEqual(matrix_ttree.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
