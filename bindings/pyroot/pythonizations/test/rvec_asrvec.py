import gc
import sys
import unittest

import numpy as np
import ROOT


def get_maximum_for_dtype(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).max


def get_minimum_for_dtype(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).min


class AsRVec(unittest.TestCase):
    """
    Tests for the AsRVec feature enabling to adopt memory of Python objects
    with an array interface member using RVec as C++ container.
    """

    # Helpers
    dtypes = ["int16", "int32", "int64", "uint16", "uint32", "uint64", "float32", "float64", "bool"]

    def check_memory_adoption(self, root_obj, np_obj):
        np_obj[0] = get_maximum_for_dtype(np_obj.dtype)
        np_obj[1] = get_minimum_for_dtype(np_obj.dtype)
        self.assertEqual(root_obj[0], np_obj[0])
        self.assertEqual(root_obj[1], np_obj[1])

    def check_size(self, expected_size, obj):
        self.assertEqual(expected_size, obj.size())

    # Tests
    def test_dtypes(self):
        """
        Test adoption of numpy arrays with different data types
        """
        for dtype in self.dtypes:
            np_obj = np.empty(2, dtype=dtype)
            root_obj = ROOT.VecOps.AsRVec(np_obj)
            self.check_memory_adoption(root_obj, np_obj)
            self.check_size(2, root_obj)

    def test_object_dtype_strings(self):
        """
        Test adoption of numpy arrays with dtype=object and dtype=str containing strings
        """
        for dtype in [object, np.str_, "<U13"]:
            with self.subTest(dtype=dtype):
                np_obj = np.array(["test_string_1", "test_string_2"], dtype=dtype)
                root_obj = ROOT.VecOps.AsRVec(np_obj)

                self.check_size(2, root_obj)
                self.assertEqual(root_obj[0], np_obj[0])
                self.assertEqual(root_obj[1], np_obj[1])

    def test_object_dtype_mixed_types(self):
        """
        Test that a TypeError is raised for numpy arrays with dtype=object
        containing elements of different types
        """
        np_obj = np.array(["string", {}], dtype=object)

        with self.assertRaises(TypeError) as context:
            ROOT.VecOps.AsRVec(np_obj)

        self.assertIn("All elements in the numpy array must be of the same type", str(context.exception))

    def test_multidim(self):
        """
        Test adoption of multi-dimensional numpy arrays
        """
        np_obj = np.array([[1, 2], [3, 4]], dtype="float32")
        rvec = ROOT.VecOps.AsRVec(np_obj)
        self.assertEqual(rvec.size(), 4)

    def test_size_zero(self):
        """
        Test adoption of numpy array with size 0
        """
        np_obj = np.array([], dtype="float32")
        rvec = ROOT.VecOps.AsRVec(np_obj)
        self.assertEqual(rvec.size(), 0)

    def test_adopt_rvec(self):
        """
        Test adoption of RVecs
        """
        rvec = ROOT.VecOps.RVec("float")(1)
        rvec[0] = 42
        rvec2 = ROOT.VecOps.AsRVec(rvec)
        self.assertEqual(rvec.size(), rvec2.size())
        self.assertEqual(rvec[0], rvec2[0])
        rvec2[0] = 43
        self.assertEqual(rvec[0], rvec2[0])

    def test_ownership(self):
        """
        Test ownership of returned RVec (to be owned by Python)
        """
        np_obj = np.array([1, 2])
        rvec = ROOT.VecOps.AsRVec(np_obj)
        self.assertEqual(rvec.__python_owns__, True)

    def test_attribute_adopted(self):
        """
        Test __adopted__ attribute of returned RVecs
        """
        np_obj = np.array([1, 2])
        rvec = ROOT.VecOps.AsRVec(np_obj)
        self.assertTrue(hasattr(rvec, "__adopted__"))
        self.assertEqual(id(rvec.__adopted__), id(np_obj))

    def test_refcount(self):
        """
        Test reference count of returned RVec

        In case of Python <=3.14, we expect a refcount of 2 for the RVec
        because the call to sys.getrefcount creates a second reference by
        itself. Starting from Python 3.14, we expect a refcount of 1 because
        there were changes to the interpreter to avoid some unnecessary ref
        counts. See also:
        https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-refcount
        We attach the adopted pyobject to the RVec and increase the refcount of the
        numpy array. After deletion of the rvec, the refcount of the numpy array
        is decreased.
        """
        extra_ref_count = int(sys.version_info < (3, 14))
        np_obj = np.array([1, 2])
        rvec = ROOT.VecOps.AsRVec(np_obj)
        gc.collect()
        self.assertEqual(sys.getrefcount(rvec), 1 + extra_ref_count)
        self.assertEqual(sys.getrefcount(np_obj), 2 + extra_ref_count)
        del rvec
        gc.collect()
        self.assertEqual(sys.getrefcount(np_obj), 1 + extra_ref_count)


if __name__ == "__main__":
    unittest.main()
