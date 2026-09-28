# Author: ROOT Team / PyROOT modernization
#
################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import sys
import unittest
import ROOT


class BufferProtocolSTLVector(unittest.TestCase):
    """
    Test Python buffer protocol (memoryview, PEP 688, zero-copy mutations)
    on ROOT's std::vector containers.
    """

    test_types = [
        ("short", "h", 2, -100, 200),
        ("unsigned short", "H", 2, 10, 500),
        ("int", "i", 4, -42, 1337),
        ("unsigned int", "I", 4, 100, 99999),
        ("float", "f", 4, 3.140000104904175, 2.7182817459106445),
        ("double", "d", 8, 3.141592653589793, 2.718281828459045),
        ("Long64_t", "q", 8, -9876543210, 1234567890123),
    ]

    def _get_view(self, obj):
        if sys.version_info >= (3, 12):
            return memoryview(obj)
        return obj.to_memoryview()

    def test_std_vector_memoryview_creation(self):
        """Verify memoryview creation and attributes across fundamental types."""
        for dtype, expected_fmt, expected_itemsize, val1, val2 in self.test_types:
            vec = ROOT.std.vector(dtype)(4)
            vec[0] = val1
            vec[1] = val2
            vec[2] = val1
            vec[3] = val2

            mv = self._get_view(vec)
            self.assertEqual(len(mv), 4)
            self.assertEqual(mv.ndim, 1)
            self.assertEqual(mv.shape, (4,))
            self.assertEqual(mv.itemsize, expected_itemsize)
            self.assertEqual(mv.format, expected_fmt)

    def test_std_vector_zero_copy_mutations(self):
        """Ensure mutations through memoryview reflect in C++ vector and vice-versa."""
        vec = ROOT.std.vector("int")(3)
        vec[0] = 10
        vec[1] = 20
        vec[2] = 30

        mv = self._get_view(vec)
        # Mutate via Python buffer protocol
        mv[0] = 999
        self.assertEqual(vec[0], 999)

        # Mutate via C++ vector indexing
        vec[1] = 888
        self.assertEqual(mv[1], 888)

    def test_std_vector_slicing_and_strides(self):
        """Test slice subviews of std::vector buffer."""
        vec = ROOT.std.vector("double")(6)
        for i in range(6):
            vec[i] = float(i * 10)

        mv = self._get_view(vec)
        sliced = mv[2:5]
        self.assertEqual(len(sliced), 3)
        self.assertEqual(list(sliced), [20.0, 30.0, 40.0])

        # Mutate through slice
        sliced[1] = 999.0
        self.assertEqual(vec[3], 999.0)

    def test_std_vector_bytes_casting(self):
        """Verify memoryview can be cast to unsigned bytes ('B') for raw I/O."""
        vec = ROOT.std.vector("int")(2)
        vec[0] = 1
        vec[1] = 2

        mv = self._get_view(vec)
        byte_view = mv.cast("B")
        self.assertEqual(len(byte_view), 2 * 4)
        raw_bytes = bytes(byte_view)
        self.assertIsInstance(raw_bytes, bytes)
        self.assertEqual(len(raw_bytes), 8)

    def test_std_vector_empty(self):
        """Verify empty std::vector produces safe zero-length memoryview."""
        vec = ROOT.std.vector("int")()
        mv = self._get_view(vec)
        self.assertEqual(len(mv), 0)
        self.assertEqual(list(mv), [])

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 688 collections.abc.Buffer requires Python 3.12+")
    def test_std_vector_pep688_isinstance_buffer(self):
        """Verify PEP 688 isinstance check against collections.abc.Buffer."""
        import collections.abc

        vec = ROOT.std.vector("float")(5)
        self.assertTrue(isinstance(vec, collections.abc.Buffer))


class BufferProtocolRVec(unittest.TestCase):
    """
    Test Python buffer protocol on ROOT::VecOps::RVec.
    """

    def _get_view(self, obj):
        if sys.version_info >= (3, 12):
            return memoryview(obj)
        return obj.to_memoryview()

    def test_rvec_float_double_int(self):
        """Test RVec of float, double, and int."""
        for dtype, fmt in [("int", "i"), ("float", "f"), ("double", "d")]:
            rvec = ROOT.VecOps.RVec[dtype](3)
            rvec[0] = 1
            rvec[1] = 2
            rvec[2] = 3

            mv = self._get_view(rvec)
            self.assertEqual(len(mv), 3)
            self.assertEqual(mv.format, fmt)

            # Mutate via buffer
            mv[1] = 77
            self.assertEqual(rvec[1], 77)

    def test_rvec_empty(self):
        """Test empty RVec buffer creation."""
        rvec = ROOT.VecOps.RVec["double"]()
        mv = self._get_view(rvec)
        self.assertEqual(len(mv), 0)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 688 collections.abc.Buffer requires Python 3.12+")
    def test_rvec_pep688_isinstance_buffer(self):
        """Verify RVec satisfies collections.abc.Buffer in Python 3.12+."""
        import collections.abc

        rvec = ROOT.VecOps.RVec["int"](2)
        self.assertTrue(isinstance(rvec, collections.abc.Buffer))


class BufferProtocolTArray(unittest.TestCase):
    """
    Test Python buffer protocol on ROOT.TArray subclasses (TArrayI, TArrayF, TArrayD).
    """

    def _get_view(self, obj):
        if sys.version_info >= (3, 12):
            return memoryview(obj)
        return obj.to_memoryview()

    def test_tarrayi(self):
        arr = ROOT.TArrayI(4)
        arr[0] = 10
        arr[1] = 20
        arr[2] = 30
        arr[3] = 40

        mv = self._get_view(arr)
        self.assertEqual(len(mv), 4)
        self.assertEqual(mv.format, "i")
        self.assertEqual(mv.itemsize, 4)

        # Mutate buffer
        mv[2] = 999
        self.assertEqual(arr[2], 999)

    def test_tarrayf(self):
        arr = ROOT.TArrayF(3)
        arr[0] = 1.5
        arr[1] = 2.5
        arr[2] = 3.5

        mv = self._get_view(arr)
        self.assertEqual(len(mv), 3)
        self.assertEqual(mv.format, "f")

        mv[0] = 4.5
        self.assertAlmostEqual(arr[0], 4.5, places=5)

    def test_tarrayd(self):
        arr = ROOT.TArrayD(2)
        arr[0] = 123.456
        arr[1] = 789.012

        mv = self._get_view(arr)
        self.assertEqual(len(mv), 2)
        self.assertEqual(mv.format, "d")
        self.assertEqual(mv.itemsize, 8)

        mv[1] = 555.555
        self.assertEqual(arr[1], 555.555)

    def test_tarray_empty(self):
        arr = ROOT.TArrayD(0)
        mv = self._get_view(arr)
        self.assertEqual(len(mv), 0)

    @unittest.skipIf(sys.version_info < (3, 12), "PEP 688 collections.abc.Buffer requires Python 3.12+")
    def test_tarray_pep688_isinstance_buffer(self):
        """Verify TArray satisfies collections.abc.Buffer in Python 3.12+."""
        import collections.abc

        arr = ROOT.TArrayD(3)
        self.assertTrue(isinstance(arr, collections.abc.Buffer))


if __name__ == "__main__":
    unittest.main()
