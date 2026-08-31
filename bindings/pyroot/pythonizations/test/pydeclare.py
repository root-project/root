"""Tests for declaring C++ functions from Python callables.

The bulk of this file is the suite that used to test the numba based
implementation of ROOT.Numba.Declare, unchanged: the decorator now translates
the callable to C++ source instead of compiling it with numba, and the point of
keeping the assertions as they were is that this is not supposed to be visible
from the outside.

PyDeclareNamespace at the end covers what is new -- the ROOT.Py.Declare
spelling, the constructs the translation supports that numba never did, and the
requirement that anything outside the supported subset is refused loudly.
"""

import gc
import importlib.util
import os
import shutil
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import ROOT
from ROOT._pydeclare import PyDeclareError

default_test_inputs = [-1.0, 0.0, 100.0]


class NumbaDeclareSimple(unittest.TestCase):
    """
    Test decorator to create C++ wrapper for Python callables with fundamental types
    """

    test_inputs = default_test_inputs

    # Test refcounts
    def test_refcount_decorator(self):
        """
        Test refcount of decorator

        In case of Python < 3.14, we expect a refcount of 2, because the call
        to sys.getrefcount can create an additional reference itself. Starting
        from Python 3.14, we expect a refcount of 1 because there were changes
        to the interpreter to avoid some unnecessary ref counts. See also:
        https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-refcount
        """
        x = ROOT.Numba.Declare(["float"], "float")
        gc.collect()
        extra_ref_count = int(sys.version_info < (3, 14))
        self.assertEqual(sys.getrefcount(x), 1 + extra_ref_count)

    # Test optional name
    def test_optional_name(self):
        """
        Test optional name of wrapper function
        """
        optname = "optname2"

        @ROOT.Numba.Declare(["float"], "float", name=optname)
        def f(x):
            return x

        self.assertTrue(hasattr(ROOT.Numba, optname))

    # Test attributes
    def test_additional_attributes(self):
        """
        Test additional attributes
        """

        @ROOT.Numba.Declare(["float"], "float")
        def fn1(x):
            return x

        gc.collect()

        self.assertTrue(hasattr(fn1, "__cpp_wrapper__"))
        self.assertTrue(isinstance(fn1.__cpp_wrapper__, str))
        self.assertLessEqual(sys.getrefcount(fn1.__cpp_wrapper__), 3)

        self.assertEqual(fn1.__pydeclare_cpp_name__, "Numba::fn1")

    # Test cling integration
    def test_cling(self):
        """
        Test function call in cling
        """

        @ROOT.Numba.Declare(["float"], "float")
        def fn12(x):
            return 2.0 * x

        ROOT.gInterpreter.ProcessLine("y12 = Numba::fn12(42.0);")
        self.assertEqual(fn12(42.0), ROOT.y12)

    # Test RDataFrame integration
    def test_rdataframe(self):
        """
        Test function call as part of RDataFrame
        """

        @ROOT.Numba.Declare(["unsigned int"], "float")
        def fn13(x):
            return 2.0 * x

        df = ROOT.RDataFrame(4).Define("x", "rdfentry_").Define("y", "Numba::fn13(x)")
        mean_x = df.Mean("x")
        mean_y = df.Mean("y")
        self.assertEqual(mean_x.GetValue(), 1.5)
        self.assertEqual(mean_y.GetValue(), 3.0)

    def test_rdataframe_temporary(self):
        """
        Test passing a temporary from an RDataFrame operation
        """

        @ROOT.Numba.Declare(["const RVecF"], "RVecF")
        def pass_temporary(v):
            return v * np.array([2.0]).astype(np.float32)

        df = ROOT.RDataFrame(1).Define("v", "ROOT::RVecF{0.f,2.f}").Define("v2", "Numba::pass_temporary(v[v > 0])")

        rvecf = df.Take["RVecF"]("v2").GetValue()[0]

        self.assertTrue(np.array_equal(rvecf, np.array([4.0])))

    def test_rdataframe_std_vector(self):
        """
        Test function call as part of RDataFrame
        """

        @ROOT.Numba.Declare(["std::vector<int>"], "std::vector<int>")
        def square_vec(x):
            return x * x

        df = ROOT.RDataFrame(4).Define("x", "std::vector{1, 2, 3}").Define("x_sq", "Numba::square_vec(x)")
        df.Display().Print()
        self.assertEqual(df.Sum("x").GetValue(), 24)
        self.assertEqual(df.Sum("x_sq").GetValue(), 56)

    def test_rdataframe_std_array(self):
        """
        Test function call as part of RDataFrame with std::array
        """

        @ROOT.Numba.Declare(["std::array<int, 3>"], "std::array<int, 3>")
        def square_array(x):
            return x * x

        df = ROOT.RDataFrame(4).Define("x", "std::array{1, 2, 3}").Define("x_sq", "Numba::square_array(x)")
        df.Display().Print()
        self.assertEqual(df.Sum("x").GetValue(), 24)
        self.assertEqual(df.Sum("x_sq").GetValue(), 56)

    def test_rdataframe_LorentzVector(self):
        """
        Test function call as part of RDataFrame with ROOT::Math::LorentzVector
        """

        @ROOT.Numba.Declare(["ROOT::Math::PtEtaPhiMVector"], "float")
        def get_m(v):
            return v.M()

        df = ROOT.RDataFrame(4).Define("v", "ROOT::Math::PtEtaPhiMVector(1, 2, 3, 4)").Define("v_m", "Numba::get_m(v)")

        self.assertEqual(df.Sum("v_m").GetValue(), 16.0)

    # Test wrappings
    def test_wrapper_in_void(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare([], "float")
        def fn2n():
            return float(42)

        x1 = fn2n()
        x2 = ROOT.Numba.fn2n()
        self.assertEqual(x1, x2)
        self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_f(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "float")
        def fn2(x):
            return float(x)

        for v in self.test_inputs:
            x1 = fn2(v)
            x2 = ROOT.Numba.fn2(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_d(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "double")
        def fn2d(x):
            return float(x)

        for v in self.test_inputs:
            x1 = fn2d(v)
            x2 = ROOT.Numba.fn2d(v)
            self.assertEqual(x1, x2)
            # NOTE: There is no double in Python because everything is a double.
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_i(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "int")
        def fn3(x):
            return int(x)

        for v in self.test_inputs:
            x1 = fn3(v)
            x2 = ROOT.Numba.fn3(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_l(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "long")
        def fn4(x):
            return x

        for v in self.test_inputs:
            x1 = fn4(v)
            x2 = ROOT.Numba.fn4(v)
            self.assertEqual(x1, x2)
            self.assertEqual(int, type(x2))

    def test_wrapper_out_u(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "unsigned int")
        def fn5(x):
            return abs(x)

        for v in self.test_inputs:
            x1 = fn5(v)
            x2 = ROOT.Numba.fn5(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), int)

    def test_wrapper_out_k(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "unsigned long")
        def fn6(x):
            return abs(x)

        for v in self.test_inputs:
            x1 = fn6(v)
            x2 = ROOT.Numba.fn6(v)
            self.assertEqual(x1, x2)
            self.assertEqual(int, type(x2))

    def test_wrapper_out_b(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["float"], "bool")
        def fn6b(x):
            return x > 0

        for v in self.test_inputs:
            x1 = fn6b(v)
            x2 = ROOT.Numba.fn6b(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_in_b(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["bool"], "bool")
        def fn6b2(x):
            return not x

        for v in [True, False]:
            x1 = fn6b2(v)
            x2 = ROOT.Numba.fn6b2(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_in_i(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["int"], "float")
        def fn7i(x):
            return 2.0 * x

        for v in [-1, 0, 1, 999]:
            x1 = fn7i(v)
            x2 = ROOT.Numba.fn7i(v)
            self.assertEqual(x1, x2)

    def test_wrapper_in_l(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["long"], "float")
        def fn7l(x):
            return 2.0 * x

        for v in [-1, 0, 1, 999]:
            x1 = fn7l(v)
            x2 = ROOT.Numba.fn7l(v)
            self.assertEqual(x1, x2)

    def test_wrapper_in_ui(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["unsigned int"], "float")
        def fn7ui(x):
            return 2.0 * x

        for v in [0, 1, 999]:
            x1 = fn7ui(v)
            x2 = ROOT.Numba.fn7ui(v)
            self.assertEqual(x1, x2)

    def test_wrapper_in_ul(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["unsigned long"], "float")
        def fn7ul(x):
            return 2.0 * x

        for v in [0, 1, 999]:
            x1 = fn7ul(v)
            x2 = ROOT.Numba.fn7ul(v)
            self.assertEqual(x1, x2)


class NumbaDeclareArray(unittest.TestCase):
    """
    Test decorator to create C++ wrapper for Python callables with RVecs
    """

    test_inputs = [default_test_inputs]

    # The global module index does not have RVec entities preloaded and
    # gInterpreter.Declare is not allowed to load libROOTVecOps for RVec.
    # Preload the library now.
    ROOT.gSystem.Load("libROOTVecOps")

    def test_wrapper_in_vecf(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<float>"], "float")
        def g1(x):
            return x.sum()

        for v in self.test_inputs:
            x1 = g1(np.array(v, dtype=np.float32))
            x2 = ROOT.Numba.g1(ROOT.VecOps.RVec("float")(v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), float)

    def test_wrapper_in_vecf_vecd(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<float>", "RVec<double>"], "float")
        def g1_2vec(x, y):
            return x.sum() + y.sum()

        for v in self.test_inputs:
            x1 = g1_2vec(np.array(v, dtype=np.float32), np.array(v, dtype=np.float64))
            x2 = ROOT.Numba.g1_2vec(ROOT.VecOps.RVec("float")(v), ROOT.VecOps.RVec("double")(v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), float)

    def test_wrapper_in_vecd(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<double>"], "float")
        def g1d(x):
            return x.sum()

        for v in self.test_inputs:
            x1 = g1d(np.array(v, dtype=np.float64))
            x2 = ROOT.Numba.g1d(ROOT.VecOps.RVec("double")(v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), float)

    def test_wrapper_in_veci(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<int>"], "int")
        def g1i(x):
            return x.sum()

        for v in self.test_inputs:
            x1 = g1i(np.array(v, dtype=np.int32))
            x2 = ROOT.Numba.g1i(ROOT.VecOps.RVec("int")(int(x) for x in v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), int)

    def test_wrapper_in_vecl(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<long>"], "int")
        def g1l(x):
            return x.sum()

        for v in self.test_inputs:
            x1 = g1l(np.array(v, dtype=np.int64))
            x2 = ROOT.Numba.g1l(ROOT.VecOps.RVec("long")(int(x) for x in v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), int)

    def test_wrapper_in_vecui(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<unsigned int>"], "int")
        def g1ui(x):
            return x.sum()

        for v in [[0, 1, 999]]:
            x1 = g1ui(np.array(v, dtype=np.uint32))
            x2 = ROOT.Numba.g1ui(ROOT.VecOps.RVec("unsigned int")(v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), int)

    def test_wrapper_in_vecul(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<unsigned long>"], "int")
        def g1ul(x):
            return x.sum()

        for v in [[0, 1, 999]]:
            x1 = g1ul(np.array(v, dtype=np.uint64))
            x2 = ROOT.Numba.g1ul(ROOT.VecOps.RVec("unsigned long")(v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), int)

    def test_wrapper_in_vecb(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<bool>"], "int")
        def g1b(x):
            return x.sum()

        for v in [[True, False, True]]:
            x1 = g1b(np.array(v, dtype=np.float32))
            x2 = ROOT.Numba.g1b(ROOT.VecOps.RVec("bool")(v))
            self.assertEqual(x1, x2)
            self.assertEqual(type(x2), int)

    def test_wrapper_out_vecf(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<float>"], "RVec<float>")
        def g2f(x):
            return x[::-1]

        for v in [[0, 1, 999]]:
            x1 = g2f(np.array(v, dtype=np.float32))
            x2 = ROOT.Numba.g2f(ROOT.VecOps.RVec("float")(v))
            self.assertTrue((x1 == x2).all())

    def test_wrapper_out_vecd(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<double>"], "RVec<double>")
        def g2d(x):
            return x[::-1]

        for v in [[0, 1, 999]]:
            x1 = g2d(np.array(v, dtype=np.float64))
            x2 = ROOT.Numba.g2d(ROOT.VecOps.RVec("double")(v))
            self.assertTrue((x1 == x2).all())

    def test_wrapper_out_veci(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<int>"], "RVec<int>")
        def g2i(x):
            return x[::-1]

        for v in [[0, 1, 999]]:
            x1 = g2i(np.array(v, dtype=np.int32))
            x2 = ROOT.Numba.g2i(ROOT.VecOps.RVec("int")(v))
            self.assertTrue((x1 == x2).all())

    def test_wrapper_out_vecl(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<long>"], "RVec<long>")
        def g2l(x):
            return x[::-1]

        for v in [[0, 1, 999]]:
            x1 = g2l(np.array(v, dtype=np.int64))
            x2 = ROOT.Numba.g2l(ROOT.VecOps.RVec("long")(v))
            self.assertTrue((x1 == x2).all())

    def test_wrapper_out_vecul(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<unsigned long>"], "RVec<unsigned long>")
        def g2ul(x):
            return x[::-1]

        for v in [[0, 1, 999]]:
            x1 = g2ul(np.array(v, dtype=np.uint64))
            x2 = ROOT.Numba.g2ul(ROOT.VecOps.RVec("unsigned long")(v))
            self.assertTrue((x1 == x2).all())

    def test_wrapper_out_vecui(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<unsigned int>"], "RVec<unsigned int>")
        def g2ui(x):
            return x[::-1]

        for v in [[0, 1, 999]]:
            x1 = g2ui(np.array(v, dtype=np.uint32))
            x2 = ROOT.Numba.g2ui(ROOT.VecOps.RVec("unsigned int")(v))
            self.assertTrue((x1 == x2).all())

    def test_wrapper_out_vecb(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<bool>"], "RVec<bool>")
        def g2b(x):
            return x[::-1]

        for v in [[True, False]]:
            x1 = g2b(np.array(v, dtype=bool))
            x2 = ROOT.Numba.g2b(ROOT.VecOps.RVec("bool")(v))
            self.assertEqual(x1[0], bool(x2[0]))
            self.assertEqual(x1[1], bool(x2[1]))

    def test_wrapper_in_vecfb_out_vecf(self):
        """
        Test wrapper with different input/output configurations
        """

        @ROOT.Numba.Declare(["RVec<float>", "RVec<bool>"], "RVec<bool>")
        def g2fb(x, y):
            return (x > 1) | y

        for vf, vb in [[[1.0, 2.0], [True, False]]]:
            x1 = g2fb(np.array(vf, dtype=np.float32), np.array(vb, dtype=bool))
            x2 = ROOT.Numba.g2fb(ROOT.VecOps.RVec("float")(vf), ROOT.VecOps.RVec("bool")(vb))
            self.assertEqual(x1[0], bool(x2[0]))
            self.assertEqual(x1[1], bool(x2[1]))

    def test_const_modifier(self):
        """
        Test const modifier in input argument type
        """

        @ROOT.Numba.Declare(["const ROOT::VecOps::RVec<float>"], "RVecF")
        def const_mod(v):
            return v * np.array([1.0, 2.0]).astype(np.float32)

        rvecf = ROOT.Numba.const_mod(ROOT.RVecF([1.0, 2.0]))

        self.assertTrue(np.array_equal(rvecf, np.array([1.0, 4.0])))

    def test_reference(self):
        """
        Test passing a reference as input argument
        """

        @ROOT.Numba.Declare(["RVec<float>&"], "RVecF")
        def pass_reference(v):
            return v * np.array([1.0, 2.0]).astype(np.float32)

        rvecf = ROOT.Numba.pass_reference(ROOT.RVecF([1.0, 2.0]))

        self.assertTrue(np.array_equal(rvecf, np.array([1.0, 4.0])))


class NumbaDeclareInferred(unittest.TestCase):
    """
    Test decorator created with a reconstructed list of arguments using RDF column types,
    and a return type inferred from the translated function.
    """

    def test_fund_types(self):
        """
        Test fundamental types
        """
        df = ROOT.RDataFrame(4).Define("x", "rdfentry_")

        with self.subTest("function"):

            def is_even(x):
                return x % 2 == 0

            df = df.Define("is_even_x_1", is_even, ["x"])
            results = df.Take["bool"]("is_even_x_1").GetValue()
            self.assertEqual(list(results), [True, False, True, False])

        with self.subTest("lambda"):
            df = df.Define("is_even_x_2", lambda x: x % 2 == 0, ["x"])
            results = df.Take["bool"]("is_even_x_2").GetValue()
            self.assertEqual(list(results), [True, False, True, False])

    def test_rvec(self):
        """
        Test RVec
        """
        df = ROOT.RDataFrame(4).Define("x", "ROOT::VecOps::RVec<int>({1, 2, 3})")

        with self.subTest("function"):

            def square_rvec(v):
                return v * v

            df = df.Define("square_rvec_1", square_rvec, ["x"])
            results = df.Take["RVec<int>"]("square_rvec_1").GetValue()[0]
            self.assertTrue(np.array_equal(results, np.array([1, 4, 9])))

        with self.subTest("lambda"):
            df = df.Define("square_rvec_2", lambda v: v * v, ["x"])
            results = df.Take["RVec<int>"]("square_rvec_2").GetValue()[0]
            self.assertTrue(np.array_equal(results, np.array([1, 4, 9])))

    def test_std_vec(self):
        """
        Test std::vector
        """
        df = ROOT.RDataFrame(4).Define("x", "std::vector<int>({1, 2, 3})")

        with self.subTest("function"):

            def square_std_vec(v):
                return v * v

            df = df.Define("square_std_vec_1", square_std_vec, ["x"])
            results = df.Take["RVec<int>"]("square_std_vec_1").GetValue()[0]
            self.assertTrue(np.array_equal(results, np.array([1, 4, 9])))

        with self.subTest("lambda"):
            df = df.Define("square_std_vec_2", lambda v: v * v, ["x"])
            results = df.Take["RVec<int>"]("square_std_vec_2").GetValue()[0]
            self.assertTrue(np.array_equal(results, np.array([1, 4, 9])))

    def test_std_array(self):
        """
        Test std::array
        """
        df = ROOT.RDataFrame(4).Define("x", "std::array<int, 3>({1, 2, 3})")

        with self.subTest("function"):

            def square_std_arr(v):
                return v * v

            df = df.Define("square_std_arr_1", square_std_arr, ["x"])
            results = df.Take["RVec<int>"]("square_std_arr_1").GetValue()[0]
            self.assertTrue(np.array_equal(results, np.array([1, 4, 9])))

        with self.subTest("lambda"):
            df = df.Define("square_std_arr_2", lambda v: v * v, ["x"])
            results = df.Take["RVec<int>"]("square_std_arr_2").GetValue()[0]
            self.assertTrue(np.array_equal(results, np.array([1, 4, 9])))

    def test_missing_signature(self):
        """
        A method call on a C++ object, with no return type given.

        The return type does not have to be inferred on the Python side: the
        call is emitted and the C++ compiler deduces the type.
        """

        def f(x):
            return x.M()

        df = ROOT.RDataFrame(4).Define("v", "ROOT::Math::PtEtaPhiMVector(1, 2, 3, 4)")
        masses = df.Define("m", f, ["v"]).Take["double"]("m").GetValue()
        self.assertAlmostEqual(masses[0], 4.0)


class PyDeclareNamespace(unittest.TestCase):
    """The ROOT.Py.Declare spelling, and what the translation adds."""

    def test_py_namespace(self):
        """
        The generated function lands in the Py namespace
        """

        @ROOT.Py.Declare(["float"], "float")
        def py_double(x):
            return 2.0 * x

        self.assertEqual(ROOT.Py.py_double(21.0), 42.0)
        self.assertEqual(py_double.__pydeclare_cpp_name__, "Py::py_double")
        self.assertIn("namespace Py", py_double.__cpp_wrapper__)

    def test_control_flow(self):
        """
        if/elif/else, for over an array, while, and early returns
        """

        @ROOT.Py.Declare(["double"], "double")
        def clamped(x):
            if x < 0.0:
                return 0.0
            elif x > 10.0:
                return 10.0
            return x

        self.assertEqual([ROOT.Py.clamped(v) for v in (-5.0, 5.0, 50.0)], [0.0, 5.0, 10.0])

        @ROOT.Py.Declare(["RVecD", "double"], "long")
        def count_above(v, thr):
            n = 0
            for x in v:
                if x > thr:
                    n = n + 1
            return n

        self.assertEqual(ROOT.Py.count_above(ROOT.RVecD([1.0, 5.0, 9.0]), 4.0), 2)

        @ROOT.Py.Declare(["long"], "double")
        def harmonic(n):
            total = 0.0
            for i in range(1, n + 1):
                total += 1.0 / i
            return total

        expected = 0.0
        for i in range(1, 5):
            expected += 1.0 / i
        self.assertAlmostEqual(ROOT.Py.harmonic(4), expected, places=15)

        @ROOT.Py.Declare(["RVecD"], "double")
        def first_negative_or_sum(v):
            i = 0
            while i < v.size:
                if v[i] < 0.0:
                    return v[i]
                i += 1
            return v.sum()

        self.assertEqual(ROOT.Py.first_negative_or_sum(ROOT.RVecD([1.0, -2.0, 3.0])), -2.0)
        self.assertEqual(ROOT.Py.first_negative_or_sum(ROOT.RVecD([1.0, 2.0, 3.0])), 6.0)

    def test_python_semantics(self):
        """
        The operators where Python and C++ disagree keep Python's meaning
        """

        @ROOT.Py.Declare(["int", "int"], "double")
        def true_division(a, b):
            return a / b

        @ROOT.Py.Declare(["int", "int"], "int")
        def floor_division(a, b):
            return a // b

        @ROOT.Py.Declare(["int", "int"], "int")
        def modulo(a, b):
            return a % b

        @ROOT.Py.Declare(["RVecD"], "double")
        def last(v):
            return v[-1]

        self.assertEqual(ROOT.Py.true_division(7, 2), 7 / 2)
        self.assertEqual(ROOT.Py.floor_division(-7, 2), -7 // 2)
        self.assertEqual(ROOT.Py.modulo(-7, 2), -7 % 2)
        self.assertEqual(ROOT.Py.last(ROOT.RVecD([1.0, 2.0, 3.0])), 3.0)

    def test_cpp_interoperability(self):
        """
        Translated code can name anything cling knows, and other declarations
        """

        @ROOT.Py.Declare(["double"], "double")
        def scaled(x):
            return 2.5 * x

        @ROOT.Py.Declare(["double"], "double")
        def combined(x):
            return ROOT.TMath.Abs(scaled(x)) + ROOT.TMath.Sqrt(4.0)

        self.assertEqual(ROOT.Py.combined(-2.0), 7.0)
        self.assertIn("TMath::Abs", combined.__cpp_wrapper__)
        self.assertIn("Py::scaled", combined.__cpp_wrapper__)

    def test_constants_are_frozen(self):
        """
        Values from the enclosing scope are inlined when the function is declared
        """
        scale = 0.5

        @ROOT.Py.Declare(["double"], "double")
        def uses_closure(x):
            return x * scale

        self.assertEqual(ROOT.Py.uses_closure(4.0), 2.0)
        self.assertIn("0.5", uses_closure.__cpp_wrapper__)
        self.assertNotIn("scale", uses_closure.__cpp_wrapper__)

        # The value is read when the function is declared, not cached per name:
        # a second function sees the rebound value.
        scale = 100.0

        @ROOT.Py.Declare(["double"], "double")
        def uses_closure_again(x):
            return x * scale

        self.assertEqual(ROOT.Py.uses_closure_again(4.0), 400.0)
        self.assertEqual(ROOT.Py.uses_closure(4.0), 2.0)

    def test_generated_code_is_available(self):
        """
        The C++ can be read back, which is the debugging story
        """

        @ROOT.Py.Declare(["std::vector<int>"], "std::vector<int>")
        def squared_vector(x):
            return x * x

        code = squared_vector.__cpp_wrapper__
        self.assertIn("std::vector<int> squared_vector(const std::vector<int> &x)", code)
        # std::vector is viewed as an RVec so that the operators are available
        self.assertIn("AsRVec", code)


class PyDeclareRefusals(unittest.TestCase):
    """Anything outside the supported subset has to be refused, not approximated.

    Each case below would otherwise be translated to C++ that compiles and
    quietly computes something else, which is the failure mode this feature
    cannot afford.
    """

    counter = 0

    def load(self, source):
        """Write a callable to a real file, so that its source can be read back.

        The translation reads the Python source in order to quote the offending
        line, so the callables under test cannot come from exec().
        """
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        PyDeclareRefusals.counter += 1
        name = "pydeclare_refusal_{}".format(PyDeclareRefusals.counter)
        path = os.path.join(directory, name + ".py")
        with open(path, "w") as out:
            out.write(textwrap.dedent(source))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.f, path

    def assertRefused(self, source, input_types, return_type, expected):
        """Declare the callable and require a clear refusal mentioning `expected`."""
        func, _ = self.load(source)
        with self.assertRaises(PyDeclareError) as caught:
            ROOT.Py.Declare(input_types, return_type, name="refused")(func)
        self.assertIn(expected, str(caught.exception))

    def test_unsupported_statements(self):
        self.assertRefused(
            "def f(x):\n    try:\n        return x\n    except Exception:\n        return 0.0\n",
            ["double"],
            "double",
            "'Try' is not supported",
        )

    def test_comprehension(self):
        self.assertRefused(
            "def f(v):\n    return sum([x * x for x in v])\n",
            ["RVecD"],
            "double",
            "List comprehensions are not supported",
        )

    def test_string_value(self):
        self.assertRefused(
            "def f(x):\n    s = 'hello'\n    return x\n",
            ["double"],
            "double",
            "String values are not supported",
        )

    def test_truth_of_an_array(self):
        self.assertRefused(
            "def f(v):\n    if v:\n        return 1.0\n    return 0.0\n",
            ["RVecD"],
            "double",
            "truth value of an array is ambiguous",
        )

    def test_variable_changing_type(self):
        self.assertRefused(
            "def f(v):\n    a = v.sum()\n    a = v\n    return 1.0\n",
            ["RVecD"],
            "double",
            "changes type",
        )

    def test_arbitrary_python_module(self):
        self.assertRefused(
            "import os\n\n\ndef f(x):\n    return os.getpid() * x\n",
            ["double"],
            "double",
            "cannot be used in translated code",
        )

    def test_unknown_cpp_name(self):
        self.assertRefused(
            "import ROOT\n\n\ndef f(x):\n    return ROOT.TMath.NoSuchFunction(x)\n",
            ["double"],
            "double",
            "has no member 'NoSuchFunction'",
        )

    def test_error_points_at_the_source_line(self):
        """The message has to say where, not just what."""
        func, path = self.load("def f(x):\n    d = {1: 2}\n    return x\n")
        with self.assertRaises(PyDeclareError) as caught:
            ROOT.Py.Declare(["double"], "double", name="refused_with_location")(func)
        message = str(caught.exception)
        self.assertIn(path, message)
        self.assertIn("line 2", message)
        self.assertIn("d = {1: 2}", message)
        # ... and name the construct it actually found.
        self.assertIn("Dict literals", message)

    def test_random_numbers(self):
        """The most important refusal: a frozen draw would be silently wrong.

        np.random.normal() does not depend on the arguments, so constant
        folding would happily evaluate it once and inline the result, giving
        every entry in the dataset the same "random" number.
        """
        self.assertRefused(
            "import numpy as np\n\n\ndef f(x):\n    return np.random.normal() * x\n",
            ["double"],
            "double",
            "'np.random' is not available",
        )
        self.assertRefused(
            "import numpy as np\n\n\ndef f(x):\n    return x + np.random.rand()\n",
            ["double"],
            "double",
            "'np.random' is not available",
        )

    def test_reading_a_file_at_declaration_time(self):
        self.assertRefused(
            "import numpy as np\n\n\ndef f(x):\n    return x + float(np.load('weights.npy'))\n",
            ["double"],
            "double",
            "'np.load' is not available",
        )

    def test_sets_and_tuples(self):
        self.assertRefused(
            "def f(x):\n    s = {1, 2}\n    return x\n", ["double"], "double", "Set literals"
        )
        self.assertRefused(
            "def f(x):\n    t = (x, x)\n    return t[0]\n", ["double"], "double", "Tuple literals"
        )

    def test_lambda_in_the_body(self):
        self.assertRefused(
            "def f(x):\n    g = lambda y: y * 2\n    return g(x)\n",
            ["double"],
            "double",
            "Nested lambdas are not supported",
        )

    def test_with_and_global(self):
        self.assertRefused(
            "def f(x):\n    with open('f') as fh:\n        return x\n",
            ["double"],
            "double",
            "'With' is not supported",
        )
        self.assertRefused(
            "def f(x):\n    global g\n    return x\n", ["double"], "double", "'Global' is not supported"
        )

    def test_f_strings_and_generators(self):
        self.assertRefused(
            "def f(x):\n    return float(len(f'{x}'))\n", ["double"], "double", "f-strings are not supported"
        )
        self.assertRefused(
            "def f(v):\n    return sum(x for x in v)\n", ["RVecD"], "double", "comprehensions are not supported"
        )

    def test_star_args(self):
        self.assertRefused(
            "def f(*args):\n    return args[0]\n", ["double"], "double", "*args"
        )

    def test_multi_dimensional_indexing(self):
        self.assertRefused(
            "def f(v):\n    return v[0, 1]\n", ["RVecD"], "double", "Multi-dimensional indexing"
        )

    def test_signature_mismatch(self):
        self.assertRefused(
            "def f(x, y):\n    return x + y\n", ["double"], "double", "takes 2 argument"
        )

    def test_mutating_an_array_that_python_would_alias(self):
        """'w = v' aliases in Python and copies here, so writing through it
        would mean two different things."""
        self.assertRefused(
            "def f(v):\n    w = v\n    w[0] = 100.0\n    return v[0]\n",
            ["RVecD"],
            "double",
            "Cannot assign to an element of 'w'",
        )
        self.assertRefused(
            "def f(v):\n    w = v[1:]\n    w[0] = 100.0\n    return v[1]\n",
            ["RVecD"],
            "double",
            "Cannot assign to an element of 'w'",
        )

    def test_reading_a_variable_that_may_be_unbound(self):
        """Python raises UnboundLocalError; a zero would look like a result."""
        self.assertRefused(
            "def f(x):\n    if x > 0:\n        y = 1.5\n    return y\n",
            ["double"],
            "double",
            "only assigned inside a nested block",
        )

    def test_conditional_expression_with_an_array_condition(self):
        """numpy raises on this too; np.where is the element-wise form."""
        self.assertRefused(
            "def f(v):\n    return v if v > 0 else -v\n",
            ["RVecD"],
            "ROOT::RVec<double>",
            "The truth value of an array is ambiguous",
        )

    def test_non_integer_slice_bounds(self):
        self.assertRefused(
            "def f(v, x):\n    return v[x:]\n",
            ["RVecD", "double"],
            "ROOT::RVec<double>",
            "Slice indices must be integers",
        )

    def test_assigning_to_an_array_parameter(self):
        self.assertRefused(
            "import numpy as np\n\n\ndef f(v):\n    v = np.array([1.0])\n    return v[0]\n",
            ["RVecD"],
            "double",
            "Cannot assign to the array parameter 'v'",
        )


class PyDeclareHasNoNumbaDependency(unittest.TestCase):
    """The translation is the point: nothing here may pull in numba."""

    def test_numba_is_not_imported(self):
        for module in ("numba", "llvmlite", "cffi"):
            self.assertNotIn(module, sys.modules, "{} was imported by ROOT.Py.Declare".format(module))


class PyDeclareAgreesWithPython(unittest.TestCase):
    """The translated function has to give what the plain Python one gives.

    Every case here computes its reference by calling the undecorated callable,
    so the assertion is against Python itself rather than against a constant
    that was read off this implementation.
    """

    counter = 0

    def check(self, func, input_types, return_type, cases, places=12):
        """Declare *func* and compare it with the Python original on *cases*."""
        PyDeclareAgreesWithPython.counter += 1
        name = "agree_{}".format(PyDeclareAgreesWithPython.counter)
        ROOT.Py.Declare(input_types, return_type, name=name)(func)
        translated = getattr(ROOT.Py, name)
        for args in cases:
            with self.subTest(func=func.__name__, args=args):
                expected = func(*args)
                got = translated(*[_as_cpp(a) for a in args])
                if isinstance(expected, np.ndarray):
                    self.assertEqual(list(got), list(expected))
                elif isinstance(expected, (float, np.floating)):
                    self.assertAlmostEqual(got, float(expected), places=places)
                else:
                    self.assertEqual(got, expected)
                    # An integer result has to stay integral: Python's '2 ** 10'
                    # is an int, and a numpy integer is one too.
                    self.assertIsInstance(got, (int, bool))

    # -- the documented Python/C++ divergences ------------------------------

    def test_true_division(self):
        def truediv(a, b):
            return a / b

        self.check(truediv, ["int", "int"], "double", [(7, 2), (-7, 2), (1, 8)])

    def test_floor_division(self):
        def floordiv(a, b):
            return a // b

        self.check(floordiv, ["int", "int"], "long", [(7, 2), (-7, 2), (7, -2), (-7, -2)])

    def test_modulo_takes_the_sign_of_the_divisor(self):
        def mod(a, b):
            return a % b

        self.check(mod, ["int", "int"], "long", [(7, 3), (-7, 3), (7, -3), (-7, -3)])

    def test_power_of_two_integers_stays_integral(self):
        def ipow(a, b):
            return a**b

        self.check(ipow, ["int", "int"], "long", [(2, 10), (3, 0), (-2, 3), (10, 6)])

    def test_power_accumulates_in_a_wide_integer(self):
        """A short base must not overflow at 2**15, as Python's int does not."""

        def wideipow(x):
            return x**3

        self.check(wideipow, ["short"], "long", [(100,), (2,), (-100,)])

    def test_power_of_floats(self):
        def fpow(a, b):
            return a**b

        self.check(fpow, ["double", "double"], "double", [(2.0, 0.5), (9.0, -1.0)])

    def test_round_is_half_to_even(self):
        def myround(x):
            return round(x)

        self.check(myround, ["double"], "long", [(0.5,), (1.5,), (2.5,), (-0.5,), (-1.5,), (3.7,)])

    def test_negative_index(self):
        def last(v):
            return v[-1]

        self.check(last, ["RVecD"], "double", [(np.array([1.0, 2.0, 3.0]),)])

    def test_sum_of_booleans_is_an_integer(self):
        def countpos(v):
            return (v > 0).sum()

        self.check(countpos, ["RVecD"], "long", [(np.array([1.0, -2.0, 3.0, -4.0]),)])

    def test_sum_of_narrow_integers_does_not_overflow(self):
        """numpy accumulates into a wide integer; the element type would wrap."""

        def total(v):
            return v.sum()

        self.check(total, ["ROOT::RVec<signed char>"], "long", [(np.array([3] * 100, dtype=np.int8),)])

    # -- slicing ------------------------------------------------------------

    def test_slices(self):
        cases = [(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),)]
        for i, body in enumerate(
            [
                "v[1:4]",
                "v[-2:]",
                "v[:3]",
                "v[::2]",
                "v[1::2]",
                "v[10:20]",
                "v[4:1:-1]",
                "v[::-1]",
                "v[-100:100]",
                "v[3:1]",
            ]
        ):
            namespace = {"np": np}
            exec("def sliced(v):\n    return {}\n".format(body), namespace)  # noqa: S102
            func = namespace["sliced"]
            func.__name__ = "sliced_{}".format(i)
            # exec'd source is unreadable, so go through a real file instead.
            with self.subTest(slice=body):
                self.check(_reload_from_file(func, body), ["RVecD"], "ROOT::RVec<double>", cases)

    # -- the parts of the subset that had no test at all --------------------

    def test_builtins(self):
        def usebuiltins(v, x):
            return float(len(v)) + max(x, 2.0) + min(x, 2.0) + abs(x) + float(sum(v)) + pow(x, 2.0)

        self.check(usebuiltins, ["RVecD", "double"], "double", [(np.array([1.0, 2.0]), -3.0)])

    def test_reductions(self):
        def reductions(v):
            return v.mean() + v.std() + v.min() + v.max() + float(v.argmin() + v.argmax()) + v.prod()

        self.check(reductions, ["RVecD"], "double", [(np.array([3.0, 1.0, 2.0]),)])

    def test_any_and_all(self):
        def anyall(v):
            return int((v > 0).any()) * 10 + int((v > 0).all())

        self.check(anyall, ["RVecD"], "long", [(np.array([1.0, -1.0]),), (np.array([1.0, 2.0]),)])

    def test_np_where(self):
        def clip(v):
            return np.where(v > 0, v, 0.0)

        self.check(clip, ["RVecD"], "ROOT::RVec<double>", [(np.array([1.0, -2.0, 3.0]),)])

    def test_elementwise_maths_and_constants(self):
        def maths(x):
            return np.sqrt(abs(x)) + np.sin(x) + np.log(abs(x) + 1.0) + np.pi

        self.check(maths, ["double"], "double", [(2.0,), (-3.0,)])

    def test_chained_comparison(self):
        def between(x):
            return 0.0 < x < 10.0

        self.check(between, ["double"], "bool", [(5.0,), (-1.0,), (20.0,)])

    def test_and_or_evaluate_to_an_operand(self):
        """Python's 'or' gives back a value, not a bool: '0.0 or 5.0' is 5.0."""

        def orelse(x, y):
            return x or y

        self.check(orelse, ["double", "double"], "double", [(0.0, 5.0), (3.0, 5.0)])

        def andalso(x, y):
            return x and y

        self.check(andalso, ["double", "double"], "double", [(0.0, 5.0), (3.0, 5.0)])

    def test_conditional_expression(self):
        def sign(x):
            return 1.0 if x > 0 else -1.0

        self.check(sign, ["double"], "double", [(2.0,), (-2.0,)])

    def test_bitwise_and_shifts(self):
        def bits(a, b):
            return (a & b) + (a | b) + (a ^ b) + (a << 2) + (a >> 1) + ~a

        self.check(bits, ["int", "int"], "long", [(12, 10), (1, 0)])

    def test_break_continue_and_assert(self):
        def loop(v):
            total = 0.0
            for x in v:
                if x < 0:
                    continue
                if x > 100.0:
                    break
                assert x == x
                total += x
            return total

        self.check(loop, ["RVecD"], "double", [(np.array([1.0, -2.0, 3.0, 200.0, 5.0]),)])

    def test_annotated_assignment(self):
        def annotated(v):
            total: "double" = 0.0  # noqa: F821 -- a C++ type, not a Python name
            for x in v:
                total += x
            return total

        self.check(annotated, ["RVecD"], "double", [(np.array([1.0, 2.0, 3.0]),)])

    def test_mask_indexing(self):
        def positives(v):
            return v[v > 0].sum()

        self.check(positives, ["RVecD"], "double", [(np.array([1.0, -2.0, 3.0]),)])

    def test_loop_variable_survives_the_loop(self):
        """Python's for does not scope its variable to the loop body."""

        def leaks(n):
            i = 0
            if n > 0:
                for i in range(n):
                    pass
            return i

        self.check(leaks, ["int"], "long", [(5,), (0,)])

    def test_unary_minus_does_not_fuse(self):
        """'-(-x)' must not come out as the decrement operator."""

        def negneg(x):
            return -(-x) + +(+x)

        self.check(negneg, ["long"], "long", [(5,), (-3,)])

    def test_parameter_can_be_reassigned(self):
        def absval(x):
            if x < 0:
                x = -x
            return x

        self.check(absval, ["double"], "double", [(-3.0,), (3.0,)])

    def test_element_assignment_on_a_locally_built_array(self):
        def build(x):
            a = np.array([x, x + 1.0])
            a[0] = 99.0
            return a

        self.check(build, ["double"], "ROOT::RVec<double>", [(1.0,)])


class PyDeclareMixedSignedness(unittest.TestCase):
    """Python has no unsigned integers, so C++'s conversions are wrong for us.

    ``nMuon`` and friends are unsigned columns, so this is reachable from
    ordinary RDataFrame code rather than being a corner case.
    """

    def check(self, name, func, input_types, return_type, args, expected):
        ROOT.Py.Declare(input_types, return_type, name=name)(func)
        self.assertEqual(getattr(ROOT.Py, name)(*args), expected)

    def test_modulo(self):
        def mixmod(a, b):
            return a % b

        self.check("mix_mod", mixmod, ["int", "unsigned int"], "long", (-7, 3), -7 % 3)

    def test_floor_division(self):
        def mixfdiv(a, b):
            return a // b

        self.check("mix_fdiv", mixfdiv, ["int", "unsigned int"], "long", (-7, 3), -7 // 3)

    def test_unary_minus_of_an_unsigned_value(self):
        def negu(x):
            return -x

        self.check("mix_neg", negu, ["unsigned int"], "long", (3,), -3)

    def test_comparison(self):
        def gt(a, b):
            return a > b

        self.check("mix_gt", gt, ["int", "unsigned int"], "bool", (-1, 2), -1 > 2)

    def test_max(self):
        def mymax(a, b):
            return max(a, b)

        self.check("mix_max", mymax, ["int", "unsigned int"], "long", (-1, 2), max(-1, 2))

    def test_min(self):
        def mymin(a, b):
            return min(a, b)

        self.check("mix_min", mymin, ["int", "unsigned int"], "long", (-1, 2), min(-1, 2))

    def test_abs_of_an_unsigned_value(self):
        """std::abs has no unsigned overload; the call would be ambiguous."""

        def myabs(x):
            return abs(x)

        self.check("mix_abs", myabs, ["unsigned int"], "unsigned int", (3,), 3)


class PyDeclareRaisesInsteadOfMisbehaving(unittest.TestCase):
    """Where Python raises, the translation must raise too.

    Every case here used to either abort the process or read out of bounds,
    which in an event loop means a crash or a silently wrong number.
    """

    def declare(self, name, func, input_types, return_type):
        ROOT.Py.Declare(input_types, return_type, name=name)(func)
        return getattr(ROOT.Py, name)

    def test_integer_division_by_zero(self):
        def fdiv(a, b):
            return a // b

        with self.assertRaises(Exception):
            self.declare("raise_fdiv", fdiv, ["int", "int"], "long")(7, 0)

    def test_modulo_by_zero(self):
        def mod(a, b):
            return a % b

        with self.assertRaises(Exception):
            self.declare("raise_mod", mod, ["int", "int"], "long")(7, 0)

    def test_true_division_of_integers_by_zero(self):
        def div(a, b):
            return a / b

        with self.assertRaises(Exception):
            self.declare("raise_div", div, ["int", "int"], "double")(7, 0)

    def test_index_past_the_end(self):
        def at(v, i):
            return v[i]

        fn = self.declare("raise_at", at, ["RVecD", "int"], "double")
        with self.assertRaises(Exception):
            fn(ROOT.VecOps.RVec("double")([1.0, 2.0, 3.0]), 100)
        with self.assertRaises(Exception):
            fn(ROOT.VecOps.RVec("double")([1.0, 2.0, 3.0]), -100)

    def test_index_past_the_end_with_an_unsigned_index(self):
        def atu(v, i):
            return v[i]

        fn = self.declare("raise_atu", atu, ["RVecD", "unsigned int"], "double")
        with self.assertRaises(Exception):
            fn(ROOT.VecOps.RVec("double")([1.0, 2.0, 3.0]), 100)

    def test_index_with_a_constant_past_the_end(self):
        def at5(v):
            return v[5]

        fn = self.declare("raise_at5", at5, ["RVecD"], "double")
        with self.assertRaises(Exception):
            fn(ROOT.VecOps.RVec("double")([1.0, 2.0, 3.0]))

    def test_where_with_branches_shorter_than_the_condition(self):
        def wh(a, b):
            return np.where(a > 0, b, 0.0)

        fn = self.declare("raise_where", wh, ["RVecD", "RVecD"], "ROOT::RVec<double>")
        with self.assertRaises(Exception):
            fn(ROOT.VecOps.RVec("double")([1.0] * 4), ROOT.VecOps.RVec("double")([5.0, 6.0]))


class PyDeclareTypeSpellings(unittest.TestCase):
    """Every spelling of an RVec that ROOT accepts elsewhere has to work here."""

    def test_rvec_alias_spellings(self):
        spellings = [
            "RVecF",
            "ROOT::RVecF",
            "ROOT::VecOps::RVecF",
            "RVec<float>",
            "ROOT::RVec<float>",
            "ROOT::VecOps::RVec<float>",
        ]
        for i, spelling in enumerate(spellings):
            with self.subTest(spelling=spelling):

                def total(v):
                    return v.sum()

                name = "spelling_{}".format(i)
                ROOT.Py.Declare([spelling], "double", name=name)(total)
                self.assertAlmostEqual(
                    getattr(ROOT.Py, name)(ROOT.VecOps.RVec("float")([1.0, 2.0, 3.0])), 6.0
                )


def _as_cpp(value):
    """numpy arrays go in as RVecs; everything else is passed straight through."""
    if isinstance(value, np.ndarray):
        return ROOT.VecOps.RVec(_CPP_OF_DTYPE[value.dtype.name])(value)
    return value


_CPP_OF_DTYPE = {
    "float64": "double",
    "float32": "float",
    "int8": "signed char",
    "int32": "int",
    "int64": "long long",
    "bool": "bool",
}


def _reload_from_file(func, body):
    """Re-create *func* in a real file, so that its source can be read."""
    directory = tempfile.mkdtemp()
    module_name = "pydeclare_slice_{}".format(func.__name__)
    path = os.path.join(directory, module_name + ".py")
    with open(path, "w") as out:
        out.write("import numpy as np\n\n\ndef sliced(v):\n    return {}\n".format(body))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.sliced.__name__ = func.__name__
    return module.sliced


if __name__ == "__main__":
    unittest.main()
