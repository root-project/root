import unittest
import ROOT
import sys

# long does not exist anymore on Python 3, map it to int
if sys.version_info[0] > 2:
    long = int


default_test_inputs = [-1.0, 0.0, 100.0]


class NumbaDeclare(unittest.TestCase):
    """
    Test decorator to create C++ wrapper for Python callables using numba
    """

    test_inputs = default_test_inputs

    # Test refcounts
    def test_refcount_decorator(self):
        """
        Test refcount of decorator
        """
        x = ROOT.Numba.Declare(["float"], "float")
        self.assertEqual(sys.getrefcount(x), 2)

    def test_refcount_pycallable(self):
        """
        Test refcount of decorated callable
        """
        def f1(x):
            return x
        def f2(x):
            return x
        fn0 = ROOT.Numba.Declare(["float"], "float")(f1)
        import numba
        ref = numba.cfunc("float32(float32)", nopython=True)(f2)
        # ROOT holds an additional reference compared to plain numba
        self.assertEqual(sys.getrefcount(f1), sys.getrefcount(f2) + 1)

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
    def test_cpp_wrapper_code(self):
        """
        Test C++ wrapper code attribute
        """
        @ROOT.Numba.Declare(["float"], "float")
        def fn1(x):
            return x
        self.assertTrue(hasattr(fn1, "__cpp_wrapper__"))
        self.assertTrue(type(fn1.__cpp_wrapper__) == str)
        self.assertEqual(sys.getrefcount(fn1.__cpp_wrapper__), 2)

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
            self.assertEqual(long, type(x2))

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
            # NOTE: cppyy does not return an Python int for unsigned int but a long.
            # This could be fixed but as well should not have any impact.
            self.assertEqual(type(x2), long)

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
            self.assertEqual(long, type(x2))

    def test_wrapper_out_b(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.Numba.Declare(["float"], "bool")
        def fn6b(x):
            return x>0
        for v in self.test_inputs:
            x1 = fn6b(v)
            x2 = ROOT.Numba.fn6b(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))


if __name__ == '__main__':
    unittest.main()
