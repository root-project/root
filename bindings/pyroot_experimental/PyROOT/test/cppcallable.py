import unittest
import ROOT
import sys


default_test_inputs = [-1.0, 0.0, 100.0]


class GenericCppCallable(unittest.TestCase):
    """
    Test decorator to create generic C++ wrapper for Python callables
    """

    test_inputs = default_test_inputs

    # Test refcounts
    def test_refcount_decorator(self):
        """
        Test refcount of decorator
        """
        x = ROOT.DeclareCppCallable([""], "", generic_only=True)
        self.assertEqual(sys.getrefcount(x), 2)

    def test_refcount_pycallable(self):
        """
        Test refcount of decorated callable
        """
        @ROOT.DeclareCppCallable([""], "", generic_only=True)
        def f0():
            pass
        self.assertEqual(sys.getrefcount(f0), 2)

    # Test attributes
    def test_cpp_wrapper_code(self):
        """
        Test C++ wrapper code attribute
        """
        @ROOT.DeclareCppCallable([""], "", generic_only=True)
        def f1():
            pass
        self.assertTrue(hasattr(f1, "__cpp_wrapper__"))
        self.assertTrue(type(f1.__cpp_wrapper__) == str)
        self.assertEqual(sys.getrefcount(f1.__cpp_wrapper__), 2)

   # Test optional name
    def test_optional_name(self):
        """
        Test optional name of wrapper function
        """
        optname = "optname1"
        @ROOT.DeclareCppCallable([""], "", name=optname, generic_only=True)
        def f():
            pass
        self.assertTrue(hasattr(ROOT.CppCallable, optname))

    # Test wrappings
    def test_wrapper_out_f(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "float", generic_only=True)
        def f2(x):
            return float(x)
        for v in self.test_inputs:
            x1 = f2(v)
            x2 = ROOT.CppCallable.f2(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_d(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "double", generic_only=True)
        def f2d(x):
            return float(x)
        for v in self.test_inputs:
            x1 = f2d(v)
            x2 = ROOT.CppCallable.f2d(v)
            self.assertEqual(x1, x2)
            # NOTE: There is no double in Python because everything is a double.
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_i(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "int", generic_only=True)
        def f3(x):
            return int(x)
        for v in self.test_inputs:
            x1 = f3(v)
            x2 = ROOT.CppCallable.f3(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_l(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "long", generic_only=True)
        def f4(x):
            return long(x)
        for v in self.test_inputs:
            x1 = f4(v)
            x2 = ROOT.CppCallable.f4(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_u(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "unsigned int", generic_only=True)
        def f5(x):
            return int(abs(x))
        for v in self.test_inputs:
            x1 = f5(v)
            x2 = ROOT.CppCallable.f5(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), int)
            # NOTE: cppyy does not return an Python int for unsigned int but a long.
            # This could be fixed but as well should not have any impact.
            self.assertEqual(type(x2), long)

    def test_wrapper_out_k(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "unsigned long", generic_only=True)
        def f6(x):
            return long(abs(x))
        for v in self.test_inputs:
            x1 = f6(v)
            x2 = ROOT.CppCallable.f6(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_out_b(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "bool", generic_only=True)
        def f6b(x):
            return bool(x>0)
        for v in self.test_inputs:
            x1 = f6b(v)
            x2 = ROOT.CppCallable.f6b(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    def test_wrapper_inout_f(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "float", generic_only=True)
        def f7(x):
            return float(x)
        for v in self.test_inputs:
            x1 = f7(v)
            x2 = ROOT.CppCallable.f7(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_d(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "float", generic_only=True)
        def f7d(x):
            return float(x)
        for v in self.test_inputs:
            x1 = f7d(v)
            x2 = ROOT.CppCallable.f7d(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_i(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["int"], "int", generic_only=True)
        def f8(x):
            return int(x)
        for v in self.test_inputs:
            v = int(v)
            x1 = f8(v)
            x2 = ROOT.CppCallable.f8(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_l(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["long"], "long", generic_only=True)
        def f9(x):
            return long(x)
        for v in self.test_inputs:
            v = long(v)
            x1 = f9(v)
            x2 = ROOT.CppCallable.f9(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_u(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["unsigned int"], "unsigned int", generic_only=True)
        def f10(x):
            return int(abs(x))
        for v in self.test_inputs:
            v = int(abs(v))
            x1 = f10(v)
            x2 = ROOT.CppCallable.f10(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_k(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["unsigned long"], "unsigned long", generic_only=True)
        def f11(x):
            return long(abs(x))
        for v in self.test_inputs:
            v = long(abs(v))
            x1 = f11(v)
            x2 = ROOT.CppCallable.f11(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_u(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["unsigned int"], "unsigned int", generic_only=True)
        def f10(x):
            return int(abs(x))
        for v in self.test_inputs:
            v = int(abs(v))
            x1 = f10(v)
            x2 = ROOT.CppCallable.f10(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_k(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["unsigned long"], "unsigned long", generic_only=True)
        def f11(x):
            return long(abs(x))
        for v in self.test_inputs:
            v = long(abs(v))
            x1 = f11(v)
            x2 = ROOT.CppCallable.f11(v)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_b(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["bool"], "bool", generic_only=True)
        def f11b(x):
            return bool(not x)
        for v in [True, False]:
            x1 = f11b(v)
            x2 = ROOT.CppCallable.f11b(v)
            self.assertEqual(x1, x2)
            self.assertEqual(not v, x2)

    # Test wrapper with STL vectors
    def test_wrapper_in_vec(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["vector<float>"], "unsigned int", generic_only=True)
        def f14(x):
            return x.size()
        for v in self.test_inputs:
            v = int(abs(v))
            w = ROOT.std.vector["float"](v)
            x1 = f14(w)
            x2 = ROOT.CppCallable.f14(w)
            self.assertEqual(x1, x2)
            self.assertEqual(v, x2)

    def test_wrapper_inout_vec(self):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["vector<float>"], "vector<float>", generic_only=True)
        def f15(x):
            for i in range(x.size()):
                x[i] *= 2.0
            return x
        for v in self.test_inputs:
            v = int(abs(v))
            w = ROOT.std.vector["float"](v)
            w.resize(v)
            for i in range(v):
                w[i] = i
            x1 = f15(w)
            x2 = ROOT.CppCallable.f15(w)
            self.assertEqual(x1, x2)

    # Test custom types
    def test_wrapper_in_customtype(self):
        """
        Test wrapper with custom types
        """
        ROOT.gInterpreter.Declare("""
        struct Foo {
            static const int foo = 42;
        };
        """)
        @ROOT.DeclareCppCallable(["Foo"], "int", generic_only=True)
        def f16(x):
            return x.foo
        x = ROOT.Foo()
        y = ROOT.CppCallable.f16(x)
        self.assertEqual(y, 42)

    def test_wrapper_out_customtype(self):
        """
        Test wrapper with custom types
        """
        ROOT.gInterpreter.Declare("""
        struct Foo2 {
            static const int foo = 42;
        };
        """)
        @ROOT.DeclareCppCallable([""], "Foo2", generic_only=True)
        def f17():
            return ROOT.Foo2()
        y = ROOT.CppCallable.f17()
        self.assertEqual(y.foo, 42)

    # Test cling integration
    def test_cling(self):
        """
        Test function call in cling
        """
        @ROOT.DeclareCppCallable(["float"], "float", generic_only=True)
        def f12(x):
            return 2.0 * x
        ROOT.gInterpreter.ProcessLine("y12 = CppCallable::f12(42.0);")
        self.assertEqual(f12(42.0), ROOT.y12)

    # Test RDataFrame integration
    def test_rdataframe(self):
        """
        Test function call as part of RDataFrame
        """
        @ROOT.DeclareCppCallable(["unsigned int"], "float", generic_only=True)
        def f13(x):
            return 2.0 * x
        df = ROOT.ROOT.RDataFrame(4).Define("x", "rdfentry_").Define("y", "CppCallable::f13(x)")
        mean_x = df.Mean("x")
        mean_y = df.Mean("y")
        self.assertEqual(mean_x.GetValue(), 1.5)
        self.assertEqual(mean_y.GetValue(), 3.0)


# Decorator which skips tests if numba is not found
def needs_numba(f):
    try:
        import numba
    except:
        return unittest.skip("Numba not found.")(f)
    return f


class NumbaCppCallable(unittest.TestCase):
    """
    Test decorator to create C++ wrapper for Python callables using numba
    """

    test_inputs = default_test_inputs

    # Test refcounts
    @needs_numba
    def test_refcount_decorator(self):
        """
        Test refcount of decorator
        """
        x = ROOT.DeclareCppCallable([""], "", numba_only=True)
        self.assertEqual(sys.getrefcount(x), 2)

    @needs_numba
    def test_refcount_pycallable(self, numba_only=True):
        """
        Test refcount of decorated callable
        """
        def f1(x):
            return x
        def f2(x):
            return x
        fn0 = ROOT.DeclareCppCallable(["float"], "float")(f1)
        import numba
        ref = numba.cfunc("float32(float32)", nopython=True)(f2)
        # ROOT holds an additional reference compared to plain numba
        self.assertEqual(sys.getrefcount(f1), sys.getrefcount(f2) + 1)

    # Test optional name
    @needs_numba
    def test_optional_name(self, numba_only=True):
        """
        Test optional name of wrapper function
        """
        optname = "optname2"
        @ROOT.DeclareCppCallable([""], "", name=optname)
        def f():
            pass
        self.assertTrue(hasattr(ROOT.CppCallable, optname))

    # Test attributes
    @needs_numba
    def test_cpp_wrapper_code(self, numba_only=True):
        """
        Test C++ wrapper code attribute
        """
        @ROOT.DeclareCppCallable([""], "")
        def fn1():
            pass
        self.assertTrue(hasattr(fn1, "__cpp_wrapper__"))
        self.assertTrue(type(fn1.__cpp_wrapper__) == str)
        self.assertEqual(sys.getrefcount(fn1.__cpp_wrapper__), 2)

    # Test cling integration
    @needs_numba
    def test_cling(self, numba_only=True):
        """
        Test function call in cling
        """
        @ROOT.DeclareCppCallable(["float"], "float")
        def fn12(x):
            return 2.0 * x
        ROOT.gInterpreter.ProcessLine("y12 = CppCallable::fn12(42.0);")
        self.assertEqual(fn12(42.0), ROOT.y12)

    # Test RDataFrame integration
    @needs_numba
    def test_rdataframe(self, numba_only=True):
        """
        Test function call as part of RDataFrame
        """
        @ROOT.DeclareCppCallable(["unsigned int"], "float")
        def fn13(x):
            return 2.0 * x
        df = ROOT.ROOT.RDataFrame(4).Define("x", "rdfentry_").Define("y", "CppCallable::fn13(x)")
        mean_x = df.Mean("x")
        mean_y = df.Mean("y")
        self.assertEqual(mean_x.GetValue(), 1.5)
        self.assertEqual(mean_y.GetValue(), 3.0)

    # Test wrappings
    @needs_numba
    def test_wrapper_out_f(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "float")
        def fn2(x):
            return float(x)
        for v in self.test_inputs:
            x1 = fn2(v)
            x2 = ROOT.CppCallable.fn2(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    @needs_numba
    def test_wrapper_out_d(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "double")
        def fn2d(x):
            return float(x)
        for v in self.test_inputs:
            x1 = fn2d(v)
            x2 = ROOT.CppCallable.fn2d(v)
            self.assertEqual(x1, x2)
            # NOTE: There is no double in Python because everything is a double.
            self.assertEqual(type(x1), type(x2))

    @needs_numba
    def test_wrapper_out_i(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "int")
        def fn3(x):
            return int(x)
        for v in self.test_inputs:
            x1 = fn3(v)
            x2 = ROOT.CppCallable.fn3(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))

    @needs_numba
    def test_wrapper_out_l(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "long")
        def fn4(x):
            return x
        for v in self.test_inputs:
            x1 = fn4(v)
            x2 = ROOT.CppCallable.fn4(v)
            self.assertEqual(x1, x2)
            self.assertEqual(long, type(x2))

    @needs_numba
    def test_wrapper_out_u(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "unsigned int")
        def fn5(x):
            return abs(x)
        for v in self.test_inputs:
            x1 = fn5(v)
            x2 = ROOT.CppCallable.fn5(v)
            self.assertEqual(x1, x2)
            # NOTE: cppyy does not return an Python int for unsigned int but a long.
            # This could be fixed but as well should not have any impact.
            self.assertEqual(type(x2), long)

    @needs_numba
    def test_wrapper_out_k(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "unsigned long")
        def fn6(x):
            return abs(x)
        for v in self.test_inputs:
            x1 = fn6(v)
            x2 = ROOT.CppCallable.fn6(v)
            self.assertEqual(x1, x2)
            self.assertEqual(long, type(x2))

    @needs_numba
    def test_wrapper_out_b(self, numba_only=True):
        """
        Test wrapper with different input/output configurations
        """
        @ROOT.DeclareCppCallable(["float"], "bool")
        def fn6b(x):
            return x>0
        for v in self.test_inputs:
            x1 = fn6b(v)
            x2 = ROOT.CppCallable.fn6b(v)
            self.assertEqual(x1, x2)
            self.assertEqual(type(x1), type(x2))


class ProxyCppCallable(unittest.TestCase):
    """
    Test decorator to create C++ wrapper for Python callables using either numba
    or the generic implementation
    """

    test_inputs = default_test_inputs

    # Test refcounts
    def test_refcount_decorator(self):
        """
        Test refcount of decorator
        """
        x = ROOT.DeclareCppCallable([""], "")
        self.assertEqual(sys.getrefcount(x), 2)

    # Test optional name
    def test_optional_name(self):
        """
        Test optional name of wrapper function
        """
        optname = "optname3"
        @ROOT.DeclareCppCallable([""], "", name=optname)
        def f():
            pass
        self.assertTrue(hasattr(ROOT.CppCallable, optname))

    # Test switching between numba and generic impl
    @needs_numba
    def test_use_numba(self):
        """
        Test switch to numba impl
        """
        @ROOT.DeclareCppCallable(["float"], "float")
        def fp0(x):
            return x
        self.assertIn("auto funcptr = reinterpret_cast<float(*)(float)>",
                fp0.__cpp_wrapper__)

    def test_use_generic(self):
        """
        Test switch to generic impl
        """
        @ROOT.DeclareCppCallable(["float"], "vector<float>", verbose=False)
        def fp1(x):
            y = ROOT.std.vector("float")(1)
            y[0] = x
            return y
        self.assertIn("auto pyresult = PyObject_CallFunction(pyfunc",
                fp1.__cpp_wrapper__)


if __name__ == '__main__':
    unittest.main()
