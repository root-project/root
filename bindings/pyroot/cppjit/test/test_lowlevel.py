import sys

import py
from pytest import mark, raises, skip
from support import (
    IS_CLING,
    IS_MAC,
    IS_VALGRIND,
    IS_WINDOWS,
    ispypy,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/datatypesDict"))


def setup_module(mod):
    setup_make("datatypes")


class TestLOWLEVEL:
    def setup_class(cls):
        import cppjit

        cls.test_dct = test_dct
        cls.datatypes = cppjit.load_reflection_info(cls.test_dct)
        cls.N = cppjit.gbl.N

    def test00_import_all(self):
        """Validity of `from cppjit.ll import *`"""

        from cppjit import ll

        for attr in ll.__all__:
            assert hasattr(ll, attr)

    def test01_llv_type(self):
        """Existence of LowLevelView type"""

        import cppjit.types

        assert cppjit.types.LowLevelView

    def test02_builtin_cpp_casts(self):
        """C++ casting of builtin types"""

        from cppjit import ll

        for cast in (ll.cast, ll.static_cast):
            assert type(cast[float](1)) == float
            assert cast[float](1) == 1.0

            assert type(cast[int](1.1)) == int
            assert cast[int](1.1) == 1

        assert len(ll.reinterpret_cast["int*"](0)) == 0
        raises(ReferenceError, ll.reinterpret_cast["int*"](0).__getitem__, 0)

    def test03_memory(self):
        """Memory allocation and free-ing"""

        import cppjit
        from cppjit import ll

        # regular C malloc/free
        mem = cppjit.gbl.malloc(16)
        cppjit.gbl.free(mem)

        # typed styles
        mem = ll.malloc[int](self.N)
        assert len(mem) == self.N
        assert not mem.__cpp_array__
        for i in range(self.N):
            mem[i] = i + 1
            assert type(mem[i]) == int
            assert mem[i] == i + 1
        cppjit.ll.free(mem)

        # C++ arrays
        mem = cppjit.ll.array_new[int](self.N)
        assert mem.__cpp_array__
        assert len(mem) == self.N
        for i in range(self.N):
            mem[i] = i + 1
            assert type(mem[i]) == int
            assert mem[i] == i + 1
        cppjit.ll.array_delete(mem)

        mem = cppjit.ll.array_new[int](self.N, managed=True)
        assert mem.__python_owns__
        mem.__python_owns__ = False
        assert not mem.__python_owns__
        mem.__python_owns__ = True
        assert mem.__python_owns__

    def test04_python_casts(self):
        """Casts to common Python pointer encapsulations"""

        import cppjit
        import cppjit.ll

        cppjit.cppdef("""namespace pycasts {
        struct SomeObject{};
        uintptr_t get_address(SomeObject* ptr) { return (intptr_t)ptr; }
        uintptr_t get_deref(void* ptr) { return (uintptr_t)(*(void**)ptr); }
        }""")

        from cppjit.gbl import pycasts

        s = pycasts.SomeObject()
        actual = pycasts.get_address(s)

        assert cppjit.ll.addressof(s) == actual
        assert cppjit.ll.as_ctypes(s).value == actual

        ptrptr = cppjit.ll.as_ctypes(s, byref=True)
        assert pycasts.get_deref(ptrptr) == actual

    def test05_array_as_ref(self):
        """Use arrays for pass-by-ref"""

        from array import array

        import cppjit

        ctd = cppjit.gbl.CppjitTestData()

        # boolean type
        b = array("b", [0])
        ctd.set_bool_r(b)
        assert b[0] == True

        # char types (as data)
        c = array("B", [0])
        ctd.set_uchar_r(c)
        assert c[0] == ord("d")

        # integer types
        i = array("h", [0])
        ctd.set_short_r(i)
        assert i[0] == -1
        i = array("H", [0])
        ctd.set_ushort_r(i)
        assert i[0] == 2
        i = array("i", [0])
        ctd.set_int_r(i)
        assert i[0] == -3
        i = array("I", [0])
        ctd.set_uint_r(i)
        assert i[0] == 4
        i = array("l", [0])
        ctd.set_long_r(i)
        assert i[0] == -5
        i = array("L", [0])
        ctd.set_ulong_r(i)
        assert i[0] == 6
        if sys.hexversion >= 0x3000000:
            i = array("q", [0])
            ctd.set_llong_r(i)
            assert i[0] == -7
            i = array("Q", [0])
            ctd.set_ullong_r(i)
            assert i[0] == 8

            # floating point types
        f = array("f", [0])
        ctd.set_float_r(f)
        assert f[0] == 5.0
        f = array("d", [0])
        ctd.set_double_r(f)
        assert f[0] == -5.0

    @mark.xfail(
        condition=IS_VALGRIND or IS_CLING,
        run=False,
        reason="Valgrind detects memory leak with invalid delete[] operator, crashes on Cling",
    )
    def test06_ctypes_as_ref_and_ptr(self):
        """Use ctypes for pass-by-ref/ptr"""

        # See:
        #  https://docs.python.org/2/library/ctypes.html#fundamental-data-types
        #
        # ctypes type       C type                                      Python type
        # ------------------------------------------------------------------------------
        # c_bool            _Bool                                       bool (1)
        #
        # c_char            char 1-character                            string
        # c_wchar           wchar_t 1-character                         unicode string

        # c_byte            char                                        int
        # c_ubyte           unsigned char                               int
        #
        # c_int8            signed char                                 int
        # c_uint8           unsigned char                               int
        # c_short           short                                       int
        # c_ushort          unsigned short                              int
        # c_int             int                                         int
        # c_uint            unsigned int                                int/long
        # c_long            long                                        int/long
        # c_ulong           unsigned long                               int/long
        # c_longlong        __int64 or long long                        int/long
        # c_ulonglong       unsigned __int64 or unsigned long long      int/long
        #
        # c_float           float                                       float
        # c_double          double                                      float
        # c_longdouble      long double                                 float

        import ctypes

        import cppjit
        import cppjit.ll

        ctd = cppjit.gbl.CppjitTestData()

        ### pass by reference/pointer and set value back

        for e in ["_r", "_p"]:
            # boolean type
            b = ctypes.c_bool(False)
            getattr(ctd, "set_bool" + e)(b)
            assert b.value == True

            # char types
            if e == "_r":
                c = ctypes.c_char(b"\0")
                getattr(ctd, "set_char" + e)(c)
                assert c.value == b"a"
                c = ctypes.c_wchar("\0")
                getattr(ctd, "set_wchar" + e)(c)
                assert c.value == "b"
                c = ctypes.c_byte(0)
                getattr(ctd, "set_schar" + e)(c)
                assert c.value == ord("c")
            c = ctypes.c_ubyte(0)
            getattr(ctd, "set_uchar" + e)(c)
            assert c.value == ord("d")

            # integer types
            i = ctypes.c_short(0)
            getattr(ctd, "set_short" + e)(i)
            assert i.value == -1
            i = ctypes.c_ushort(0)
            getattr(ctd, "set_ushort" + e)(i)
            assert i.value == 2
            i = ctypes.c_int(0)
            getattr(ctd, "set_int" + e)(i)
            assert i.value == -3
            i = ctypes.c_uint(0)
            getattr(ctd, "set_uint" + e)(i)
            assert i.value == 4
            i = ctypes.c_long(0)
            getattr(ctd, "set_long" + e)(i)
            assert i.value == -5
            i = ctypes.c_ulong(0)
            getattr(ctd, "set_ulong" + e)(i)
            assert i.value == 6
            i = ctypes.c_longlong(0)
            getattr(ctd, "set_llong" + e)(i)
            assert i.value == -7
            i = ctypes.c_ulonglong(0)
            getattr(ctd, "set_ullong" + e)(i)
            assert i.value == 8

            # floating point types
            f = ctypes.c_float(0)
            getattr(ctd, "set_float" + e)(f)
            assert f.value == 5.0
            f = ctypes.c_double(0)
            getattr(ctd, "set_double" + e)(f)
            assert f.value == -5.0
            f = ctypes.c_longdouble(0)
            getattr(ctd, "set_ldouble" + e)(f)
            assert f.value == 10.0

            ### pass by pointer and set value back, now using byref (not recommended)

        cb = ctypes.byref

        # boolean type
        b = ctypes.c_bool(False)
        ctd.set_bool_p(cb(b))
        assert b.value == True

        # char types
        c = ctypes.c_ubyte(0)
        ctd.set_uchar_p(cb(c))
        assert c.value == ord("d")

        # integer types
        i = ctypes.c_int8(0)
        ctd.set_int8_p(cb(i))
        assert i.value == -27
        i = ctypes.c_uint8(0)
        ctd.set_uint8_p(cb(i))
        assert i.value == 28
        i = ctypes.c_short(0)
        ctd.set_short_p(cb(i))
        assert i.value == -1
        i = ctypes.c_ushort(0)
        ctd.set_ushort_p(cb(i))
        assert i.value == 2
        i = ctypes.c_int(0)
        ctd.set_int_p(cb(i))
        assert i.value == -3
        i = ctypes.c_uint(0)
        ctd.set_uint_p(cb(i))
        assert i.value == 4
        i = ctypes.c_long(0)
        ctd.set_long_p(cb(i))
        assert i.value == -5
        i = ctypes.c_ulong(0)
        ctd.set_ulong_p(cb(i))
        assert i.value == 6
        i = ctypes.c_longlong(0)
        ctd.set_llong_p(cb(i))
        assert i.value == -7
        i = ctypes.c_ulonglong(0)
        ctd.set_ullong_p(cb(i))
        assert i.value == 8

        # floating point types
        f = ctypes.c_float(0)
        ctd.set_float_p(cb(f))
        assert f.value == 5.0
        f = ctypes.c_double(0)
        ctd.set_double_p(cb(f))
        assert f.value == -5.0

        ### pass by ptr/ptr with allocation (ptr/ptr is ambiguous in it's task, so many
        # types are allowed to pass; this tests allocation into the pointer)

        from ctypes import POINTER

        import cppjit.ll

        # boolean type
        b = POINTER(ctypes.c_bool)()
        ctd.set_bool_ppa(b)
        assert b[0] == True
        assert b[1] == False
        assert b[2] == True
        cppjit.ll.array_delete(b)

        # char types
        c = POINTER(ctypes.c_ubyte)()
        ctd.set_uchar_ppa(c)
        assert c[0] == ord("k")
        assert c[1] == ord("l")
        assert c[2] == ord("m")
        cppjit.ll.array_delete(c)

        # integer types
        i = POINTER(ctypes.c_int8)()
        ctd.set_int8_ppa(i)
        assert i[0] == -27
        assert i[1] == -28
        assert i[2] == -29
        cppjit.ll.array_delete["void"](i)  # template resolves as signed char*
        i = POINTER(ctypes.c_uint8)()
        ctd.set_uint8_ppa(i)
        assert i[0] == 28
        assert i[1] == 29
        assert i[2] == 30
        cppjit.ll.array_delete["void"](i)  # template resolves as unsigned char*
        i = POINTER(ctypes.c_short)()
        ctd.set_short_ppa(i)
        assert i[0] == -1
        assert i[1] == -2
        assert i[2] == -3
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_ushort)()
        ctd.set_ushort_ppa(i)
        assert i[0] == 4
        assert i[1] == 5
        assert i[2] == 6
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_int)()
        ctd.set_int_ppa(i)
        assert i[0] == -7
        assert i[1] == -8
        assert i[2] == -9
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_uint)()
        ctd.set_uint_ppa(i)
        assert i[0] == 10
        assert i[1] == 11
        assert i[2] == 12
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_long)()
        ctd.set_long_ppa(i)
        assert i[0] == -13
        assert i[1] == -14
        assert i[2] == -15
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_ulong)()
        ctd.set_ulong_ppa(i)
        assert i[0] == 16
        assert i[1] == 17
        assert i[2] == 18
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_longlong)()
        ctd.set_llong_ppa(i)
        assert i[0] == -19
        assert i[1] == -20
        assert i[2] == -21
        cppjit.ll.array_delete(i)
        i = POINTER(ctypes.c_ulonglong)()
        ctd.set_ullong_ppa(i)
        assert i[0] == 22
        assert i[1] == 23
        assert i[2] == 24
        cppjit.ll.array_delete(i)

        # floating point types
        f = POINTER(ctypes.c_float)()
        ctd.set_float_ppa(f)
        assert f[0] == 5
        assert f[1] == 10
        assert f[2] == 20
        cppjit.ll.array_delete(f)
        f = POINTER(ctypes.c_double)()
        ctd.set_double_ppa(f)
        assert f[0] == -5
        assert f[1] == -10
        assert f[2] == -20
        cppjit.ll.array_delete(f)
        f = POINTER(ctypes.c_longdouble)()
        ctd.set_ldouble_ppa(f)
        assert f[0] == 5
        assert f[1] == 10
        assert f[2] == 20
        cppjit.ll.array_delete(f)

    def test07_ctypes_pointer_types(self):
        """Use ctypes for pass-by-ptr/ptr-ptr"""

        if ispypy:
            skip("memory corruption")

        # See:
        #  https://docs.python.org/2/library/ctypes.html#fundamental-data-types
        #
        # ctypes type       C type                                      Python type
        # ------------------------------------------------------------------------------
        # c_char_p          char* (NULL terminated)                     string or None
        # c_wchar_p         wchar_t* (NULL terminated)                  unicode or None
        # c_void_p          void*                                       int/long or None

        import ctypes

        import cppjit

        ctd = cppjit.gbl.CppjitTestData()

        ptr = ctypes.c_char_p()
        for meth in ["char", "cchar"]:
            val = getattr(ctd, "set_" + meth + "_ppm")(ptr)
            assert ctd.freeit(ptr) == val

        ptr = ctypes.c_wchar_p()
        for meth in ["wchar", "cwchar"]:
            val = getattr(ctd, "set_" + meth + "_ppm")(ptr)
            assert ctd.freeit(ptr) == val

        ptr = ctypes.c_void_p()
        val = ctd.set_void_ppm(ptr)
        assert ctd.freeit(ptr) == val

    def test08_ctypes_type_correctness(self):
        """If types don't match with ctypes, expect exceptions"""

        import ctypes

        import cppjit

        ctd = cppjit.gbl.CppjitTestData()

        meth_types = ["bool", "double"]
        if not IS_WINDOWS:
            meth_types.append("long")

        i = ctypes.c_int(0)
        for ext in ["_r", "_p"]:
            for meth in meth_types:
                with raises(TypeError):
                    getattr(ctd, "set_" + meth + ext)(i)

    def test09_numpy_bool_array(self):
        """Test passing of numpy bool array"""

        import cppjit

        try:
            import numpy as np
        except ImportError:
            skip("numpy is not installed")

        cppjit.cppdef("int convert_bool(bool* x) {return *x;}")

        x = np.array([True], dtype=bool)
        assert cppjit.gbl.convert_bool(x)

    @mark.xfail(condition=IS_MAC, run=False, reason="Crashes on OSX")
    def test10_array_of_const_char_star(self):
        """Test passting of const char*[]"""

        import ctypes

        import cppjit

        def py2c(pyargs):
            cargsn = (ctypes.c_char_p * len(pyargs))(*pyargs)
            return ctypes.POINTER(ctypes.c_char_p)(cargsn)

        pyargs = [b"hello", b"world"]

        cargs = py2c(pyargs)
        v = cppjit.gbl.ArrayOfCStrings.takes_array_of_cstrings(cargs, len(pyargs))
        assert len(v) == len(pyargs)
        assert list(v) == [x.decode() for x in pyargs]

        for t in (tuple, list):
            for pyargs in (t(["aap", "noot", "mies"]), t([b"zus", "jet", "tim"])):
                v = cppjit.gbl.ArrayOfCStrings.takes_array_of_cstrings(
                    pyargs, len(pyargs)
                )
                assert len(v) == len(pyargs)
                assert t(v) == t([type(x) == str and x or x.decode() for x in pyargs])

        # debatable, but the following works:
        pyargs = ["aap", 1, "mies"]
        with raises(TypeError):
            cppjit.gbl.ArrayOfCStrings.takes_array_of_cstrings(pyargs, len(pyargs))

        pyargs = ["aap", None, "mies"]
        with raises(TypeError):
            cppjit.gbl.ArrayOfCStrings.takes_array_of_cstrings(pyargs, len(pyargs))

    def test11_array_of_const_char_ref(self):
        """Test passting of const char**&"""

        import ctypes

        import cppjit
        import cppjit.ll

        # IN parameter case
        cppjit.cppdef("""\
        namespace ConstCharStarStarRef {
        int initialize(int& argc, char**& argv) {
            argv[0][0] = 'H';
            argv[1][0] = 'W';
            return argc;
        } }""")

        initialize = cppjit.gbl.ConstCharStarStarRef.initialize

        def py2c(pyargs):
            cargsn = (ctypes.c_char_p * len(pyargs))(*pyargs)
            return ctypes.POINTER(ctypes.c_char_p)(cargsn)

        pyargs = [b"hello", b"world"]
        cargs = py2c(pyargs)

        assert initialize(ctypes.c_int(len(pyargs)), py2c(pyargs)) == len(pyargs)
        assert cargs[0] == b"Hello"
        assert cargs[1] == b"World"

        # OUT parameter case
        cppjit.cppdef("""\
        namespace ConstCharStarStarRef {
        void fill(int& argc, char**& argv) {
            argc = 2;
            argv = new char*[argc];
            argv[0] = new char[6]; strcpy(argv[0], "Hello");
            argv[1] = new char[6]; strcpy(argv[1], "World");
        } }""")

        fill = cppjit.gbl.ConstCharStarStarRef.fill

        argc = ctypes.c_int(0)
        ptr = ctypes.c_void_p()

        fill(argc, ptr)

        assert argc.value == 2
        argv = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char_p))
        assert argv[0] == b"Hello"
        assert argv[1] == b"World"

        voidpp = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))
        for i in range(argc.value):
            cppjit.ll.array_delete(
                ctypes.cast(voidpp[i], ctypes.POINTER(ctypes.c_ubyte))
            )
        cppjit.ll.array_delete["char*"](
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char_p))
        )

    def test12_null_array(self):
        """Null low level view as empty list"""

        import cppjit

        cppjit.cppdef("""\
        namespace NullArray {
           double* gime_null() { return nullptr; }
        }""")

        ns = cppjit.gbl.NullArray

        assert not ns.gime_null()
        assert list(ns.gime_null()) == []

    def test13_array_interface(self):
        """Test usage of __array__ from numpy"""

        import cppjit

        try:
            import numpy as np
        except ImportError:
            skip("numpy is not installed")

        cppjit.cppdef("""\
        namespace ArrayConversions {
            int ivals[] = {1, 2, 3};
        }""")

        ns = cppjit.gbl.ArrayConversions

        a = ns.ivals

        b = np.array(a, copy=True)  # default behavior
        assert len(a) == len(b)
        a[0] = 4
        assert a[0] == 4
        assert b[0] == 1

        b = np.array(a, copy=False)
        assert b[0] == 4
        a[0] = 1
        assert b[0] == 1

        b = np.array(a, dtype=np.int32, copy=False)
        assert b[0] == 1
        a[0] = 5
        assert b[0] == 5

    def test14_templated_arrays(self):
        """Use of arrays in template types"""

        import cppjit

        assert cppjit.gbl.std.vector[int].value_type == "int"
        assert (
            cppjit.gbl.std.vector[cppjit.gbl.std.vector[int]].value_type
            == "std::vector<int>"
        )
        assert cppjit.gbl.std.vector["int[1]"].value_type == "int[1]"

    def test15_templated_arrays_gmpxx(self):
        """Use of gmpxx array types in templates"""

        import cppjit

        try:
            cppjit.include("gmpxx.h")
            cppjit.load_library("gmpxx")
        except (ImportError, RuntimeError):
            skip("gmpxx not installed")

        assert cppjit.gbl.std.vector[cppjit.gbl.mpz_class].value_type

        cppjit.cppdef("""\
        namespace test15_templated_arrays_gmpxx::vector {
           template <typename T>
           using value_type = typename T::value_type;
        }""")

        g = cppjit.gbl
        assert g.test15_templated_arrays_gmpxx.vector.value_type[
            g.std.vector[g.mpz_class]
        ]

    def test16_addressof_nullptr(self):
        import cppjit
        from cppjit import gbl

        cppjit.cppdef(r"""
        namespace LLV {
        int x = 10;
        int *ptr_x = &x;
        int *ptr_null = nullptr;
        }
        """)

        assert type(gbl.LLV.ptr_x) == cppjit._backend.LowLevelView
        assert type(gbl.LLV.ptr_null) == cppjit._backend.LowLevelView
        assert cppjit.addressof(gbl.LLV.ptr_x)
        assert cppjit.addressof(gbl.LLV.ptr_null) == 0

    def test17_array_delete_multidim(self):
        """Free a multidimensional (jagged) heap array with ll.array_delete"""

        import ctypes

        import cppjit
        import cppjit.ll

        # C++ allocates a jagged 2D array (array of pointers to rows)
        cppjit.cppdef("""\
        namespace ArrayDeleteMD {
        void make2d(int& n, int& m, double**& a) {
            n = 2; m = 3;
            a = new double*[n];
            for (int i = 0; i < n; ++i) {
                a[i] = new double[m];
                for (int j = 0; j < m; ++j) a[i][j] = 10.*i + j;
            }
        } }""")

        n = ctypes.c_int(0)
        m = ctypes.c_int(0)
        ptr = ctypes.c_void_p()
        cppjit.gbl.ArrayDeleteMD.make2d(n, m, ptr)

        assert n.value == 2
        assert m.value == 3
        rows = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))
        for i in range(n.value):
            row = ctypes.cast(rows[i], ctypes.POINTER(ctypes.c_double))
            for j in range(m.value):
                assert row[j] == 10.0 * i + j

        # delete[] each row, then the (outer) array of row pointers
        for i in range(n.value):
            cppjit.ll.array_delete(
                ctypes.cast(rows[i], ctypes.POINTER(ctypes.c_double))
            )
        cppjit.ll.array_delete["void"](ptr)

    def test18_array_delete_fixed(self):
        """Free a fixed-size contiguous multidimensional heap array"""

        import ctypes

        import cppjit
        import cppjit.ll

        # C++ allocates a contiguous 2D array (single new int[3][4])
        cppjit.cppdef("""\
        namespace ArrayDeleteFixed {
        intptr_t make(int& r, int& c) {
            r = 3; c = 4;
            int (*a)[4] = new int[3][4];
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j) a[i][j] = c*i + j;
            return (intptr_t)a;
        } }""")

        r = ctypes.c_int(0)
        c = ctypes.c_int(0)
        addr = cppjit.gbl.ArrayDeleteFixed.make(r, c)

        blk = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int))
        for k in range(r.value * c.value):
            assert blk[k] == k

        # a single contiguous allocation is released with a single delete[]
        cppjit.ll.array_delete(
            ctypes.cast(ctypes.c_void_p(addr), ctypes.POINTER(ctypes.c_int))
        )


class TestMULTIDIMARRAYS:
    def setup_class(cls):
        import cppjit

        cls.test_dct = test_dct
        cls.datatypes = cppjit.load_reflection_info(cls.test_dct)
        cls.numeric_builtin_types = [
            "short",
            "unsigned short",
            "int",
            "unsigned int",
            "long",
            "unsigned long",
            "long long",
            "unsigned long long",
            "float",
            "double",
        ]
        cls.nbt_short_names = [
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "llong",
            "ullong",
            "float",
            "double",
        ]
        try:
            import numpy as np

            if IS_WINDOWS:
                np_long, np_ulong = np.int32, np.uint32
            else:
                np_long, np_ulong = np.int64, np.uint64
            cls.numpy_builtin_types = [
                np.short,
                np.ushort,
                np.int32,
                np.uint32,
                np_long,
                np_ulong,
                np.longlong,
                np.ulonglong,
                np.float32,
                np.double,
            ]
        except ImportError:
            pass

    def _data_m(self, lbl):
        return [
            ("m_" + tp.replace(" ", "_") + lbl, tp) for tp in self.numeric_builtin_types
        ]

    def test01_2D_arrays(self):
        """Access and use of 2D data members"""

        import cppjit

        ns = cppjit.gbl.MultiDimArrays
        h = ns.DataHolder()

        data2a = self._data_m("2a")
        for m, tp in data2a:
            getattr(h, m).reshape((5, 7))

            arr = getattr(h, m)
            assert arr.shape == (5, 7)
            elem_tp = getattr(cppjit.gbl, tp)
            for i in range(5):
                for j in range(7):
                    val = elem_tp(5 * i + j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

            for i in range(5):
                for j in range(7):
                    arr[i][j] = elem_tp(4 + 5 * i + j)

            for i in range(5):
                for j in range(7):
                    val = elem_tp(4 + 5 * i + j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

        data2c = self._data_m("2c")
        for m, tp in data2c:
            arr = getattr(h, m)
            arr.reshape((3, 5))  # its own shape, the only one it accepts
            assert arr.shape == (3, 5)
            elem_tp = getattr(cppjit.gbl, tp)
            for i in range(3):
                for j in range(5):
                    val = elem_tp(3 * i + j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

    def test02_assign_2D_arrays(self):
        """Direct assignment of 2D arrays"""

        import cppjit

        try:
            import numpy as np
        except ImportError:
            skip("numpy is not installed")

        ns = cppjit.gbl.MultiDimArrays
        h = ns.DataHolder()

        # copy assignment
        data2c = self._data_m("2c")
        for itp, (m, tp) in enumerate(data2c):
            setattr(h, m, np.ones((3, 5), dtype=self.numpy_builtin_types[itp]))

            arr = getattr(h, m)
            assert arr.shape == (3, 5)
            val = getattr(cppjit.gbl, tp)(1)
            for i in range(3):
                for j in range(5):
                    assert arr[i][j] == val
                    assert arr[i, j] == val

        # size checking for copy assignment
        for itp, (m, tp) in enumerate(data2c):
            with raises(ValueError):
                setattr(h, m, np.ones((5, 5), dtype=self.numpy_builtin_types[itp]))

            with raises(ValueError):
                setattr(h, m, np.ones((3, 7), dtype=self.numpy_builtin_types[itp]))

        # pointer assignment
        N, M = 11, 7
        data2b = self._data_m("2b")
        for itp, (m, tp) in enumerate(data2b):
            setattr(h, m, getattr(h, "new_" + self.nbt_short_names[itp] + "2d")(N, M))

            arr = getattr(h, m)
            elem_tp = getattr(cppjit.gbl, tp)
            for i in range(N):
                for j in range(M):
                    val = elem_tp(7 * i + j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

            assert arr[2][3] != 10
            arr[2][3] = 10
            assert arr[2][3] == 10

    def test03_3D_arrays(self):
        """Access and use of 3D data members"""

        import cppjit

        ns = cppjit.gbl.MultiDimArrays
        h = ns.DataHolder()

        data3a = self._data_m("3a")
        for m, tp in data3a:
            getattr(h, m).reshape((5, 7, 11))

            arr = getattr(h, m)
            assert arr.shape == (5, 7, 11)
            elem_tp = getattr(cppjit.gbl, tp)
            for i in range(5):
                for j in range(7):
                    for k in range(11):
                        val = elem_tp(7 * i + 3 * j + k)
                        assert arr[i][j][k] == val
                        assert arr[i, j, k] == val

            for i in range(5):
                for j in range(7):
                    for k in range(11):
                        arr[i][j][k] = elem_tp(4 + 7 * i + 3 * j + k)

            for i in range(5):
                for j in range(7):
                    for k in range(11):
                        val = elem_tp(4 + 7 * i + 3 * j + k)
                        assert arr[i][j][k] == val
                        assert arr[i, j, k] == val

        data3c = self._data_m("3c")
        for m, tp in data3c:
            arr = getattr(h, m)
            assert arr.shape == (3, 5, 7)
            elem_tp = getattr(cppjit.gbl, tp)
            for i in range(3):
                for j in range(5):
                    for k in range(7):
                        val = elem_tp(3 * i + 2 * j + k)
                        assert arr[i][j][k] == val
                        assert arr[i, j, k] == val

    def test04_malloc(self):
        """Use of malloc to create multi-dim arrays"""

        import cppjit
        import cppjit.ll

        cppjit.cppdef("""\
        namespace MallocChecker {
        template<typename T>
        struct Foo {
            T* bar;

            Foo() {}
            Foo(T* other) : bar(other) {}

            bool eq(T* other) { return bar == other; }
        };

        template<typename T>
        auto create(T* other) {
            return Foo<T>(other);
        } }""")

        ns = cppjit.gbl.MallocChecker

        for dtype in [
            "int",
            "int*",
            "int**",
        ]:
            bar = cppjit.ll.malloc[dtype](4)
            assert len(bar) == 4

            # variable assignment
            foo = ns.Foo[dtype]()
            foo.bar = bar
            assert foo.eq(bar)

            # pointer passed to the constructor
            foo2 = ns.Foo[dtype](bar)
            assert foo2.eq(bar)

            # pointer passed to a function
            foo3 = ns.create[dtype](bar)
            assert foo3.eq(bar)

            cppjit.ll.free(bar)

    def test05_char_multidim(self):
        """Multi-dimensional char arrays"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace StringArray {
           char str_array[3][8] = {"s1\0", "s23\0", "s456\0"};
        }""")

        ns = cppjit.gbl.StringArray

        for i, v in enumerate(("s1", "s23", "s456")):
            assert len(ns.str_array[i]) == 8
            assert list(ns.str_array[i])[: len(v)] == list(v)

    def test06_fixed_multidim_array_itemsize(self):
        """conversion of fixed-length array low level views into NumPy arrays"""
        import cppjit

        try:
            import numpy as np
        except ImportError:
            skip("numpy is not installed")

        cases = [
            ("float", np.float32, (3, 5)),
            ("int", np.intc, (2, 6)),
            ("short", np.short, (5, 3)),
            ("unsigned char", np.ubyte, (4, 4)),
            ("int32_t", np.int32, (2, 8)),
            ("uint16_t", np.uint16, (7, 3)),
        ]

        for cpp_type, np_dtype, (rows, cols) in cases:
            tag = cpp_type.replace(" ", "_")
            cppjit.cppdef(f"""
                struct cpp_arr_{tag} {{
                    {cpp_type} a[{rows}][{cols}];
                }};
            """)
            s = getattr(cppjit.gbl, f"cpp_arr_{tag}")()

            itemsize = np.dtype(np_dtype).itemsize
            mv = memoryview(s.a)
            assert mv.ndim == 2
            assert mv.shape == (rows, cols)
            assert mv.itemsize == itemsize
            assert mv.strides == (cols * itemsize, itemsize)

            arr = np.array(s.a, dtype=np_dtype)
            assert arr.shape == (rows, cols)

    def test07_3D_custom_struct(self):
        import cppjit
        from cppjit import gbl

        cppjit.cppdef(r"""
        constexpr int S = 4;

        struct Klass {
            static int i;
            int k;
            Klass() : k(++i) {}
        };
        int Klass::i = 0;
        Klass klasses[S][S + 3][S + 7];

        bool consume_klass(Klass* c, int i, int j, int k) {
            if (c->k == ((S + 7) * (i * (S + 3) + j) + (k + 1))) return true;
            return false;
        }
        """)

        assert gbl.klasses
        # assert type(gbl.klasses) == cppjit._backend.LowLevelView # FIXME: https://github.com/compiler-research/cpyrt/issues/141

        for i in range(gbl.S):
            for j in range(gbl.S + 3):
                for k in range(gbl.S + 7):
                    assert gbl.consume_klass(gbl.klasses[i][j][k], i, j, k)

    def test08_reshape_sets_unknown_dimensions_only(self):
        """Reshaping fills in the dimensions the type leaves open"""

        import cppjit
        import cppjit.ll

        h = cppjit.gbl.MultiDimArrays.DataHolder()

        # a fixed-size array accepts only its own shape, and that must leave
        # its strides intact (a plain reshape used to corrupt them)
        arr = h.m_int2c
        assert arr.shape == (3, 5)
        strides = memoryview(arr).strides
        arr.reshape((3, 5))
        assert arr.shape == (3, 5)
        assert memoryview(arr).strides == strides
        for i in range(3):
            for j in range(5):
                assert arr[i][j] == 3 * i + j
                assert arr[i, j] == 3 * i + j

        raises(ValueError, arr.reshape, (5, 3))
        raises(ValueError, arr.reshape, (15,))
        assert arr.shape == (3, 5)
        assert arr[2][4] == 3 * 2 + 4

        # unknown dimensions can be set one at a time and, once set, stay
        arr = h.m_int2a
        assert len(arr.shape) == 2
        assert arr.shape[1] == -1
        raises(ValueError, arr.reshape, (35,))
        arr.reshape((5, -1))
        assert arr.shape[0] == 5 and arr.shape[1] == -1
        raises(ValueError, arr.reshape, (7, -1))
        arr.reshape((5, 7))
        assert arr.shape == (5, 7)
        for i in range(5):
            for j in range(7):
                assert arr[i][j] == h.m_int2a[i, j]

        # a rank-1 pointer view cannot become multi-dimensional either
        buf = cppjit.ll.malloc["int"](6)
        assert buf.shape == (6,)
        raises(ValueError, buf.reshape, (2, 3))
        assert buf.shape == (6,)
        cppjit.ll.free(buf)

        # a dimension that would overflow the byte size is rejected, too; the
        # outermost one counts row pointers here, not ints (used to slip by)
        arr = cppjit.gbl.MultiDimArrays.DataHolder().m_int2a
        raises(ValueError, arr.reshape, (5, sys.maxsize))
        raises(ValueError, arr.reshape, (sys.maxsize // 4 - 1, -1))

        # a freshly constructed view has no dimensions to set
        v = cppjit._backend.LowLevelView()
        raises(TypeError, v.reshape, ())


class TestCSTRINGARRAY:
    def test01_cstring_array_from_str(self):
        """A Python string can be assigned to a const char** data member"""

        import cppjit

        cppjit.cppdef("""\
        namespace CStringArray {
            struct S { const char** names = nullptr; };
            const char* as_chars(S& s) { return (const char*)s.names; }
        }""")

        ns = cppjit.gbl.CStringArray
        s = ns.S()

        s.names = "abc"
        assert ns.as_chars(s) == "abc"
