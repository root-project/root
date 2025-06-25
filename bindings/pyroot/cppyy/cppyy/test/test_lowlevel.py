import py, sys, pytest, os
from pytest import mark, raises, skip
from support import setup_make, pylong, pyunicode, IS_WINDOWS, ispypy


currpath = os.getcwd()
test_dct = currpath + "/libdatatypesDict"


class TestLOWLEVEL:
    def setup_class(cls):
        import cppyy

        cls.test_dct = test_dct
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

        at_least_17 = 201402 < cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;")
        cls.has_nested_namespace = at_least_17

    def test00_import_all(self):
        """Validity of `from cppyy.ll import *`"""

        from cppyy import ll

        for attr in ll.__all__:
            assert hasattr(ll, attr)

    def test01_llv_type(self):
        """Existence of LowLevelView type"""

        import cppyy.types

        assert cppyy.types.LowLevelView

    def test02_builtin_cpp_casts(self):
        """C++ casting of builtin types"""

        from cppyy import ll

        for cast in (ll.cast, ll.static_cast):
            assert type(cast[float](1)) == float
            assert cast[float](1) == 1.

            assert type(cast[int](1.1)) == int
            assert cast[int](1.1) == 1

        assert len(ll.reinterpret_cast['int*'](0)) == 0
        raises(ReferenceError, ll.reinterpret_cast['int*'](0).__getitem__, 0)

    def test03_memory(self):
        """Memory allocation and free-ing"""

        import cppyy
        from cppyy import ll

      # regular C malloc/free
        mem = cppyy.gbl.malloc(16)
        cppyy.gbl.free(mem)

      # typed styles
        mem = cppyy.ll.malloc[int](self.N)
        assert len(mem) == self.N
        assert not mem.__cpp_array__
        for i in range(self.N):
            mem[i] = i+1
            assert type(mem[i]) == int
            assert mem[i] == i+1
        cppyy.ll.free(mem)

      # C++ arrays
        mem = cppyy.ll.array_new[int](self.N)
        assert mem.__cpp_array__
        assert len(mem) == self.N
        for i in range(self.N):
            mem[i] = i+1
            assert type(mem[i]) == int
            assert mem[i] == i+1
        cppyy.ll.array_delete(mem)

        mem = cppyy.ll.array_new[int](self.N, managed=True)
        assert     mem.__python_owns__
        mem.__python_owns__ = False
        assert not mem.__python_owns__
        mem.__python_owns__ = True
        assert     mem.__python_owns__

    def test04_python_casts(self):
        """Casts to common Python pointer encapsulations"""

        import cppyy, cppyy.ll

        cppyy.cppdef("""namespace pycasts {
        struct SomeObject{};
        uintptr_t get_address(SomeObject* ptr) { return (intptr_t)ptr; }
        uintptr_t get_deref(void* ptr) { return (uintptr_t)(*(void**)ptr); }
        }""")

        from cppyy.gbl import pycasts

        s = pycasts.SomeObject()
        actual = pycasts.get_address(s)

        assert cppyy.ll.addressof(s) == actual
        assert cppyy.ll.as_ctypes(s).value == actual

        ptrptr = cppyy.ll.as_ctypes(s, byref=True)
        assert pycasts.get_deref(ptrptr) == actual

    @mark.xfail()
    def test05_array_as_ref(self):
        """Use arrays for pass-by-ref"""

        import cppyy, sys
        from array import array

        ctd = cppyy.gbl.CppyyTestData()

      # boolean type
        b = array('b', [0]); ctd.set_bool_r(b); assert b[0] == True

      # char types (as data)
        c = array('B', [0]); ctd.set_uchar_r(c); assert c[0] == ord('d')

      # integer types
        i = array('h', [0]);     ctd.set_short_r(i);  assert i[0] == -1
        i = array('H', [0]);     ctd.set_ushort_r(i); assert i[0] ==  2
        i = array('i', [0]);     ctd.set_int_r(i);    assert i[0] == -3
        i = array('I', [0]);     ctd.set_uint_r(i);   assert i[0] ==  4
        i = array('l', [0]);     ctd.set_long_r(i);   assert i[0] == -5
        i = array('L', [0]);     ctd.set_ulong_r(i);  assert i[0] ==  6
        if sys.hexversion >= 0x3000000:
            i = array('q', [0]); ctd.set_llong_r(i);  assert i[0] == -7
            i = array('Q', [0]); ctd.set_ullong_r(i); assert i[0] ==  8

      # floating point types
        f = array('f', [0]);     ctd.set_float_r(f);  assert f[0] ==  5.
        f = array('d', [0]);     ctd.set_double_r(f); assert f[0] == -5.

    @mark.xfail()
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

        import cppyy, ctypes
        import cppyy.ll

        ctd = cppyy.gbl.CppyyTestData()

      ### pass by reference/pointer and set value back

        for e in ['_r', '_p']:
          # boolean type
            b = ctypes.c_bool(False);      getattr(ctd, 'set_bool'+e)(b);     assert b.value == True

          # char types
            if e == '_r':
                c = ctypes.c_char(b'\0');  getattr(ctd, 'set_char'+e)(c);     assert c.value == b'a'
                c = ctypes.c_wchar(u'\0'); getattr(ctd, 'set_wchar'+e)(c);    assert c.value == u'b'
                c = ctypes.c_byte(0);      getattr(ctd, 'set_schar'+e)(c);    assert c.value == ord('c')
            c = ctypes.c_ubyte(0);         getattr(ctd, 'set_uchar'+e)(c);    assert c.value == ord('d')

          # integer types
            i = ctypes.c_short(0);         getattr(ctd, 'set_short'+e)(i);    assert i.value == -1
            i = ctypes.c_ushort(0);        getattr(ctd, 'set_ushort'+e)(i);   assert i.value ==  2
            i = ctypes.c_int(0);           getattr(ctd, 'set_int'+e)(i);      assert i.value == -3
            i = ctypes.c_uint(0);          getattr(ctd, 'set_uint'+e)(i);     assert i.value ==  4
            i = ctypes.c_long(0);          getattr(ctd, 'set_long'+e)(i);     assert i.value == -5
            i = ctypes.c_ulong(0);         getattr(ctd, 'set_ulong'+e)(i);    assert i.value ==  6
            i = ctypes.c_longlong(0);      getattr(ctd, 'set_llong'+e)(i);    assert i.value == -7
            i = ctypes.c_ulonglong(0);     getattr(ctd, 'set_ullong'+e)(i);   assert i.value ==  8

          # floating point types
            f = ctypes.c_float(0);         getattr(ctd, 'set_float'+e)(f);    assert f.value ==  5.
            f = ctypes.c_double(0);        getattr(ctd, 'set_double'+e)(f);   assert f.value == -5.
            f = ctypes.c_longdouble(0);    getattr(ctd, 'set_ldouble'+e)(f);  assert f.value == 10.

      ### pass by pointer and set value back, now using byref (not recommended)

        cb = ctypes.byref

      # boolean type
        b = ctypes.c_bool(False);     ctd.set_bool_p(cb(b));     assert b.value == True

      # char types
        c = ctypes.c_ubyte(0);        ctd.set_uchar_p(cb(c));    assert c.value == ord('d')

      # integer types
        i = ctypes.c_int8(0);         ctd.set_int8_p(cb(i));     assert i.value == -27
        i = ctypes.c_uint8(0);        ctd.set_uint8_p(cb(i));    assert i.value ==  28
        i = ctypes.c_short(0);        ctd.set_short_p(cb(i));    assert i.value == -1
        i = ctypes.c_ushort(0);       ctd.set_ushort_p(cb(i));   assert i.value ==  2
        i = ctypes.c_int(0);          ctd.set_int_p(cb(i));      assert i.value == -3
        i = ctypes.c_uint(0);         ctd.set_uint_p(cb(i));     assert i.value ==  4
        i = ctypes.c_long(0);         ctd.set_long_p(cb(i));     assert i.value == -5
        i = ctypes.c_ulong(0);        ctd.set_ulong_p(cb(i));    assert i.value ==  6
        i = ctypes.c_longlong(0);     ctd.set_llong_p(cb(i));    assert i.value == -7
        i = ctypes.c_ulonglong(0);    ctd.set_ullong_p(cb(i));   assert i.value ==  8

      # floating point types
        f = ctypes.c_float(0);        ctd.set_float_p(cb(f));    assert f.value ==  5.
        f = ctypes.c_double(0);       ctd.set_double_p(cb(f));   assert f.value == -5.

      ### pass by ptr/ptr with allocation (ptr/ptr is ambiguous in it's task, so many
        # types are allowed to pass; this tests allocation into the pointer)

        from ctypes import POINTER

      # boolean type
        b = POINTER(ctypes.c_bool)();     ctd.set_bool_ppa(b);
        assert b[0] == True; assert b[1] == False; assert b[2] == True
        cppyy.ll.array_delete(b)

      # char types
        c = POINTER(ctypes.c_ubyte)();    ctd.set_uchar_ppa(c)
        assert c[0] == ord('k'); assert c[1] == ord('l'); assert c[2] == ord('m')
        cppyy.ll.array_delete(c)

      # integer types
        i = POINTER(ctypes.c_int8)();         ctd.set_int8_ppa(i)
        assert i[0] == -27; assert i[1] == -28; assert i[2] == -29
        cppyy.ll.array_delete['void'](i)    # template resolves as signed char*
        i = POINTER(ctypes.c_uint8)();       ctd.set_uint8_ppa(i)
        assert i[0] ==  28; assert i[1] ==  29; assert i[2] ==  30
        cppyy.ll.array_delete['void'](i)    # template resolves as unsigned char*
        i = POINTER(ctypes.c_short)();        ctd.set_short_ppa(i)
        assert i[0] ==  -1; assert i[1] ==  -2; assert i[2] ==  -3
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_ushort)();       ctd.set_ushort_ppa(i)
        assert i[0] ==   4; assert i[1] ==   5; assert i[2] ==   6
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_int)();          ctd.set_int_ppa(i)
        assert i[0] ==  -7; assert i[1] ==  -8; assert i[2] ==  -9
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_uint)();         ctd.set_uint_ppa(i)
        assert i[0] ==  10; assert i[1] ==  11; assert i[2] ==  12
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_long)();         ctd.set_long_ppa(i)
        assert i[0] == -13; assert i[1] == -14; assert i[2] == -15
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_ulong)();        ctd.set_ulong_ppa(i)
        assert i[0] ==  16; assert i[1] ==  17; assert i[2] ==  18
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_longlong)();     ctd.set_llong_ppa(i)
        assert i[0] == -19; assert i[1] == -20; assert i[2] == -21
        cppyy.ll.array_delete(i)
        i = POINTER(ctypes.c_ulonglong)();    ctd.set_ullong_ppa(i)
        assert i[0] ==  22; assert i[1] ==  23; assert i[2] ==  24
        cppyy.ll.array_delete(i)

      # floating point types
        f = POINTER(ctypes.c_float)();        ctd.set_float_ppa(f)
        assert f[0] ==   5; assert f[1] ==  10; assert f[2] ==  20
        cppyy.ll.array_delete(f)
        f = POINTER(ctypes.c_double)();       ctd.set_double_ppa(f)
        assert f[0] ==  -5; assert f[1] == -10; assert f[2] == -20
        cppyy.ll.array_delete(f)
        f = POINTER(ctypes.c_longdouble)();   ctd.set_ldouble_ppa(f)
        assert f[0] ==   5; assert f[1] ==  10; assert f[2] ==  20
        cppyy.ll.array_delete(f)

    def test07_ctypes_pointer_types(self):
        """Use ctypes for pass-by-ptr/ptr-ptr"""

        if ispypy:
            skip('memory corruption')

        # See:
        #  https://docs.python.org/2/library/ctypes.html#fundamental-data-types
        #
        # ctypes type       C type                                      Python type
        # ------------------------------------------------------------------------------
        # c_char_p          char* (NULL terminated)                     string or None
        # c_wchar_p         wchar_t* (NULL terminated)                  unicode or None
        # c_void_p          void*                                       int/long or None

        import cppyy, ctypes

        ctd = cppyy.gbl.CppyyTestData()

        ptr = ctypes.c_char_p()
        for meth in ['char', 'cchar']:
            val = getattr(ctd, 'set_'+meth+'_ppm')(ptr)
            assert ctd.freeit(ptr) == val

        ptr = ctypes.c_wchar_p()
        for meth in ['wchar', 'cwchar']:
            val = getattr(ctd, 'set_'+meth+'_ppm')(ptr)
            assert ctd.freeit(ptr) == val

        ptr = ctypes.c_void_p()
        val = ctd.set_void_ppm(ptr)
        assert ctd.freeit(ptr) == val

    def test08_ctypes_type_correctness(self):
        """If types don't match with ctypes, expect exceptions"""

        import cppyy, ctypes

        ctd = cppyy.gbl.CppyyTestData()

        meth_types = ['bool', 'double']
        if not IS_WINDOWS:
            meth_types.append('long')

        i = ctypes.c_int(0);
        for ext in ['_r', '_p']:
            for meth in meth_types:
                with raises(TypeError): getattr(ctd, 'set_'+meth+ext)(i)

    def test09_numpy_bool_array(self):
        """Test passing of numpy bool array"""

        import cppyy
        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        cppyy.cppdef('int convert_bool(bool* x) {return *x;}')

        x = np.array([True], dtype=bool)
        assert cppyy.gbl.convert_bool(x)

    def test10_array_of_const_char_star(self):
        """Test passting of const char*[]"""

        import cppyy, ctypes

        def py2c(pyargs):
            cargsn = (ctypes.c_char_p * len(pyargs))(*pyargs)
            return ctypes.POINTER(ctypes.c_char_p)(cargsn)

        pyargs = [b'hello', b'world']

        cargs = py2c(pyargs)
        v = cppyy.gbl.ArrayOfCStrings.takes_array_of_cstrings(cargs, len(pyargs))
        assert len(v) == len(pyargs)
        assert list(v) == [x.decode() for x in pyargs]

        for t in (tuple, list):
            for pyargs in (t(['aap', 'noot', 'mies']), t([b'zus', 'jet', 'tim'])):
                v = cppyy.gbl.ArrayOfCStrings.takes_array_of_cstrings(pyargs, len(pyargs))
                assert len(v) == len(pyargs)
                assert t(v) == t([type(x) == str and x or x.decode() for x in pyargs])

      # debatable, but the following works:
        pyargs = ['aap', 1, 'mies']
        with raises(TypeError):
            cppyy.gbl.ArrayOfCStrings.takes_array_of_cstrings(pyargs, len(pyargs))

        pyargs = ['aap', None, 'mies']
        with raises(TypeError):
            cppyy.gbl.ArrayOfCStrings.takes_array_of_cstrings(pyargs, len(pyargs))

    def test11_array_of_const_char_ref(self):
        """Test passting of const char**&"""

        import cppyy, ctypes
        import cppyy.ll

      # IN parameter case
        cppyy.cppdef("""\
        namespace ConstCharStarStarRef {
        int initialize(int& argc, char**& argv) {
            argv[0][0] = 'H';
            argv[1][0] = 'W';
            return argc;
        } }""")

        initialize = cppyy.gbl.ConstCharStarStarRef.initialize

        def py2c(pyargs):
            cargsn = (ctypes.c_char_p * len(pyargs))(*pyargs)
            return ctypes.POINTER(ctypes.c_char_p)(cargsn)

        pyargs = [b'hello', b'world']
        cargs = py2c(pyargs)

        assert initialize(ctypes.c_int(len(pyargs)), py2c(pyargs)) == len(pyargs)
        assert cargs[0] == b'Hello'
        assert cargs[1] == b'World'

      # OUT parameter case
        cppyy.cppdef("""\
        namespace ConstCharStarStarRef {
        void fill(int& argc, char**& argv) {
            argc = 2;
            argv = new char*[argc];
            argv[0] = new char[6]; strcpy(argv[0], "Hello");
            argv[1] = new char[6]; strcpy(argv[1], "World");
        } }""")

        fill = cppyy.gbl.ConstCharStarStarRef.fill

        argc = ctypes.c_int(0)
        ptr = ctypes.c_void_p()

        fill(argc, ptr)

        assert argc.value == 2
        argv = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char_p))
        assert argv[0] == b"Hello"
        assert argv[1] == b"World"

        voidpp = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))
        for i in range(argc.value):
            cppyy.ll.free(ctypes.cast(voidpp[i], ctypes.c_void_p))
        cppyy.ll.free(ptr)

    def test12_null_array(self):
        """Null low level view as empty list"""

        import cppyy

        cppyy.cppdef("""\
        namespace NullArray {
           double* gime_null() { return nullptr; }
        }""")

        ns = cppyy.gbl.NullArray

        assert not ns.gime_null()
        assert list(ns.gime_null()) == []

    def test13_array_interface(self):
        """Test usage of __array__ from numpy"""

        import cppyy

        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        cppyy.cppdef("""\
        namespace ArrayConversions {
            int ivals[] = {1, 2, 3};
        }""")

        ns = cppyy.gbl.ArrayConversions

        a = ns.ivals

        b = np.array(a, copy=True)     # default behavior
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

        import cppyy

        assert cppyy.gbl.std.vector[int].value_type == 'int'
        assert cppyy.gbl.std.vector[cppyy.gbl.std.vector[int]].value_type == 'std::vector<int>'
        assert cppyy.gbl.std.vector['int[1]'].value_type == 'int[1]'

    @mark.xfail()
    def test15_templated_arrays_gmpxx(self):
        """Use of gmpxx array types in templates"""

        if not self.has_nested_namespace:
            return

        import cppyy

        try:
            cppyy.include("gmpxx.h")
            cppyy.load_library('gmpxx')
        except ImportError:
            skip("gmpxx not installed")

        assert cppyy.gbl.std.vector[cppyy.gbl.mpz_class].value_type

        cppyy.cppdef("""\
        namespace test15_templated_arrays_gmpxx::vector {
           template <typename T>
           using value_type = typename T::value_type;
        }""")


        g = cppyy.gbl
        assert g.test15_templated_arrays_gmpxx.vector.value_type[g.std.vector[g.mpz_class]]


class TestMULTIDIMARRAYS:
    def setup_class(cls):
        import cppyy

        cls.test_dct = test_dct
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.numeric_builtin_types = [
            'short', 'unsigned short', 'int', 'unsigned int', 'long', 'unsigned long',
            'long long', 'unsigned long long', 'float', 'double'
        ]
        cls.nbt_short_names = [
            'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'llong', 'ullong', 'float', 'double'
        ]
        try:
            import numpy as np
            if IS_WINDOWS:
                np_long, np_ulong = np.int32, np.uint32
            else:
                np_long, np_ulong = np.int64, np.uint64
            cls.numpy_builtin_types = [
                np.short, np.ushort, np.int32, np.uint32, np_long, np_ulong,
                np.longlong, np.ulonglong, np.float32, np.double
            ]
        except ImportError:
            pass

    def _data_m(self, lbl):
        return [('m_'+tp.replace(' ', '_')+lbl, tp) for tp in self.numeric_builtin_types]

    @mark.xfail()
    def test01_2D_arrays(self):
        """Access and use of 2D data members"""

        import cppyy

        ns = cppyy.gbl.MultiDimArrays
        h = ns.DataHolder()

        data2a = self._data_m('2a')
        for m, tp in data2a:
            getattr(h, m).reshape((5, 7))

            arr = getattr(h, m)
            assert arr.shape == (5, 7)
            elem_tp = getattr(cppyy.gbl, tp)
            for i in range(5):
                for j in range(7):
                    val = elem_tp(5*i+j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

            for i in range(5):
                for j in range(7):
                    arr[i][j] = elem_tp(4+5*i+j)

            for i in range(5):
                for j in range(7):
                    val = elem_tp(4+5*i+j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

        data2c = self._data_m('2c')
        for m, tp in data2c:
            arr = getattr(h, m)
            assert arr.shape == (3, 5)
            elem_tp = getattr(cppyy.gbl, tp)
            for i in range(3):
                for j in range(5):
                    val = elem_tp(3*i+j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

    @mark.xfail()
    def test02_assign_2D_arrays(self):
        """Direct assignment of 2D arrays"""

        import cppyy

        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        ns = cppyy.gbl.MultiDimArrays
        h = ns.DataHolder()

      # copy assignment
        data2c = self._data_m('2c')
        for itp, (m, tp) in enumerate(data2c):
            setattr(h, m, np.ones((3, 5), dtype=self.numpy_builtin_types[itp]))

            arr = getattr(h, m)
            assert arr.shape == (3, 5)
            val = getattr(cppyy.gbl, tp)(1)
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
        data2b = self._data_m('2b')
        for itp, (m, tp) in enumerate(data2b):
            setattr(h, m, getattr(h, 'new_'+self.nbt_short_names[itp]+'2d')(N, M))

            arr = getattr(h, m)
            elem_tp = getattr(cppyy.gbl, tp)
            for i in range(N):
                for j in range(M):
                    val = elem_tp(7*i+j)
                    assert arr[i][j] == val
                    assert arr[i, j] == val

            assert arr[2][3] != 10
            arr[2][3] = 10
            assert arr[2][3] == 10

    @mark.xfail()
    def test03_3D_arrays(self):
        """Access and use of 3D data members"""

        import cppyy

        ns = cppyy.gbl.MultiDimArrays
        h = ns.DataHolder()

        data3a = self._data_m('3a')
        for m, tp in data3a:
            getattr(h, m).reshape((5, 7, 11))

            arr = getattr(h, m)
            assert arr.shape == (5, 7, 11)
            elem_tp = getattr(cppyy.gbl, tp)
            for i in range(5):
                for j in range(7):
                    for k in range(11):
                        val = elem_tp(7*i+3*j+k)
                        assert arr[i][j][k] == val
                        assert arr[i, j, k] == val

            for i in range(5):
                for j in range(7):
                    for k in range(11):
                        arr[i][j][k] = elem_tp(4+7*i+3*j+k)

            for i in range(5):
                for j in range(7):
                    for k in range(11):
                        val = elem_tp(4+7*i+3*j+k)
                        assert arr[i][j][k] == val
                        assert arr[i, j, k] == val

        data3c = self._data_m('3c')
        for m, tp in data3c:
            arr = getattr(h, m)
            assert arr.shape == (3, 5, 7)
            elem_tp = getattr(cppyy.gbl, tp)
            for i in range(3):
                for j in range(5):
                    for k in range(7):
                        val = elem_tp(3*i+2*j+k)
                        assert arr[i][j][k] == val
                        assert arr[i, j, k] == val

    def test04_malloc(self):
        """Use of malloc to create multi-dim arrays"""

        import cppyy
        import cppyy.ll

        cppyy.cppdef("""\
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

        ns = cppyy.gbl.MallocChecker

        for dtype in ["int", "int*", "int**",]:
            bar = cppyy.ll.malloc[dtype](4)
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

            cppyy.ll.free(bar)

    def test05_char_multidim(self):
        """Multi-dimensional char arrays"""

        import cppyy

        cppyy.cppdef(r"""\
        namespace StringArray {
           char str_array[3][7] = {"s1\0", "s23\0", "s456\0"};
        }""")

        ns = cppyy.gbl.StringArray

        for i, v in enumerate(("s1", "s23", "s456")):
            assert len(ns.str_array[i]) == 7
            assert ns.str_array[i].as_string() == v


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
