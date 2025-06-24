import py, sys, pytest, os
from pytest import mark, raises, skip
from support import setup_make, pylong, pyunicode, IS_MAC_ARM

currpath = os.getcwd()
test_dct = currpath + "/libdatatypesDict"


class TestDATATYPES:
    def setup_class(cls):
        import cppyy

        cls.test_dct = test_dct
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

        at_least_17 = 201402 < cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;")
        cls.has_byte     = at_least_17
        cls.has_optional = at_least_17

    @mark.skip()
    def test01_instance_data_read_access(self):
        """Read access to instance public data and verify values"""

        import cppyy
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        # reading boolean type
        assert c.m_bool == False
        assert not c.get_bool(); assert not c.get_bool_cr(); assert not c.get_bool_r()

        # reading char types
        assert c.m_char  == 'a'
        assert c.m_schar == 'b'
        assert c.m_uchar == 'c'
        assert type(c.m_wchar) == pyunicode
        assert c.m_wchar == u'D'
        assert type(c.m_char16) == pyunicode
        assert c.m_char16 == u'\u00df'
        assert type(c.m_char32) == pyunicode
        assert c.m_char32 == u'\u00df'

        # reading integer types
        assert c.m_int8    == - 9; assert c.get_int8_cr()    == - 9; assert c.get_int8_r()    == - 9
        assert c.m_uint8   ==   9; assert c.get_uint8_cr()   ==   9; assert c.get_uint8_r()   ==   9
        if self.has_byte:
            assert c.m_byte == ord('d'); assert c.get_byte_cr() == ord('d'); assert c.get_byte_r() == ord('d')
        assert c.m_short   == -11; assert c.get_short_cr()   == -11; assert c.get_short_r()   == -11
        assert c.m_ushort  ==  11; assert c.get_ushort_cr()  ==  11; assert c.get_ushort_r()  ==  11
        assert c.m_int     == -22; assert c.get_int_cr()     == -22; assert c.get_int_r()     == -22
        assert c.m_uint    ==  22; assert c.get_uint_cr()    ==  22; assert c.get_uint_r()    ==  22
        assert c.m_long    == -33; assert c.get_long_cr()    == -33; assert c.get_long_r()    == -33
        assert c.m_ulong   ==  33; assert c.get_ulong_cr()   ==  33; assert c.get_ulong_r()   ==  33
        assert c.m_llong   == -44; assert c.get_llong_cr()   == -44; assert c.get_llong_r()   == -44
        assert c.m_ullong  ==  44; assert c.get_ullong_cr()  ==  44; assert c.get_ullong_r()  ==  44
        assert c.m_long64  == -55; assert c.get_long64_cr()  == -55; assert c.get_long64_r()  == -55
        assert c.m_ulong64 ==  55; assert c.get_ulong64_cr() ==  55; assert c.get_ulong64_r() ==  55

        # reading floating point types
        assert round(c.m_float          + 66.,  5) == 0
        assert round(c.get_float_cr()   + 66.,  5) == 0
        assert round(c.get_float_r()    + 66.,  5) == 0
        assert round(c.m_double         + 77., 11) == 0
        assert round(c.get_double_cr()  + 77., 11) == 0
        assert round(c.get_double_r()   + 77., 11) == 0
        assert round(c.m_ldouble        + 88., 24) == 0
        assert round(c.get_ldouble_cr() + 88., 24) == 0
        assert round(c.get_ldouble_r()  + 88., 24) == 0
        assert round(c.get_ldouble_def()  -1., 24) == 0
        assert round(c.get_ldouble_def(2) -2., 24) == 0

        # complex<double> type
        assert type(c.get_complex()) == complex
        assert round(c.get_complex().real    -  99., 11) == 0
        assert round(c.get_complex().imag    - 101., 11) == 0
        assert repr(c.get_complex()) == '(99+101j)'
        assert round(c.get_complex_cr().real -  99., 11) == 0
        assert round(c.get_complex_cr().imag - 101., 11) == 0
        assert round(c.get_complex_r().real  -  99., 11) == 0
        assert round(c.get_complex_r().imag  - 101., 11) == 0
        assert complex(cppyy.gbl.std.complex['double'](1, 2)) == complex(1, 2)
        assert repr(cppyy.gbl.std.complex['double'](1, 2)) == '(1+2j)'

        # complex<int> retains C++ type in all cases (but includes pythonization to
        # resemble Python's complex more closely
        assert type(c.get_icomplex()) == cppyy.gbl.std.complex[int]
        assert round(c.get_icomplex().real    - 121., 11) == 0
        assert round(c.get_icomplex().imag    - 141., 11) == 0
        assert repr(c.get_icomplex()) == '(121+141j)'
        assert round(c.get_icomplex_cr().real - 121., 11) == 0
        assert round(c.get_icomplex_cr().imag - 141., 11) == 0
        assert type(c.get_icomplex_r()) == cppyy.gbl.std.complex[int]
        assert round(c.get_icomplex_r().real  - 121., 11) == 0
        assert round(c.get_icomplex_r().imag  - 141., 11) == 0
        assert complex(cppyy.gbl.std.complex['int'](1, 2)) == complex(1, 2)

        # _Complex double type
        assert type(c.get_ccomplex()) == complex
        assert round(c.get_ccomplex().real    - 151., 11) == 0
        assert round(c.get_ccomplex().imag    - 161., 11) == 0
        assert repr(c.get_ccomplex()) == '(151+161j)'
        assert round(c.get_ccomplex_cr().real - 151., 11) == 0
        assert round(c.get_ccomplex_cr().imag - 161., 11) == 0
        assert round(c.get_ccomplex_r().real  - 151., 11) == 0
        assert round(c.get_ccomplex_r().imag  - 161., 11) == 0

        # complex overloads
        cppyy.cppdef("""
        namespace ComplexOverload {
          template<typename T>
          struct CO {
            CO(std::size_t sz) : m_size(sz), m_cplx(std::complex<T>(7,42)) {}
            CO(std::complex<T> cplx) : m_size(42), m_cplx(cplx) {}

            std::size_t m_size;
            std::complex<T> m_cplx;
          };
        }""")

        COd = cppyy.gbl.ComplexOverload.CO['double']
        COf = cppyy.gbl.ComplexOverload.CO['float']
        scf = cppyy.gbl.std.complex['float']

        assert COd(2).m_size     == 2
        assert COd(2).m_cplx     == 7.+42j
        assert COd(3.14).m_size  == 42
        assert COd(3.14).m_cplx  == 3.14+0j
        assert COd(9.+7j).m_size == 42
        assert COd(9.+7j).m_cplx == 9.+7j

        assert COf(2).m_size     == 2
        assert COf(2).m_cplx     == scf(7, 42)
        assert COf(3.14).m_size  == 42
        assert COf(3.14).m_cplx  == scf(3.14, 0)
        assert COf(9.+7j).m_size == 42
        assert COf(9.+7j).m_cplx == scf(9., 7.)

        # reading of enum types
        assert c.m_enum == CppyyTestData.kNothing
        assert c.m_enum == c.kNothing

        # reading of boolean array
        for i in range(self.N):
            assert c.m_bool_array[i]        ==   bool(i%2)
            assert c.get_bool_array()[i]    ==   bool(i%2)
            assert c.m_bool_array2[i]       ==   bool((i+1)%2)
            assert c.get_bool_array2()[i]   ==   bool((i+1)%2)

        # reading of integer array types
        names = ['schar', 'uchar',   'int8', 'uint8',  'short', 'ushort',     'int',   'uint',     'long',  'ulong']
        alpha = [ (1, 2),  (1, 2), (-1, -2),  (3, 4), (-5, -6),   (7, 8), (-9, -10), (11, 12), (-13, -14), (15, 16)]
        if self.has_byte: names.append('byte'); alpha.append((3,4))

        for j in range(self.N):
            assert getattr(c, 'm_%s_array'    % names[i])[i]   == alpha[i][0]*i
            assert getattr(c, 'get_%s_array'  % names[i])()[i] == alpha[i][0]*i
            assert getattr(c, 'm_%s_array2'   % names[i])[i]   == alpha[i][1]*i
            assert getattr(c, 'get_%s_array2' % names[i])()[i] == alpha[i][1]*i

        # reading of floating point array types
        for k in range(self.N):
            assert round(c.m_float_array[k]   + 13.*k, 5) == 0
            assert round(c.m_float_array2[k]  + 14.*k, 5) == 0
            assert round(c.m_double_array[k]  + 15.*k, 8) == 0
            assert round(c.m_double_array2[k] + 16.*k, 8) == 0

        # out-of-bounds checks
        raises(IndexError, c.m_schar_array.__getitem__,  self.N)
        raises(IndexError, c.m_uchar_array.__getitem__,  self.N)
        if self.has_byte:
            raises(IndexError, c.m_byte_array.__getitem__,   self.N)
        raises(IndexError, c.m_int8_array.__getitem__,   self.N)
        raises(IndexError, c.m_uint8_array.__getitem__,  self.N)
        raises(IndexError, c.m_short_array.__getitem__,  self.N)
        raises(IndexError, c.m_ushort_array.__getitem__, self.N)
        raises(IndexError, c.m_int_array.__getitem__,    self.N)
        raises(IndexError, c.m_uint_array.__getitem__,   self.N)
        raises(IndexError, c.m_long_array.__getitem__,   self.N)
        raises(IndexError, c.m_ulong_array.__getitem__,  self.N)
        raises(IndexError, c.m_float_array.__getitem__,  self.N)
        raises(IndexError, c.m_double_array.__getitem__, self.N)

        # can not access an instance member on the class
        raises(AttributeError, getattr, CppyyTestData, 'm_bool')
        raises(AttributeError, getattr, CppyyTestData, 'm_int')

        assert not hasattr(CppyyTestData, 'm_bool')
        assert not hasattr(CppyyTestData, 'm_int')

        c.__destruct__()

    @mark.xfail()
    def test02_instance_data_write_access(self):
        """Test write access to instance public data and verify values"""

        import cppyy
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        # boolean types through functions
        c.set_bool(True);  assert c.get_bool() == True
        c.set_bool(0);     assert c.get_bool() == False

        # boolean types through data members
        c.m_bool = True;   assert c.get_bool() == True
        c.set_bool(True);  assert c.m_bool     == True
        c.m_bool = 0;      assert c.get_bool() == False
        c.set_bool(0);     assert c.m_bool     == False
        raises(ValueError, c.set_bool, 10)

        # char types through functions
        c.set_char('c');   assert c.get_char()  == 'c'
        c.set_uchar('e');  assert c.get_uchar() == 'e'
        c.set_wchar(u'F'); assert c.get_wchar() == u'F'
        assert type(c.get_wchar()) == pyunicode
        c.set_char16(u'\u00f2');     assert c.get_char16() == u'\u00f2'
        c.set_char32(u'\U0001f31c'); assert c.get_char32() == u'\U0001f31c'

        # char types through data members
        c.m_char = 'b';    assert c.get_char()  ==     'b'
        c.m_char = 40;     assert c.get_char()  == chr(40)
        c.set_char('c');   assert c.m_char      ==     'c'
        c.set_char(41);    assert c.m_char      == chr(41)
        c.m_uchar = 'd';   assert c.get_uchar() ==     'd'
        c.m_uchar = 42;    assert c.get_uchar() == chr(42)
        c.set_uchar('e');  assert c.m_uchar     ==     'e'
        c.set_uchar(43);   assert c.m_uchar     == chr(43)
        c.m_wchar = u'G';  assert c.get_wchar() ==    u'G'
        c.set_wchar(u'H'); assert c.m_wchar     ==    u'H'
        c.m_char16 = u'\u00f3';  assert c.get_char16() == u'\u00f3'
        c.set_char16(u'\u00f4'); assert c.m_char16     == u'\u00f4'
        c.m_char32 = u'\U0001f31d';  assert c.get_char32() == u'\U0001f31d'
        c.set_char32(u'\U0001f31e'); assert c.m_char32     == u'\U0001f31e'

        raises(ValueError, c.set_char,   "string")
        raises(ValueError, c.set_char,   500)
        raises(ValueError, c.set_uchar,  "string")
        raises(ValueError, c.set_uchar,  -1)
        raises(ValueError, c.set_wchar,  "string")
        raises(ValueError, c.set_char16, "string")
        raises(ValueError, c.set_char32, "string")

        # integer types
        names = ['int8', 'uint8', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'llong', 'ullong']
        if self.has_byte: names.append('byte')

        for i in range(len(names)):
            setattr(c, 'm_'+names[i], i)
            assert eval('c.get_%s()' % names[i]) == i

        for i in range(len(names)):
            getattr(c, 'set_'+names[i])(2*i)
            assert eval('c.m_%s' % names[i]) == 2*i

        for i in range(len(names)):
            getattr(c, 'set_'+names[i]+'_cr')(3*i)
            assert eval('c.m_%s' % names[i]) == 3*i

        for i in range(len(names)):
            getattr(c, 'set_'+names[i]+'_rv')(4*i)
            assert eval('c.m_%s' % names[i]) == 4*i

        for i in range(len(names)):
            setattr(c, 'm_'+names[i], cppyy.default)
            assert eval('c.get_%s()' % names[i]) == 0

        for i in range(len(names)):
            getattr(c, 'set_'+names[i])(cppyy.default)
            assert eval('c.m_%s' % names[i]) == 0

        for i in range(len(names)):
            getattr(c, 'set_'+names[i]+'_cr')(cppyy.default)
            assert eval('c.m_%s' % names[i]) == 0

        # float types through functions
        c.set_float(0.123);   assert round(c.get_float()   - 0.123, 5) == 0
        c.set_double(0.456);  assert round(c.get_double()  - 0.456, 8) == 0
        c.set_ldouble(0.789); assert round(c.get_ldouble() - 0.789, 8) == 0

        # float types through data members
        c.m_float = 0.123;       assert round(c.get_float()   - 0.123, 5) == 0
        c.set_float(0.234);      assert round(c.m_float       - 0.234, 5) == 0
        c.set_float_cr(0.456);   assert round(c.m_float       - 0.456, 5) == 0
        c.m_double = 0.678;      assert round(c.get_double()  - 0.678, 8) == 0
        c.set_double(0.890);     assert round(c.m_double      - 0.890, 8) == 0
        c.set_double_cr(0.012);  assert round(c.m_double      - 0.012, 8) == 0
        c.m_ldouble = 0.876;     assert round(c.get_ldouble() - 0.876, 8) == 0
        c.set_ldouble(0.098);    assert round(c.m_ldouble     - 0.098, 8) == 0
        c.set_ldouble_cr(0.210); assert round(c.m_ldouble     - 0.210, 8) == 0

        names = ['float', 'double', 'ldouble']
        for i in range(len(names)):
            setattr(c, 'm_'+names[i], cppyy.default)
            assert eval('c.get_%s()' % names[i]) == 0.

        for i in range(len(names)):
            getattr(c, 'set_'+names[i])(cppyy.default)
            assert eval('c.m_%s' % names[i]) == 0.

        for i in range(len(names)):
            getattr(c, 'set_'+names[i]+'_cr')(cppyy.default)
            assert eval('c.m_%s' % names[i]) == 0.

        # (non-)writing of enum types
        raises(TypeError, setattr, CppyyTestData, 'kNothing', 42)

        # arrays; there will be pointer copies, so destroy the current ones
        c.destroy_arrays()

        # integer arrays (skip int8_t and uint8_t as these are presented as (unsigned) char still)
        names = ['uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']
        if self.has_byte: names.append('byte')

        import array
        a = range(self.N)
        atypes = ['B', 'h', 'H', 'i', 'I', 'l', 'L']
        if self.has_byte: atypes.append('B')
        for j in range(len(names)):
            b = array.array(atypes[j], a)
            setattr(c, 'm_'+names[j]+'_array', b)     # buffer copies
            for i in range(self.N):
                assert eval('c.m_%s_array[i]' % names[j]) == b[i]

            setattr(c, 'm_'+names[j]+'_array2', b)    # pointer copies
            assert 3 < self.N
            b[3] = 28
            for i in range(self.N):
                assert eval('c.m_%s_array2[i]' % names[j]) == b[i]

        # can not write to constant data
        assert c.m_const_int == 17
        raises(TypeError, setattr, c, 'm_const_int', 71)

        c.__destruct__()

    def test03_array_passing(self):
        """Test passing of array arguments"""

        import cppyy, array, sys
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        a = range(self.N)
        # test arrays in mixed order, to give overload resolution a workout
        for t in ['d', 'i', 'f', 'H', 'I', 'h', 'L', 'l']:
            b = array.array(t, a)

            # typed passing
            ca = c.pass_array(b)
            if t != 'l': assert type(ca[0]) == type(b[0])
            else: assert type(ca[0]) == pylong   # 'l' returns PyInt for small values in p2
            assert len(b) == self.N
            for i in range(self.N):
                assert ca[i] == b[i]

            # void* passing
            ca = eval('c.pass_void_array_%s(b)' % t)
            if t != 'l': assert type(ca[0]) == type(b[0])
            else: assert type(ca[0]) == pylong  # 'l' returns PyInt for small values in p2
            assert len(b) == self.N
            for i in range(self.N):
                assert ca[i] == b[i]

        # NULL/nullptr passing (will use short*)
        assert not c.pass_array(0)
        raises(Exception, c.pass_array(0).__getitem__, 0)    # raises SegfaultException
        assert raises(TypeError, c.pass_array, None)
        assert not c.pass_array(cppyy.nullptr)
        raises(Exception, c.pass_array(cppyy.nullptr).__getitem__, 0) # id. id.

        c.__destruct__()

    @mark.xfail()
    def test04_class_read_access(self):
        """Test read access to class public data and verify values"""

        import cppyy, sys
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        # char types
        assert CppyyTestData.s_char     == 'c'
        assert c.s_char                 == 'c'
        assert CppyyTestData.s_uchar    == 'u'
        assert c.s_uchar                == 'u'
        assert CppyyTestData.s_wchar    == u'U'
        assert c.s_wchar                == u'U'
        assert CppyyTestData.s_char16   == u'\u6c29'
        assert c.s_char16               == u'\u6c29'
        assert CppyyTestData.s_char32   == u'\U0001f34b'
        assert c.s_char32               == u'\U0001f34b'

        assert type(c.s_wchar)              == pyunicode
        assert type(CppyyTestData.s_wchar)  == pyunicode
        assert type(c.s_char16)             == pyunicode
        assert type(CppyyTestData.s_char16) == pyunicode
        assert type(c.s_char32)             == pyunicode
        assert type(CppyyTestData.s_char32) == pyunicode

        # integer types
        if self.has_byte:
            assert CppyyTestData.s_byte == ord('b')
            assert c.s_byte             == ord('b')
        assert CppyyTestData.s_int8     == - 87
        assert c.s_int8                 == - 87
        assert CppyyTestData.s_uint8    ==   87
        assert c.s_uint8                ==   87
        assert CppyyTestData.s_short    == -101
        assert c.s_short                == -101
        assert c.s_ushort               ==  255
        assert CppyyTestData.s_ushort   ==  255
        assert CppyyTestData.s_int      == -202
        assert c.s_int                  == -202
        assert c.s_uint                 ==  202
        assert CppyyTestData.s_uint     ==  202
        assert CppyyTestData.s_long     == -pylong(303)
        assert c.s_long                 == -pylong(303)
        assert c.s_ulong                ==  pylong(303)
        assert CppyyTestData.s_ulong    ==  pylong(303)
        assert CppyyTestData.s_llong    == -pylong(404)
        assert c.s_llong                == -pylong(404)
        assert c.s_ullong               ==  pylong(404)
        assert CppyyTestData.s_ullong   ==  pylong(404)

        # floating point types
        assert round(CppyyTestData.s_float   + 606., 5) == 0
        assert round(c.s_float               + 606., 5) == 0
        assert round(CppyyTestData.s_double  + 707., 8) == 0
        assert round(c.s_double              + 707., 8) == 0
        assert round(CppyyTestData.s_ldouble + 808., 8) == 0
        assert round(c.s_ldouble             + 808., 8) == 0

        c.__destruct__()

    def test05_class_data_write_access(self):
        """Test write access to class public data and verify values"""

        import cppyy, sys
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        # char types
        CppyyTestData.s_char             = 'a'
        assert c.s_char                 == 'a'
        c.s_char                         = 'b'
        assert CppyyTestData.s_char     == 'b'
        CppyyTestData.s_uchar            = 'c'
        assert c.s_uchar                == 'c'
        c.s_uchar                        = 'd'
        assert CppyyTestData.s_uchar    == 'd'
        raises(ValueError, setattr, CppyyTestData, 's_uchar', -1)
        raises(ValueError, setattr, c,             's_uchar', -1)
        CppyyTestData.s_wchar            = u'K'
        assert c.s_wchar                == u'K'
        c.s_wchar                        = u'L'
        assert CppyyTestData.s_wchar    == u'L'
        CppyyTestData.s_char16           = u'\u00df'
        assert c.s_char16               == u'\u00df'
        c.s_char16                       = u'\u00ef'
        assert CppyyTestData.s_char16   == u'\u00ef'
        CppyyTestData.s_char32           = u'\u00df'
        assert c.s_char32               == u'\u00df'
        c.s_char32                       = u'\u00ef'
        assert CppyyTestData.s_char32   == u'\u00ef'

        # integer types
        if self.has_byte:
            c.s_byte                     =   66
            assert CppyyTestData.s_byte ==   66
            CppyyTestData.s_byte         =   66
            assert c.s_byte             ==   66
        c.s_short                        = -102
        assert CppyyTestData.s_short    == -102
        CppyyTestData.s_short            = -203
        assert c.s_short                == -203
        c.s_ushort                       =  127
        assert CppyyTestData.s_ushort   ==  127
        CppyyTestData.s_ushort           =  227
        assert c.s_ushort               ==  227
        CppyyTestData.s_int              = -234
        assert c.s_int                  == -234
        c.s_int                          = -321
        assert CppyyTestData.s_int      == -321
        CppyyTestData.s_uint             = 1234
        assert c.s_uint                 == 1234
        c.s_uint                         = 4321
        assert CppyyTestData.s_uint     == 4321
        raises(ValueError, setattr, c,             's_uint', -1)
        raises(ValueError, setattr, CppyyTestData, 's_uint', -1)
        CppyyTestData.s_long             = -pylong(87)
        assert c.s_long                 == -pylong(87)
        c.s_long                         = pylong(876)
        assert CppyyTestData.s_long     == pylong(876)
        CppyyTestData.s_ulong            = pylong(876)
        assert c.s_ulong                == pylong(876)
        c.s_ulong                        = pylong(678)
        assert CppyyTestData.s_ulong    == pylong(678)
        raises(ValueError, setattr, CppyyTestData, 's_ulong', -1)
        raises(ValueError, setattr, c,             's_ulong', -1)

        # floating point types
        CppyyTestData.s_float                      = -3.1415
        assert round(c.s_float, 5)                == -3.1415
        c.s_float                                  =  3.1415
        assert round(CppyyTestData.s_float, 5)    ==  3.1415
        import math
        c.s_double                                 = -math.pi
        assert CppyyTestData.s_double             == -math.pi
        CppyyTestData.s_double                     =  math.pi
        assert c.s_double                         ==  math.pi
        c.s_ldouble                                = -math.pi
        assert CppyyTestData.s_ldouble            == -math.pi
        CppyyTestData.s_ldouble                    =  math.pi
        assert c.s_ldouble                        ==  math.pi

        c.__destruct__()

    def test06_range_access(self):
        """Test the ranges of integer types"""

        import cppyy, sys
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        # TODO: should these be TypeErrors, or should char/bool raise
        #       ValueErrors? In any case, consistency is needed ...
        raises(ValueError, setattr, c, 'm_uint',  -1)
        raises(ValueError, setattr, c, 'm_ulong', -1)

        c.__destruct__()

    @mark.xfail()
    def test07_type_conversions(self):
        """Test conversions between builtin types"""

        import cppyy, sys
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        c.m_double = -1
        assert round(c.m_double + 1.0, 8) == 0

        raises(TypeError, setattr, c.m_double,  'c')
        raises(TypeError, setattr, c.m_int,     -1.)
        raises(TypeError, setattr, c.m_int,      1.)
        raises(TypeError, setattr, c.m_long,     3.14)
        raises(TypeError, setattr, c.m_ulong,    3.14)
        raises(TypeError, setattr, c.m_llong,    3.14)
        raises(TypeError, setattr, c.m_ullong,   3.14)

        raises(TypeError, c.set_int,             3.14)
        raises(TypeError, c.set_long,            3.14)
        raises(TypeError, c.set_ulong,           3.14)
        raises(TypeError, c.set_llong,           3.14)
        raises(TypeError, c.set_ullong,          3.14)

        raises(TypeError, c.set_int_cr,          3.14)
        raises(TypeError, c.set_long_cr,         3.14)
        raises(TypeError, c.set_ulong_cr,        3.14)
        raises(TypeError, c.set_llong_cr,        3.14)
        raises(TypeError, c.set_ullong_cr,       3.14)

        c.__destruct__()

    def test08_global_builtin_type(self):
        """Test access to a global builtin type"""

        import cppyy
        gbl = cppyy.gbl

        assert gbl.g_int == gbl.get_global_int()

        gbl.set_global_int(32)
        assert gbl.get_global_int() == 32
        assert gbl.g_int == 32

        gbl.g_int = 22
        assert gbl.get_global_int() == 22
        assert gbl.g_int == 22

      # if setting before checking, the change must be reflected on the C++ side
        gbl.g_some_global_string = "Python"
        assert gbl.get_some_global_string() == "Python"
        assert gbl.g_some_global_string2 == "C++"
        assert gbl.get_some_global_string2() == "C++"
        gbl.g_some_global_string2 = "Python"
        assert gbl.get_some_global_string2() == "Python"

        assert gbl.g_some_global_string16 == u'z\u00df\u6c34'
        assert gbl.g_some_global_string32 == u'z\u00df\u6c34\U0001f34c'

        NS = gbl.SomeStaticDataNS
        NS.s_some_static_string = "Python"
        assert NS.get_some_static_string() == "Python"
        assert NS.s_some_static_string2 == "C++"
        assert NS.get_some_static_string2() == "C++"
        NS.s_some_static_string2 = "Python"
        assert NS.get_some_static_string2() == "Python"

     # verify that non-C++ data can still be set
        gbl.g_python_only = "Python"
        assert gbl.g_python_only == "Python"

        NS.s_python_only = "Python"
        assert NS.s_python_only == "Python"

    def test08a_global_object(self):
        """Test access to global objects by value"""

        import cppyy
        gbl = cppyy.gbl

        assert gbl.gData.fData == 5.

    def test09_global_ptr(self):
        """Test access of global objects through a pointer"""

        import cppyy
        gbl = cppyy.gbl

        with raises(ReferenceError):
            gbl.g_pod.m_int

        c = gbl.CppyyTestPod()
        c.m_int = 42
        c.m_double = 3.14

        gbl.set_global_pod(c)
        assert gbl.is_global_pod(c)
        assert gbl.g_pod.m_int == 42
        assert gbl.g_pod.m_double == 3.14

        d = gbl.get_global_pod()
        assert gbl.is_global_pod(d)
        assert c == d
        assert id(c) == id(d)

        e = gbl.CppyyTestPod()
        e.m_int = 43
        e.m_double = 2.14

        gbl.g_pod = e
        assert gbl.is_global_pod(e)
        assert gbl.g_pod.m_int == 43
        assert gbl.g_pod.m_double == 2.14

    def test10_enum(self):
        """Test access to enums"""

        import cppyy
        gbl = cppyy.gbl

        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        # test that the enum is accessible as a type
        assert CppyyTestData.EWhat

        assert CppyyTestData.kNothing   ==   6
        assert CppyyTestData.kSomething == 111
        assert CppyyTestData.kLots      ==  42

        assert CppyyTestData.EWhat(CppyyTestData.kNothing) == CppyyTestData.kNothing
        assert CppyyTestData.EWhat(6) == CppyyTestData.kNothing
        # TODO: only allow instantiations with correct values (C++11)

        assert c.get_enum() == CppyyTestData.kNothing
        assert c.m_enum == CppyyTestData.kNothing

        c.m_enum = CppyyTestData.kSomething
        assert c.get_enum() == CppyyTestData.kSomething
        assert c.m_enum == CppyyTestData.kSomething

        c.set_enum(CppyyTestData.kLots)
        assert c.get_enum() == CppyyTestData.kLots
        assert c.m_enum == CppyyTestData.kLots

        assert c.s_enum == CppyyTestData.s_enum
        assert c.s_enum == CppyyTestData.kNothing
        assert CppyyTestData.s_enum == CppyyTestData.kNothing

        c.s_enum = CppyyTestData.kSomething
        assert c.s_enum == CppyyTestData.s_enum
        assert c.s_enum == CppyyTestData.kSomething
        assert CppyyTestData.s_enum == CppyyTestData.kSomething

        # global enums
        assert gbl.EFruit          # test type accessible
        assert gbl.kApple  == 78
        assert gbl.kBanana == 29
        assert gbl.kCitrus == 34
        assert gbl.EFruit.__name__     == 'EFruit'
        assert gbl.EFruit.__cpp_name__ == 'EFruit'

        assert gbl.EFruit.kApple  == 78
        assert gbl.EFruit.kBanana == 29
        assert gbl.EFruit.kCitrus == 34

        assert gbl.NamedClassEnum.E1 == 42
        assert gbl.NamedClassEnum.__name__     == 'NamedClassEnum'
        assert gbl.NamedClassEnum.__cpp_name__ == 'NamedClassEnum'

        assert gbl.EnumSpace.E
        assert gbl.EnumSpace.EnumClass.E1 == -1   # anonymous
        assert gbl.EnumSpace.EnumClass.E2 == -1   # named type

        assert gbl.EnumSpace.NamedClassEnum.E1 == -42
        assert gbl.EnumSpace.NamedClassEnum.__name__     == 'NamedClassEnum'
        assert gbl.EnumSpace.NamedClassEnum.__cpp_name__ == 'EnumSpace::NamedClassEnum'

        raises(TypeError, setattr, gbl.EFruit, 'kBanana', 42)

        assert gbl.g_enum == gbl.EFruit.kBanana
        gbl.g_enum = gbl.EFruit.kCitrus
        assert gbl.g_enum == gbl.EFruit.kCitrus

        # typedef enum
        assert gbl.EnumSpace.letter_code
        assert gbl.EnumSpace.AA == 1
        assert gbl.EnumSpace.BB == 2

    @mark.xfail()
    def test11_typed_enums(self):
        """Determine correct types of enums"""

        import cppyy

        cppyy.cppdef("""\
        namespace TrueEnumTypes {
        class Test {
            enum nums { ZERO = 0, ONE = 1 };
            enum dir : char { left = 'l', right = 'r' };
            enum rank : long { FIRST = 1, SECOND };
            enum vraioufaux : bool { faux = false, vrai = true };
        }; }""")

        sc = cppyy.gbl.TrueEnumTypes.Test

        assert sc.nums.ZERO == 0
        assert sc.nums.ONE  == 1
        assert type(sc.nums.ZERO) == sc.nums
        assert isinstance(sc.nums.ZERO, int)
        assert 'int' in repr(sc.nums.ZERO)
        assert str(sc.nums.ZERO) == '0'

        assert sc.dir.left  == 'l'
        assert sc.dir.right == 'r'
        assert type(sc.dir.left) == sc.dir
        assert isinstance(sc.dir.left, str)
        assert 'char' in repr(sc.dir.left)
        assert str(sc.dir.left) == "'l'"

        assert sc.rank.FIRST  == 1
        assert sc.rank.SECOND == 2
        assert type(sc.rank.FIRST) == sc.rank
        assert isinstance(sc.rank.FIRST, pylong)
        assert 'long' in repr(sc.rank.FIRST)
        assert str(sc.rank.FIRST) == '1' or str(sc.rank.FIRST) == '1L'

        assert sc.vraioufaux.faux == False
        assert sc.vraioufaux.vrai == True
        assert type(sc.vraioufaux.faux) == bool  # no bool as base class
        assert isinstance(sc.vraioufaux.faux, bool)

    @mark.xfail()
    def test12_enum_scopes(self):
        """Enum accessibility and scopes"""

        import cppyy

        cppyy.cppdef("""\
        enum              { g_one   = 1, g_two  = 2 };
        enum       GEnum1 { g_three = 3, g_four = 4 };
        enum class GEnum2 { g_five  = 5, g_six  = 6 };

        namespace EnumScopes {
            enum              { n_one   = 1, n_two  = 2 };
            enum       NEnum1 { n_three = 3, n_four = 4 };
            enum class NEnum2 { n_five  = 5, n_six  = 6 };
        }

        class EnumClass {
        public:
            enum              { c_one   = 1, c_two  = 2 };
            enum       CEnum1 { c_three = 3, c_four = 4 };
            enum class CEnum2 { c_five  = 5, c_six  = 6 };
        }; """)

        gn = cppyy.gbl
        assert not hasattr(gn, 'n_one')
        assert not hasattr(gn, 'c_one')
        assert gn.g_two  == 2
        assert gn.g_four == 4
        assert gn.GEnum1.g_three == 3
        assert gn.GEnum1.g_three == gn.g_three
        assert type(gn.GEnum1.g_three) == type(gn.g_three)
        assert not hasattr(gn, 'g_five')
        assert gn.GEnum2.g_six == 6

        ns = cppyy.gbl.EnumScopes
        assert not hasattr(ns, 'g_one')
        assert not hasattr(ns, 'c_one')
        assert ns.n_two  == 2
        assert ns.n_four == 4
        assert ns.NEnum1.n_three == 3
        assert ns.NEnum1.n_three == ns.n_three
        assert type(ns.NEnum1.n_three) == type(ns.n_three)
        assert not hasattr(ns, 'n_five')
        assert ns.NEnum2.n_six == 6

        cl = cppyy.gbl.EnumClass
        assert not hasattr(cl, 'g_one')
        assert not hasattr(cl, 'n_one')
        assert cl.c_two  == 2
        assert cl.c_four == 4
        assert cl.CEnum1.c_three == 3
        assert cl.CEnum1.c_three == cl.c_three
        assert type(cl.CEnum1.c_three) == type(cl.c_three)
        assert not hasattr(cl, 'c_five')
        assert cl.CEnum2.c_six == 6

    def test13_string_passing(self):
        """Test passing/returning of a const char*"""

        import cppyy
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert c.get_valid_string('aap') == 'aap'
        assert c.get_invalid_string() == ''

        assert c.get_valid_wstring(u'aap') == u'aap'
        assert c.get_invalid_wstring() == u''

        assert c.get_valid_string16(u'z\u00df\u6c34') == u'z\u00df\u6c34'
        assert c.get_invalid_string16() == u''

        assert c.get_valid_string32(u'z\u00df\u6c34\U0001f34c') == u'z\u00df\u6c34\U0001f34c'
        assert c.get_invalid_string32() == u''

    def test14_copy_constructor(self):
        """Test copy constructor"""

        import cppyy
        FourVector = cppyy.gbl.FourVector

        t1 = FourVector(1., 2., 3., -4.)
        t2 = FourVector(0., 0., 0.,  0.)
        t3 = FourVector(t1)

        assert t1 == t3
        assert t1 != t2

        for i in range(4):
            assert t1[i] == t3[i]

    def test15_object_returns(self):
        """Test access to and return of PODs"""

        import cppyy

        c = cppyy.gbl.CppyyTestData()

        assert c.m_pod.m_int == 888
        assert c.m_pod.m_double == 3.14

        pod = c.get_pod_val()
        assert pod.m_int == 888
        assert pod.m_double == 3.14

        assert c.get_pod_val_ptr().m_int == 888
        assert c.get_pod_val_ptr().m_double == 3.14
        c.get_pod_val_ptr().m_int = 777
        assert c.get_pod_val_ptr().m_int == 777

        assert c.get_pod_val_ref().m_int == 777
        assert c.get_pod_val_ref().m_double == 3.14
        c.get_pod_val_ref().m_int = 666
        assert c.get_pod_val_ref().m_int == 666

        assert c.get_pod_ptrref().m_int == 666
        assert c.get_pod_ptrref().m_double == 3.14

    def test16_object_arguments(self):
        """Test setting and returning of a POD through arguments"""

        import cppyy

        c = cppyy.gbl.CppyyTestData()
        assert c.m_pod.m_int == 888
        assert c.m_pod.m_double == 3.14

        p = cppyy.gbl.CppyyTestPod()
        p.m_int = 123
        assert p.m_int == 123
        p.m_double = 321.
        assert p.m_double == 321.

        c.set_pod_val(p)
        assert c.m_pod.m_int == 123
        assert c.m_pod.m_double == 321.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_ptr_in(p)
        assert c.m_pod.m_int == 123
        assert c.m_pod.m_double == 321.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_ptr_out(p)
        assert p.m_int == 888
        assert p.m_double == 3.14

        p.m_int = 555
        p.m_double = 666.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_ref(p)
        assert c.m_pod.m_int == 555
        assert c.m_pod.m_double == 666.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_ptrptr_in(p)
        assert c.m_pod.m_int == 555
        assert c.m_pod.m_double == 666.
        assert p.m_int == 555
        assert p.m_double == 666.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_void_ptrptr_in(p)
        assert c.m_pod.m_int == 555
        assert c.m_pod.m_double == 666.
        assert p.m_int == 555
        assert p.m_double == 666.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_ptrptr_out(p)
        assert c.m_pod.m_int == 888
        assert c.m_pod.m_double == 3.14
        assert p.m_int == 888
        assert p.m_double == 3.14

        p.m_int = 777
        p.m_double = 888.

        c = cppyy.gbl.CppyyTestData()
        c.set_pod_void_ptrptr_out(p)
        assert c.m_pod.m_int == 888
        assert c.m_pod.m_double == 3.14
        assert p.m_int == 888
        assert p.m_double == 3.14

    def test17_nullptr_passing(self):
        """Integer 0 ('NULL') and nullptr allowed to pass through instance*"""

        import cppyy

        for o in (0, cppyy.nullptr):
            c = cppyy.gbl.CppyyTestData()
            assert c.m_pod.m_int == 888
            assert c.m_pod.m_double == 3.14
            assert not not c.m_ppod

            c.set_pod_ptr(o)
            assert not c.m_ppod
            assert not c.get_pod_ptr()

    def test18_respect_privacy(self):
        """Test that privacy settings are respected"""

        import cppyy
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        assert isinstance(c, CppyyTestData)

        raises(AttributeError, getattr, c, 'm_owns_arrays')

        c.__destruct__()

    def test19_object_and_pointer_comparisons(self):
        """Verify object and pointer comparisons"""

        import cppyy
        gbl = cppyy.gbl

        c1 = cppyy.bind_object(0, gbl.CppyyTestData)
        assert c1 == None
        assert None == c1

        c2 = cppyy.bind_object(0, gbl.CppyyTestData)
        assert c1 == c2
        assert c2 == c1

        # FourVector overrides operator==
        l1 = cppyy.bind_object(0, gbl.FourVector)
        assert l1 == None
        assert None == l1

        assert c1 != l1
        assert l1 != c1

        l2 = cppyy.bind_object(0, gbl.FourVector)
        assert l1 == l2
        assert l2 == l1

        l3 = gbl.FourVector(1, 2, 3, 4)
        l4 = gbl.FourVector(1, 2, 3, 4)
        l5 = gbl.FourVector(4, 3, 2, 1)
        assert l3 == l4
        assert l4 == l3

        assert l3 != None                 # like this to ensure __ne__ is called
        assert None != l3                 # id.
        assert l3 != l5
        assert l5 != l3

    def test20_object_comparisons_with_cpp__eq__(self):
        """Comparisons with C++ providing __eq__/__ne__"""

        import cppyy

        cppyy.cppdef("""
        namespace MoreComparisons {
        struct Comparable1 {
            Comparable1(int i) : fInt(i) {}
            int fInt;
            static bool __eq__(const Comparable1& self, const Comparable1& other) {
                return self.fInt == other.fInt;
            }
            static bool __ne__(const Comparable1& self, const Comparable1& other) {
                return self.fInt != other.fInt;
            }
        };

        struct Comparable2 {
            Comparable2(int i) : fInt(i) {}
            int fInt;
            bool __eq__(const Comparable2& other) {
                return fInt == other.fInt;
            }
            bool __ne__(const Comparable2& other) {
                return fInt != other.fInt;
            }
        }; }""")

        ns = cppyy.gbl.MoreComparisons

        c1_1 = ns.Comparable1(42)
        c1_2 = ns.Comparable1(42)
        c1_3 = ns.Comparable1(43)

        assert     ns.Comparable1.__dict__['__eq__'](c1_1, c1_2)
        assert not ns.Comparable1.__dict__['__eq__'](c1_1, c1_3)
        assert not ns.Comparable1.__dict__['__ne__'](c1_1, c1_2)
        assert     ns.Comparable1.__dict__['__ne__'](c1_1, c1_3)

      # the following works as a side-effect of a workaround for vector calls and
      # it is probably preferable to have it working, so leave the discrepancy for
      # now: python's aggressive end-of-life schedule will catch up soon enough
        if 0x3080000 <= sys.hexversion:
            assert     c1_1 == c1_2
            assert not c1_1 != c1_2
        else:
            with raises(TypeError):
                c1_1 == c1_2
            with raises(TypeError):
                c1_1 != c1_2

        c2_1 = ns.Comparable2(27)
        c2_2 = ns.Comparable2(27)
        c2_3 = ns.Comparable2(28)

        assert     c2_1 == c2_1
        assert     c2_1 == c2_2
        assert not c2_1 == c2_3
        assert not c2_1 != c2_1
        assert not c2_1 != c2_2
        assert     c2_1 != c2_3

    def test21_object_validity(self):
        """Test object validity checking"""

        from cppyy import gbl

        d = gbl.CppyyTestPod()

        assert d
        assert not not d

        d2 = gbl.get_null_pod()

        assert not d2

    @mark.xfail()
    def test22_buffer_shapes(self):
        """Correctness of declared buffer shapes"""

        import cppyy

        cppyy.cppdef("""\
        namespace ShapeTester {
        enum Enum{One, Two, Three};

        template<typename T>
        struct Foo {
          T a[5];
          T aa[5][4];
          T aaa[5][4][3];
        }; }""")

        ns = cppyy.gbl.ShapeTester

        for dtype in ["int", "double", "long", "bool", "char", "void*", "ShapeTester::Enum"]:
            foo = ns.Foo[dtype]()
            if dtype != 'char':
                assert foo.a.shape   == (5,)
            else:
              # TODO: verify the following is for historic reasons and should be modified
              # once bug #344 (bitbucket) is fixed
                assert len(foo.a) == 5
            assert foo.aa.shape  == (5, 4)
            assert foo.aaa.shape == (5, 4, 3)

    def test23_buffer_reshaping(self):
        """Test usage of buffer sizing"""

        import cppyy
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()
        byte_array_names = []
        if self.has_byte:
            byte_array_names = ['get_byte_array', 'get_byte_array2']
        for func in ['get_bool_array',   'get_bool_array2',
                     'get_uchar_array',  'get_uchar_array2',
                     'get_int8_array',   'get_int8_array2',
                     'get_uint8_array',  'get_uint8_array2',
                     'get_short_array',  'get_short_array2',
                     'get_ushort_array', 'get_ushort_array2',
                     'get_int_array',    'get_int_array2',
                     'get_uint_array',   'get_uint_array2',
                     'get_long_array',   'get_long_array2',
                     'get_ulong_array',  'get_ulong_array2']+\
                     byte_array_names:
            arr = getattr(c, func)()
            arr.reshape((self.N,))
            assert len(arr) == self.N

            raises(ValueError, arr.reshape, (1, 2))
            assert len(arr) == self.N

            raises(TypeError, arr.reshape, 2*self.N)
            assert len(arr) == self.N

            l = list(arr)
            for i in range(self.N):
                assert arr[i] == l[i]

    def test24_voidp(self):
        """Test usage of void* data"""

        import cppyy
        CppyyTestData = cppyy.gbl.CppyyTestData

        c = CppyyTestData()

        assert not cppyy.nullptr

        assert c.s_voidp                is cppyy.nullptr
        assert CppyyTestData.s_voidp    is cppyy.nullptr

        assert c.m_voidp                is cppyy.nullptr
        assert c.get_voidp()            is cppyy.nullptr

        c2 = CppyyTestData()
        assert c2.m_voidp               is cppyy.nullptr
        c.set_voidp(c2.m_voidp)
        assert c.m_voidp                is cppyy.nullptr
        c.set_voidp(c2.get_voidp())
        assert c.m_voidp                is cppyy.nullptr
        c.set_voidp(cppyy.nullptr)
        assert c.m_voidp                is cppyy.nullptr

        c.set_voidp(c2)
        def address_equality_test(a, b):
            assert cppyy.addressof(a) == cppyy.addressof(b)
            b2 = cppyy.bind_object(a, CppyyTestData)
            assert b is b2    # memory regulator recycles
            b3 = cppyy.bind_object(cppyy.addressof(a), CppyyTestData)
            assert b is b3    # likewise

        address_equality_test(c.m_voidp, c2)
        address_equality_test(c.get_voidp(), c2)

        def null_test(null):
            c.m_voidp = null
            assert c.m_voidp is cppyy.nullptr
        map(null_test, [0, cppyy.nullptr])

        c.m_voidp = c2
        address_equality_test(c.m_voidp,     c2)
        address_equality_test(c.get_voidp(), c2)

        c.s_voidp = c2
        address_equality_test(c.s_voidp, c2)

        cppyy.cppdef("""\
        namespace VoidP {
        void* vvv[3][5][7];
        struct Init {
          Init() {
            for (size_t i = 0; i < 3; ++i) {
              for (size_t j = 0; j < 5; ++j) {
                for (size_t k = 0; k < 7; ++k)
                  vvv[i][j][k] = (void*)(i+j+k);
              }
            }
          }
        } _init; }""")

        ns = cppyy.gbl.VoidP
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    assert int(ns.vvv[i,j,k]) == i+j+k

    @mark.skip()
    def test25_byte_arrays(self):
        """Usage of unsigned char* as byte array and std::byte*"""

        import array, cppyy, ctypes

        buf = b'123456789'
        total = 0
        for c in buf:
            try:
                total += ord(c)        # p2
            except TypeError:
                total += c             # p3

        def run(self, f, buf, total):

            # The following create a unique type for fixed-size C arrays: ctypes.c_char_Array_9
            # and neither inherits from a non-sized type nor implements the buffer interface.
            # As such, it can't be handled. TODO?
            #pbuf = ctypes.create_string_buffer(len(buf), buf)
            #assert f(pbuf, len(buf)) == total

            pbuf = array.array('B', buf)
            assert f(pbuf, len(buf)) == total

            pbuf = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
            assert f(pbuf, len(buf)) == total

            pbuf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte * len(buf)))[0]
            assert f(pbuf, len(buf)) == total

        run(self, cppyy.gbl.sum_uc_data, buf, total)

        if self.has_byte:
            run(self, cppyy.gbl.sum_byte_data, buf, total)

    @mark.xfail(run=not IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test26_function_pointers(self):
        """Function pointer passing"""

        import cppyy

        fi1 = cppyy.gbl.sum_of_int1
        fi2 = cppyy.gbl.sum_of_int2
        fd  = cppyy.gbl.sum_of_double
        fdd = cppyy.gbl.call_double_double

        assert 5 == fi1(2, 3)
        assert 5. == fd(5., 0.)

        raises(TypeError, fdd, fi1, 2, 3)

        assert  5. == fdd(fd, 5., 0.)
        assert -1. == fdd(cppyy.nullptr, 5., 0.)

        fip = cppyy.gbl.sum_of_int_ptr
        assert 5 == fip(2, 3)

        cppyy.gbl.sum_of_int_ptr = cppyy.gbl.sum_of_int2
        assert 7 == cppyy.gbl.sum_of_int_ptr(2, 3)

        cppyy.gbl.sum_of_int_ptr = cppyy.nullptr
        assert not cppyy.gbl.sum_of_int_ptr
        with raises(cppyy.gbl.std.bad_function_call):
            cppyy.gbl.sum_of_int_ptr(2, 3)
        with raises(AttributeError):
            cppyy.gbl.sim_of_int_ptr   # incorrect spelling

        cppyy.gbl.sum_of_int_ptr = cppyy.gbl.sum_of_int1
        assert fip is cppyy.gbl.sum_of_int_ptr   # b/c cached

        o = cppyy.gbl.sum_of_int_struct()
        o.sum_of_int_ptr = cppyy.gbl.sum_of_int1
        assert 5 == o.sum_of_int_ptr(2, 3)

        o.sum_of_int_ptr = cppyy.gbl.sum_of_int2
        assert 7 == o.sum_of_int_ptr(2, 3)

        def sum_in_python(i1, i2):
            return i1-i2
        cppyy.gbl.sum_of_int_ptr = sum_in_python
        assert 1 == cppyy.gbl.call_sum_of_int(3, 2)

        def sum_in_python(i1, i2, i3):
            return i1+i2+i3
        cppyy.gbl.sum_of_int_ptr = sum_in_python
        with raises(TypeError):
            cppyy.gbl.call_sum_of_int(3, 2)

        cppyy.cppdef(r"""\
        namespace FuncPtrReturn {
            typedef std::string (*func_t)(void);
            std::string hello() { return "Hello, World!"; }
            func_t foo() { return hello; }
        }""")

        ns = cppyy.gbl.FuncPtrReturn
        assert ns.foo()() == "Hello, World!"

    @mark.xfail(run=False, condition=IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test27_callable_passing(self):
        """Passing callables through function pointers"""

        import cppyy, gc

        fdd = cppyy.gbl.call_double_double
        fii = cppyy.gbl.call_int_int
        fv  = cppyy.gbl.call_void
        fri = cppyy.gbl.call_refi
        frl = cppyy.gbl.call_refl
        frd = cppyy.gbl.call_refd

        assert 'call_double_double' in str(fdd)
        assert 'call_refd' in str(frd)

        def pyf(arg0, arg1):
            return arg0+arg1

        assert type(fdd(pyf, 2, 3)) == float
        assert fdd(pyf, 2, 3) == 5.

        assert type(fii(pyf, 2, 3)) == int
        assert fii(pyf, 2, 3) == 5

        def pyf(arg0, arg1):
            return arg0*arg1

        assert fdd(pyf, 2, 3) == 6.
        assert fii(pyf, 2, 3) == 6

        # call of void function
        global retval
        retval = None
        def voidf(i):
            global retval
            retval = i

        assert retval is None
        assert fv(voidf, 5) == None
        assert retval == 5

        # call of function with reference argument
        def reff(ref):
            ref.value = 5
        assert fri(reff) == 5
        assert frl(reff) == pylong(5)
        assert frd(reff) == 5.

        # callable that does not accept weak-ref
        import math
        assert fdd(math.atan2, 0, 3.) == 0.

        # error testing
        raises(TypeError, fii, None, 2, 3)

        def pyf(arg0, arg1):
            return arg0/arg1

        raises(ZeroDivisionError, fii, pyf, 1, 0)

        def pyd(arg0, arg1):
            return arg0*arg1
        c = cppyy.gbl.StoreCallable(pyd)
        assert c(3, 3) == 9.

        c.set_callable(lambda x, y: x*y)
        assert c(3, 3) == 9.           # life line protected

        c.__dict__.clear()             # destroys life lines
        gc.collect()
        raises(TypeError, c, 3, 3) # lambda gone out of scope

    @mark.xfail(run=False, condition=IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test28_callable_through_function_passing(self):
        """Passing callables through std::function"""

        import cppyy, gc

        fdd = cppyy.gbl.call_double_double_sf
        fii = cppyy.gbl.call_int_int_sf
        fv  = cppyy.gbl.call_void_sf
        fri = cppyy.gbl.call_refi_sf
        frl = cppyy.gbl.call_refl_sf
        frd = cppyy.gbl.call_refd_sf

        assert 'call_double_double_sf' in str(fdd)
        assert 'call_refd_sf' in str(frd)

        def pyf(arg0, arg1):
            return arg0+arg1

        assert type(fdd(pyf, 2, 3)) == float
        assert fdd(pyf, 2, 3) == 5.

        assert type(fii(pyf, 2, 3)) == int
        assert fii(pyf, 2, 3) == 5

        def pyf(arg0, arg1):
            return arg0*arg1

        assert fdd(pyf, 2, 3) == 6.
        assert fii(pyf, 2, 3) == 6

        # call of void function
        global retval
        retval = None
        def voidf(i):
            global retval
            retval = i

        assert retval is None
        assert fv(voidf, 5) == None
        assert retval == 5

        # call of function with reference argument
        def reff(ref):
            ref.value = 5
        assert fri(reff) == 5
        assert frl(reff) == pylong(5)
        assert frd(reff) == 5.

        # callable that does not accept weak-ref
        import math
        assert fdd(math.atan2, 0, 3.) == 0.

        # error testing
        raises(TypeError, fii, None, 2, 3)

        def pyf(arg0, arg1):
            return arg0/arg1

        raises(ZeroDivisionError, fii, pyf, 1, 0)

        def pyd(arg0, arg1):
            return arg0*arg1
        c = cppyy.gbl.StoreCallable(pyd)
        assert c(3, 3) == 9.

        c.set_callable(lambda x, y: x*y)
        assert c(3, 3) == 9.           # life line protected

        c.__dict__.clear()             # destroys life lines
        gc.collect()
        raises(TypeError, c, 3, 3) # lambda gone out of scope

    def test29_std_function_life_lines(self):
        """Life lines to std::function data members"""

        import cppyy, gc

        cppyy.cppdef("""\
        namespace BoundMethod2StdFunction {
        class Base {
        public:
            virtual ~Base() {}

            std::function<std::string()> execute;
            std::string do_execute() {
                return execute();
            }
        }; } """)

        ns = cppyy.gbl.BoundMethod2StdFunction

        class Derived(ns.Base):
            def __init__(self):
                super(Derived, self).__init__()
                self.execute = self.xyz

            def xyz(self):
                return "xyz"

        d = Derived()
        assert d.do_execute() == "xyz"
        assert d.do_execute() == "xyz"

        gc.collect()
        assert d.do_execute() == "xyz"

        d.execute = d.xyz
        assert d.do_execute() == "xyz"

    def test30_multi_dim_arrays_of_builtins(test):
        """Multi-dim arrays of builtins"""

        import cppyy, ctypes

        cppyy.cppdef("""
        template<class T, int nlayers>
        struct MultiDimTest {
            T* layers[nlayers];

            MultiDimTest(int width, int height) {
                for (int i=0; i<nlayers; ++i) {
                    layers[i] = new T[width*height];
                    for (int j=0; j<width*height; ++j)
                        layers[i][j] = j*2;
                }
            }
            ~MultiDimTest() { for (int i=0; i<nlayers; ++i) delete[] layers[i]; }
        };
        """)

        from cppyy.gbl import MultiDimTest

        nlayers = 3; width = 8; height = 4
        for (cpptype, ctype) in (('unsigned char', ctypes.c_ubyte),
                                 ('int',           ctypes.c_int),
                                 ('double',        ctypes.c_double)):
            m = MultiDimTest[cpptype, nlayers](width, height)

            for i in range(nlayers):
                buf = m.layers[i]
                p = (ctype * len(buf)).from_buffer(buf)
                assert [p[j] for j in range(width*height)] == [2*j for j in range(width*height)]

    @mark.xfail()
    def test31_anonymous_union(self):
        """Anonymous unions place there fields in the parent scope"""

        import cppyy

        cppyy.cppdef("""\
        namespace AnonUnion {
        struct Event1 {
            Event1() : num(1) { shrd.a = 5.; }
            int num;
            union SomeUnion {
                double a;
                int b;
            } shrd;
        };

        struct Event2 {
            Event2() : num(1) { shrd.a = 5.; }
            int num;
            union {
                double a;
                int b;
            } shrd;
        };

        struct Event3 {
            Event3(double d) : a(d) {}
            Event3(int i) : b(i) {}
            union {
                double a;
                int b;
            };
        };

        struct Event4 {
            Event4(int i) : num(1), b(i) {}
            Event4(double d) : num(2), a(d) {}
            int num;
            union {
                double a;
                int b;
            };
        };

        struct PointXYZI {
            PointXYZI() : intensity(5.) {}
            double x, y, z, i;
            union {
                int offset1;
                struct {
                   int offset2;
                   float intensity;
                };
                float data_c[4];
            };
        }; }""")

        # named union
        e = cppyy.gbl.AnonUnion.Event1()
        assert e.num == 1
        raises(AttributeError, getattr, e, 'a')
        raises(AttributeError, getattr, e, 'b')
        assert e.shrd.a == 5.

        # anonymous union, with field name
        e = cppyy.gbl.AnonUnion.Event2()
        assert e.num == 1
        raises(AttributeError, getattr, e, 'a')
        raises(AttributeError, getattr, e, 'b')
        assert e.shrd.a == 5.

        # anonymous union, no field name
        e = cppyy.gbl.AnonUnion.Event3(42)
        assert e.b == 42

        e = cppyy.gbl.AnonUnion.Event3(5.)
        assert e.a == 5.

        # anonymous union, no field name, with offset
        e = cppyy.gbl.AnonUnion.Event4(42)
        assert e.num == 1
        assert e.b == 42

        e = cppyy.gbl.AnonUnion.Event4(5.)
        assert e.num == 2
        assert e.a == 5.

        # anonumous struct in anonymous union
        p = cppyy.gbl.AnonUnion.PointXYZI()
        assert type(p.x) == float
        assert type(p.data_c[0]) == float
        assert p.intensity == 5.

    @mark.xfail()
    def test32_anonymous_struct(self):
        """Anonymous struct creates an unnamed type"""

        import cppyy

        cppyy.cppdef("""\
        namespace AnonStruct {
        class Foo1 {
        public:
            Foo1() { bar.x = 5; }
            struct { int x; } bar;
        };

        class Foo2 {
        public:
            Foo2() { bar.x = 5; baz.x = 7; }
            struct { int x; } bar;
            struct { int x; } baz;
        };

        typedef struct {
          struct {
            struct {
              struct {
                struct {
                  struct {
                    const char* (*foo)(const char* s);
                  } kmmdemo;
                } justamouse;
              } com;
            } root;
          } kotlin;
        } libuntitled1_ExportedSymbols;

        } """)

        ns = cppyy.gbl.AnonStruct

        foo = ns.Foo1()
        assert foo.bar.x == 5
        assert not hasattr(foo.bar, 'bar')

        foo = ns.Foo2()
        assert foo.bar.x == 5
        assert foo.baz.x == 7

        assert 'foo' in dir(ns.libuntitled1_ExportedSymbols().kotlin.root.com.justamouse.kmmdemo)

    @mark.xfail()
    def test33_pointer_to_array(self):
        """Usability of pointer to array"""

        import cppyy

        AoS = cppyy.gbl.ArrayOfStruct

        bar = AoS.Bar1()
        assert bar.fArr[0].fVal == 42
        assert bar.fArr[1].fVal == 13

        bar = AoS.Bar2(4)
        for i in range(4):
            assert bar.fArr[i].fVal == 2*i

        cppyy.cppdef("""
        namespace ArrayOfStruct {
        union Bar3 {         // not supported in dictionary
            Foo fArr[];      // clang only
            int fBuf;        // to allow indexing fArr w/o crashing
        };
        }""")

        bar = AoS.Bar3()
        assert cppyy.sizeof(AoS.Bar3) >= cppyy.sizeof(AoS.Foo)
        arr = bar.fArr
        arr.size = 1
        for f in arr:
            assert type(f) == AoS.Foo
        assert type(bar.fArr[0]) == AoS.Foo

    def test34_object_pointers(self):
        """Read/write access to objects through pointers"""

        import cppyy

        c = cppyy.gbl.CppyyTestData()

        assert cppyy.gbl.CppyyTestData.s_strv == "Hello"
        assert c.s_strv                       == "Hello"
        assert not cppyy.gbl.CppyyTestData.s_strp
        assert not c.s_strp

        c.s_strv                               = "World"
        assert cppyy.gbl.CppyyTestData.s_strv == "World"

      # assign on nullptr is a pointer copy
        sn = cppyy.gbl.std.string("aap")
        cppyy.gbl.CppyyTestData.s_strp = sn
        assert c.s_strp               == "aap"

      # assign onto the existing object
        cppyy.gbl.CppyyTestData.s_strp.__assign__(cppyy.gbl.std.string("noot"))
        assert c.s_strp               == "noot"
        assert sn                     == "noot"  # set through pointer

    def test35_restrict(self):
        """Strip __restrict keyword from use"""

        import cppyy

        cppyy.cppdef("std::string restrict_call(const char*__restrict s) { return s; }")

        assert cppyy.gbl.restrict_call("aap") == "aap"

    def test36_legacy_matrix(self):
        """Handling of legacy matrix"""

        import cppyy

        cppyy.cppdef("""\
        namespace Pointer2D {
        int** g_matrix;

        int** create_matrix(int n, int m) {
            int** mat = (int**)malloc(n*sizeof(int*));
            int* arr = (int*)malloc(n*m*sizeof(int));
            for (int i = 0; i < n; ++i) {
                mat[i] = arr + i*m;
                for(int j = 0; j < m; ++j) {
                    mat[i][j] = 13+i*n+j;
                }
            }
            g_matrix = mat;
            return mat;
        }

        bool destroy_matrix(int** mat, int n, int m) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (mat[i][j] != 13+i*n+j)
                        return false;
                }
            }
            g_matrix = nullptr;
            free(mat[0]);
            free(mat);
            return true;
        } }""")

        ns = cppyy.gbl.Pointer2D;

        N, M = 2, 3
        m = ns.create_matrix(N, M)
        g = ns.g_matrix;

        for i in range(N):
            for j in range(M):
                assert m[i][j] == 13+i*N+j
                assert g[i][j] == 13+i*N+j

        assert ns.destroy_matrix(m, N, M)

        m = ns.create_matrix(N, M)
        assert ns.destroy_matrix(ns.g_matrix, N, M)

    def test37_legacy_matrix_of_structs(self):
        """Handling of legacy matrix of structs"""

        import cppyy

        cppyy.cppdef("""\
        namespace StructPointer2D {
        typedef struct {
            int x,y,z;  // 'z' so that sizeof(xy) != sizeof(void*)
        } xy;

        xy** g_matrix;

        xy** create_matrix(int n, int m) {
            xy** mat = (xy**)malloc(n*sizeof(xy*));
            xy* arr = (xy*)malloc(n*m*sizeof(xy));
            for(int i=0; i<n; i++) {
                mat[i] = arr + i*m;
                for(int j=0; j<m; j++) {
                    mat[i][j].x = i+13;
                    mat[i][j].y = j+7;
                }
            }
            g_matrix = mat;
            return mat;
        }

        bool destroy_matrix(xy** mat, int n, int m) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (mat[i][j].x != i+13)
                        return false;
                    if (mat[i][j].y != j+7)
                        return false;
                }
            }
            g_matrix = nullptr;
            free(mat[0]);
            free(mat);
            return true;
        } }""")

        ns = cppyy.gbl.StructPointer2D;

        N, M = 2, 3
        m = ns.create_matrix(N, M)
        g = ns.g_matrix;

        assert (m.x, m.y) == (13, 7)
        assert (g.x, g.y) == (13, 7)

        for i in range(N):
            assert (m[i].x, m[i].y) == (i+13, 7)
            assert (g[i].x, g[i].y) == (i+13, 7)
            for j in range(M):
                assert (m[i][j].x, m[i][j].y) == (i+13, j+7)
                assert (g[i][j].x, g[i][j].y) == (i+13, j+7)

        assert ns.destroy_matrix(m, N, M)

        m = ns.create_matrix(N, M)
        assert ns.destroy_matrix(ns.g_matrix, N, M)

    def test38_plain_old_data(self):
        """Initializer construction of PODs"""

        import cppyy

        cppyy.cppdef("""\
        struct SomePOD_A { };
        struct SomePOD_B { int fInt; };
        struct SomePOD_C { int fInt; double fDouble; };
        struct SomePOD_D { std::array<double, 3> fArr; };
        struct SomePOD_E { int* fPtrInt; };
        struct SomePOD_F { std::array<double, 3>* fPtrArr; };

        namespace LotsOfPODS {
          struct SomePOD_A { };
          struct SomePOD_B { int fInt; };
          struct SomePOD_C { int fInt; double fDouble; };
          struct SomePOD_D { std::array<double, 3> fArr; };
          struct SomePOD_E { int* fPtrInt; };
          struct SomePOD_F { std::array<double, 3>* fPtrArr; };
        }""")

        for ns in [cppyy.gbl, cppyy.gbl.LotsOfPODS]:
          # no data member POD
            assert ns.SomePOD_A()

          # single data member POD
            b0 = ns.SomePOD_B()
            assert b0.__python_owns__
            assert b0.fInt == 0
            b1 = ns.SomePOD_B(42)
            assert b1.__python_owns__
            assert b1.fInt == 42
            b2 = ns.SomePOD_B(fInt = 17)
            assert b2.__python_owns__
            assert b2.fInt == 17

          # dual data member POD
            c0 = ns.SomePOD_C()
            assert c0.__python_owns__
            assert c0.fInt     == 0
            assert c0.fDouble  == 0.
            c1a = ns.SomePOD_C(42)
            assert c1a.__python_owns__
            assert c1a.fInt    == 42
            assert c1a.fDouble == 0.
            c1b = ns.SomePOD_C(fInt = 17)
            assert c1b.__python_owns__
            assert c1b.fInt    == 17
            assert c1b.fDouble == 0.
            c1c = ns.SomePOD_C(fDouble = 5.)
            assert c1c.__python_owns__
            assert c1c.fInt    == 0
            assert c1c.fDouble == 5.
            c2a = ns.SomePOD_C(88, 10.)
            assert c2a.__python_owns__
            assert c2a.fInt    == 88
            assert c2a.fDouble == 10.
            c2b = ns.SomePOD_C(fDouble=5., fInt=77)
            assert c2b.__python_owns__
            assert c2b.fInt    == 77
            assert c2b.fDouble ==5.

          # object type data member POD
            d0 = ns.SomePOD_D()
            assert d0.__python_owns__
            assert len(d0.fArr) == 3
            assert d0.fArr[0] == 0.
            d1 = ns.SomePOD_D((1., 2., 3.))
            assert d1.__python_owns__
            assert len(d1.fArr) == 3
            assert list(d1.fArr) == [1., 2., 3]

          # ptr type data member POD
            e0 = ns.SomePOD_E()
            assert e0.__python_owns__

          # ptr to object type data member pOD
            f0 = ns.SomePOD_F()
            assert f0.__python_owns__
            arr = cppyy.gbl.std.array['double', 3]((1., 2., 3.))
            f1 = ns.SomePOD_F(arr)
            assert f1.__python_owns__
            assert len(f1.fPtrArr) == 3
            assert list(f1.fPtrArr) == [1., 2., 3]

    def test39_aggregates(self):
        """Initializer construction of aggregates"""

        import cppyy

        cppyy.cppdef("""\
        namespace AggregateTest {
        class Atom {
        public:
            using size_type = std::size_t;

            struct AtomicNumber {
                size_type a_n = 0;
            };

            std::array<double, 3> coords;
        }; }""")

        ns = cppyy.gbl.AggregateTest

        Z = ns.Atom.AtomicNumber(5)
        assert Z.a_n == 5

        a = ns.Aggregate1()
        assert a.sInt == 17

        a = ns.Aggregate2()
        assert a.sInt == 27
        assert a.fInt == 42

        a = ns.Aggregate2(fInt=13)
        assert a.fInt == 13

        cppyy.cppdef("""\
        namespace AggregateTest {

        typedef enum _TYPE { DATA=0, SHAPE } TYPE;

        typedef struct _Buf {
          int val;
          const char *name;
          TYPE buf_type;
        } Buf; }""")

        ns = cppyy.gbl.AggregateTest

        b = ns.Buf(val=10, name="aap", buf_type=ns.SHAPE)

        assert b.val      == 10
        assert b.name     == "aap"
        assert b.buf_type == ns.SHAPE

    @mark.skip()
    def test40_more_aggregates(self):
        """More aggregate testings (used to fail/report errors)"""

        import cppyy

        cppyy.cppdef("""\
        namespace AggregateTest {
        enum E { A=1 };

        struct R1 { E e; };
        R1 make_R1() { return {A}; }

        struct S {
            S(int x): x(x) {}
            S(const S&) = delete;
            int x;
        }; }""")

        ns = cppyy.gbl.AggregateTest

        r1 = ns.make_R1()
        assert r1.e == ns.E.A

        if self.has_optional:
            cppyy.cppdef("""\
            namespace AggregateTest {
            struct R2 {
                std::optional<S> s = {};
            };

            R2 make_R2() {
                return {1};
            } }""")

            r2 = ns.make_R2()
            assert r2.s.x == 1

    @mark.xfail()
    def test41_complex_numpy_arrays(self):
        """Usage of complex numpy arrays"""

        import cppyy

        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        cppyy.cppdef("""\
        namespace ComplexArrays {
        typedef std::complex<float> fcomp;
        fcomp fcompdot(const fcomp* arr1, const fcomp* arr2, const int N) {
            fcomp res{0.f, 0.f};
            for (int i=0; i<N; ++i)
                res += arr1[i]*arr2[i];
            return res;
        }

        typedef std::complex<double> dcomp;
        dcomp dcompdot(const dcomp* arr1, const dcomp* arr2, const int N) {
            dcomp res{0., 0.};
            for (int i=0; i<N; ++i)
                res += arr1[i]*arr2[i];
            return res;
        } }""")

        def pycompdot(a, b, N):
            c = 0.+0.j
            for i in range(N):
                c += a[i]*b[i]
            return c

        ns = cppyy.gbl.ComplexArrays

        for ctype, func in [(np.complex64,  ns.fcompdot),
                            (np.complex128, ns.dcompdot)]:

            Acl = np.array([1.+2.j, 3.+4.j], dtype=ctype)
            Bcl = np.array([5.+6.j, 7.+8.j], dtype=ctype)

            pyCcl = pycompdot(Acl, Bcl, 2)

            Ccl = func(Acl, Bcl, 2)
            assert complex(Ccl) == pyCcl

    @mark.xfail()
    def test42_mixed_complex_arithmetic(self):
        """Mixin of Python and C++ std::complex in arithmetic"""

        import cppyy

        c = cppyy.gbl.std.complex['double'](1, 2)
        p = 1+2j

        assert c*c     == p*p
        assert c*c*c   == p*p*p
        assert c*(c*c) == p*(p*p)
        assert (c*c)*c == (p*p)*p

    @mark.xfail()
    def test43_ccharp_memory_handling(self):
        """cppyy side handled memory of C strings"""

        import cppyy

        cppyy.cppdef("""\
        namespace StructWithCChar {
        typedef struct {
            const char* name;
            int val;
        } BufInfo; }""")

        ns = cppyy.gbl.StructWithCChar

        a_str, b_str = 'abc', 'def'
        a = ns.BufInfo(a_str, 4)
        b = ns.BufInfo(b_str, 5)

        assert a.name == 'abc'
        assert a.val  == 4
        assert b.name == 'def'
        assert b.val  == 5

        a = ns.BufInfo('abc', 4)
        b = ns.BufInfo('def', 5)

        assert a.name == 'abc'
        assert a.val  == 4
        assert b.name == 'def'
        assert b.val  == 5

        a.name = 'ghi'
        b.name = 'jkl'

        assert a.name == 'ghi'
        assert b.name == 'jkl'

        a_str, b_str = 'mno', 'pqr'
        a = ns.BufInfo(val=4)
        b = ns.BufInfo(val=5)

        a.name = a_str
        b.name = b_str

        assert a.name == 'mno'
        assert a.val  == 4
        assert b.name == 'pqr'
        assert b.val  == 5

    def test44_buffer_memory_handling(self):
        """cppyy side handled memory of LL buffers"""

        import cppyy, gc
        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        cppyy.cppdef("""\
        namespace StructWithBuf {
        typedef struct {
            double* data1;
            double* data2;
            int size;
        } BufInfo; }""")

        ns = cppyy.gbl.StructWithBuf

        N = 10
        buf1 = ns.BufInfo(
            np.array(range(N), dtype=np.float64), 2.*np.array(range(N), dtype=np.float64), N)
        gc.collect()

        for i in range(buf1.size):
            assert buf1.data1[i] == 1.*i
            assert buf1.data2[i] == 2.*i

        buf2 = ns.BufInfo()
        buf2.data1 = 4.*np.array(range(N), dtype=np.float64)
        buf2.data2 = 5.*np.array(range(N), dtype=np.float64)
        buf2.size = N
        gc.collect()

        assert len(buf2.data1) == N
        for i in range(buf2.size):
            assert buf2.data1[i] == 4.*i
            assert buf2.data2[i] == 5.*i

        for i in range(buf1.size):
            assert buf1.data1[i] == 1.*i
            assert buf1.data2[i] == 2.*i

    def test45_const_ref_data(self):
        """Proper indirection for addressing const-ref data"""

        import cppyy

        cppyy.cppdef("""\
        namespace ConstRefData {
        struct A {
            std::string name;
        };

        A gA{"Hello, World!"};

        struct B {
            B() : body1(gA), body2(gA) {}
            A body1;
            const A& body2;
        }; }""")

        ns = cppyy.gbl.ConstRefData

        b = ns.B()
        assert b.body1.name == b.body2.name

    @mark.xfail()
    def test46_small_int_enums(self):
        """Proper typing of small int enums"""

        import cppyy

        cppyy.cppdef(r"""\
        namespace SmallEnums {
            enum class Enum { E0 = 0, E1 = 1, EN1 = -1, EN2 = -2 };

            enum class Int8Enum : int8_t { E0 = 0, E1 = 1, EN1 = -1, EN2 = -2 };
            enum class Int16Enum : int16_t { E0 = 0, E1 = 1, EN1 = -1, EN2 = -2 };

            enum class UInt8Enum : uint8_t { E0 = 0, E1 = 1, EMAX = 255 };
            enum class UInt16Enum : uint16_t { E0 = 0, E1 = 1, EMAX = 65535 };

            enum class CharEnum : char { E0 = '0', E1 = '1' };
            enum class SCharEnum : signed char { E0 = '0', E1 = '1' };
            enum class UCharEnum : unsigned char { E0 = '0', E1 = '1' };
        }""")

        ns = cppyy.gbl.SmallEnums

        for eclsname in ('Enum', 'Int8Enum', 'Int16Enum'):
            ecls = getattr(ns, eclsname)
            for ename, val in (('E0', 0), ('E1', 1), ('EN1', -1), ('EN2', -2)):
                assert getattr(ecls, ename) == val

        for eclsname in ('UInt8Enum', 'UInt16Enum'):
            ecls = getattr(ns, eclsname)
            for ename, val in (('E0', 0), ('E1', 1)):
                assert getattr(ecls, ename) == val

        assert ns.UInt8Enum.EMAX  ==   255
        assert ns.UInt16Enum.EMAX == 65535

        for eclsname in ('CharEnum', 'SCharEnum'):
            ecls = getattr(ns, eclsname)
            for ename, val in (('E0', '0'), ('E1', '1')):
                assert getattr(ecls, ename) == val

        # TODO: this is b/c unsigned char is considered a "byte" type by default;
        # it's meaning should probably be configurable?
        assert ns.UCharEnum.E0 == ord('0')
        assert ns.UCharEnum.E1 == ord('1')

        cppyy.cppdef(r"""\
        namespace SmallEnums {
            Int8Enum  func_int8()  { return Int8Enum::EN1; }
            UInt8Enum func_uint8() { return UInt8Enum::EMAX; }
        }""")

        assert ns.func_int8()  == -1
        assert ns.func_uint8() == 255

    @mark.xfail()
    def test47_hidden_name_enum(self):
        """Usage of hidden name enum"""

        import cppyy
        cppyy.cppdef(r"""
        namespace EnumFunctionSameName {
            enum foo { FOO = 42 };
            void foo() {}
            unsigned int bar(enum foo f) { return (unsigned int)f; }
            struct Bar { Bar(enum foo f) : fFoo(f) {} enum foo fFoo; };
        }""")

        ns = cppyy.gbl.EnumFunctionSameName

        assert ns.bar(ns.FOO) == 42
        assert ns.Bar(ns.FOO)
        assert ns.Bar(ns.FOO).fFoo == 42

        # TODO:
        cppyy.cppdef(r"""
        namespace EnumFunctionSameName {
            template<enum foo>
            struct BarT {};
        }""")
        #ns.BarT[ns.FOO]()

        # TODO:
        cppyy.cppdef(r"""
        namespace EnumFunctionSameName {
            enum class Foo : int8_t { FOO };
            void Foo() {}
            void bar(enum Foo) {}
        }""")
        #ns.bar(ns.Foo.FOO)

    def test48_bool_typemap(self):
        """Test mapping of bool type typedefs"""

        import cppyy

        cppyy.cppdef("""
        struct BoolTypeMapTest {
            typedef bool BoolType;
        };
        """)

        bt = cppyy.gbl.BoolTypeMapTest.BoolType

        assert bt.__name__ == 'BoolType'
        assert bt.__cpp_name__ == 'BoolTypeMapTest::BoolType'
        assert bt(1)
        assert bt(1) == True
        assert bt(1) != False
        assert bt(1) is True
        assert bt() == bt(0)
        assert not bt()
        assert bt() == False
        assert bt() != True
        assert bt() is False
        assert str(bt(1)) == 'True'
        assert str(bt(0)) == 'False'

    @mark.xfail(condition=IS_MAC_ARM, reason="Fails on mac-beta ARM64")
    def test49_addressof_method(self):
        """Use of addressof for (const) methods"""

        import cppyy

        assert cppyy.addressof(cppyy.gbl.std.vector[int].at.__overload__(':any:', False))
        assert cppyy.addressof(cppyy.gbl.std.vector[int].at.__overload__(':any:', True))

    def test50_int8_uint8_global_arrays(self):
        """Access to int8_t/uint8_t arrays that are global variables"""

        import cppyy

        ns = cppyy.gbl.Int8_Uint8_Arrays

        assert [ns.test[i]  for i in range(6)] == [-0x12, -0x34, -0x56, -0x78, 0x0, 0x0]
        assert [ns.utest[i] for i in range(6)] == [ 0x12,  0x34,  0x56,  0x78, 0x0, 0x0]

    def test51_polymorphic_types_in_maps(self):
        """Auto down-cast polymorphic types in maps"""
        import cppyy
        from cppyy import gbl

        cppyy.cppdef(
            """
        namespace PolymorphicMaps {
        struct Base {
            int x;
            Base(int x) : x(x) {}
            virtual ~Base() {}
        } b(1);

        struct Derived : public Base {
            int y;
            Derived(int i) : Base(i), y(i) {}
        } d(1);

        std::map<int, Base*> getBaseMap() {
            std::map<int, Base*> m;
            m[1] = &b;
            m[2] = &d;
            return m;
        }
        }  // namespace PolymorphicMaps
        """
        )

        for k, v in gbl.PolymorphicMaps.getBaseMap():
            if k == 1:
                assert type(v) == gbl.PolymorphicMaps.Base
            else:
                assert type(v) == gbl.PolymorphicMaps.Derived


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
