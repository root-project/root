import py, os, sys
from pytest import raises
from .support import setup_make, maxvalue

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("stltypesDict.so"))

def setup_module(mod):
    setup_make("stltypesDict.so")


class TestSTLVECTOR:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

    def test01_builtin_type_vector_types(self):
        """Test access to std::vector<int>/std::vector<double>"""

        import cppyy

        assert cppyy.gbl.std        is cppyy.gbl.std
        assert cppyy.gbl.std.vector is cppyy.gbl.std.vector

        assert callable(cppyy.gbl.std.vector)

        type_info = (
            ("int",     int),
            ("float",   "float"),
            ("double",  "double"),
        )

        for c_type, p_type in type_info:
            tv1 = getattr(cppyy.gbl.std, 'vector<%s>' % c_type)
            tv2 = cppyy.gbl.std.vector(p_type)
            assert tv1 is tv2
            assert tv1.iterator is cppyy.gbl.std.vector(p_type).iterator

            #----- 
            v = tv1(); v += range(self.N)    # default args from Reflex are useless :/
            if p_type == int:                # only type with == and != reflected in .xml
                assert v.begin().__eq__(v.begin())
                assert v.begin() == v.begin()
                assert v.end() == v.end()
                assert v.begin() != v.end()
                assert v.end() != v.begin()

            #-----
            for i in range(self.N):
                v[i] = i
                assert v[i] == i
                assert v.at(i) == i

            assert v.size() == self.N
            assert len(v) == self.N

            #-----
            v = tv1()
            for i in range(self.N):
                v.push_back(i)
                assert v.size() == i+1
                assert v.at(i) == i
                assert v[i] == i

            assert v.size() == self.N
            assert len(v) == self.N

    def test02_user_type_vector_type(self):
        """Test access to an std::vector<just_a_class>"""

        import cppyy

        assert cppyy.gbl.std        is cppyy.gbl.std
        assert cppyy.gbl.std.vector is cppyy.gbl.std.vector

        assert callable(cppyy.gbl.std.vector)

        tv1 = getattr(cppyy.gbl.std, 'vector<just_a_class>')
        tv2 = cppyy.gbl.std.vector('just_a_class')
        tv3 = cppyy.gbl.std.vector(cppyy.gbl.just_a_class)

        assert tv1 is tv2
        assert tv2 is tv3

        v = tv3()
        assert hasattr(v, 'size' )
        assert hasattr(v, 'push_back' )
        assert hasattr(v, '__getitem__' )
        assert hasattr(v, 'begin' )
        assert hasattr(v, 'end' )

        for i in range(self.N):
            v.push_back(cppyy.gbl.just_a_class())
            v[i].m_i = i
            assert v[i].m_i == i

        assert len(v) == self.N
        v.__destruct__()

    def test03_empty_vector_type(self):
        """Test behavior of empty std::vector<int>"""

        import cppyy

        v = cppyy.gbl.std.vector(int)()
        for arg in v:
            pass
        v.__destruct__()

    def test04_vector_iteration(self):
        """Test iteration over an std::vector<int>"""

        import cppyy

        v = cppyy.gbl.std.vector(int)()

        for i in range(self.N):
            v.push_back(i)
            assert v.size() == i+1
            assert v.at(i) == i
            assert v[i] == i

        assert v.size() == self.N
        assert len(v) == self.N

        i = 0
        for arg in v:
            assert arg == i
            i += 1

        assert list(v) == [i for i in range(self.N)]

        v.__destruct__()

    def test05_push_back_iterables_with_iadd(self):
        """Test usage of += of iterable on push_back-able container"""

        import cppyy

        v = cppyy.gbl.std.vector(int)()

        v += [1, 2, 3]
        assert len(v) == 3
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3

        v += (4, 5, 6)
        assert len(v) == 6
        assert v[3] == 4
        assert v[4] == 5
        assert v[5] == 6

        raises(TypeError, v.__iadd__, (7, '8'))  # string shouldn't pass
        assert len(v) == 7   # TODO: decide whether this should roll-back

        v2 = cppyy.gbl.std.vector(int)()
        v2 += [8, 9]
        assert len(v2) == 2
        assert v2[0] == 8
        assert v2[1] == 9

        v += v2
        assert len(v) == 9
        assert v[6] == 7
        assert v[7] == 8
        assert v[8] == 9

    def test06_vector_indexing(self):
        """Test python-style indexing to an std::vector<int>"""

        import cppyy

        v = cppyy.gbl.std.vector(int)()

        for i in range(self.N):
            v.push_back(i)

        raises(IndexError, 'v[self.N]')
        raises(IndexError, 'v[self.N+1]')

        assert v[-1] == self.N-1
        assert v[-2] == self.N-2

        assert len(v[0:0]) == 0
        assert v[1:2][0] == v[1]

        v2 = v[2:-1]
        assert len(v2) == self.N-3     # 2 off from start, 1 from end
        assert v2[0] == v[2]
        assert v2[-1] == v[-2]
        assert v2[self.N-4] == v[-2]

    def test07_vector_bool(self):
        """Usability of std::vector<bool> which can be a specialization"""

        import cppyy

        vb = cppyy.gbl.std.vector(bool)(8)
        assert [x for x in vb] == [False]*8

        vb[0] = True
        assert vb[0]
        vb[-1] = True
        assert vb[7]

        assert [x for x in vb] == [True]+[False]*6+[True]

        assert len(vb[4:8]) == 4
        assert list(vb[4:8]) == [False]*3+[True]

class estSTLSTRING:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_string_argument_passing(self):
        """Test mapping of python strings and std::string"""

        import cppyy
        std = cppyy.gbl.std
        stringy_class = cppyy.gbl.stringy_class

        c, s = stringy_class(""), std.string("test1")

        # pass through const std::string&
        c.set_string1(s)
        assert c.get_string1() == s

        c.set_string1("test2")
        assert c.get_string1() == "test2"

        # pass through std::string (by value)
        s = std.string("test3")
        c.set_string2(s)
        assert c.get_string1() == s

        c.set_string2("test4")
        assert c.get_string1() == "test4"

        # getting through std::string&
        s2 = std.string()
        c.get_string2(s2)
        assert s2 == "test4"

        raises(TypeError, c.get_string2, "temp string")

    def test02_string_data_access(self):
        """Test access to std::string object data members"""

        import cppyy
        std = cppyy.gbl.std
        stringy_class = cppyy.gbl.stringy_class

        c, s = stringy_class("dummy"), std.string("test string")

        c.m_string = "another test"
        assert c.m_string == "another test"
        assert str(c.m_string) == c.m_string
        assert c.get_string1() == "another test"

        c.m_string = s
        assert str(c.m_string) == s
        assert c.m_string == s
        assert c.get_string1() == s

    def test03_string_with_null_character(self):
        """Test that strings with NULL do not get truncated"""

        return # don't bother; is fixed in cling-support

        import cppyy
        std = cppyy.gbl.std
        stringy_class = cppyy.gbl.stringy_class

        t0 = "aap\0noot"
        assert t0 == "aap\0noot"

        c, s = stringy_class(""), std.string(t0, len(t0))

        c.set_string1(s)
        assert t0 == c.get_string1()
        assert s == c.get_string1()


class TestSTLLIST:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = 13

    def test01_builtin_list_type(self):
        """Test access to a list<int>"""

        import cppyy
        from cppyy.gbl import std

        type_info = (
            ("int",     int),
            ("float",   "float"),
            ("double",  "double"),
        )

        for c_type, p_type in type_info:
            tl1 = getattr(std, 'list<%s>' % c_type)
            tl2 = cppyy.gbl.std.list(p_type)
            assert tl1 is tl2
            assert tl1.iterator is cppyy.gbl.std.list(p_type).iterator

            #-----
            a = tl1()
            for i in range(self.N):
                a.push_back( i )

            assert len(a) == self.N
            assert 11 < self.N
            assert 11 in a

            #-----
            ll = list(a)
            for i in range(self.N):
                assert ll[i] == i

            for val in a:
                assert ll[ll.index(val)] == val

    def test02_empty_list_type(self):
        """Test behavior of empty list<int>"""

        import cppyy
        from cppyy.gbl import std

        a = std.list(int)()
        for arg in a:
            pass


class TestSTLMAP:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = 13

    def test01_builtin_map_type(self):
        """Test access to a map<int,int>"""

        import cppyy
        std = cppyy.gbl.std

        a = std.map(int, int)()
        for i in range(self.N):
            a[i] = i
            assert a[i] == i

        assert len(a) == self.N

        for key, value in a:
            assert key == value
        assert key   == self.N-1
        assert value == self.N-1

        # add a variation, just in case
        m = std.map(int, int)()
        for i in range(self.N):
            m[i] = i*i
            assert m[i] == i*i

        for key, value in m:
            assert key*key == value
        assert key   == self.N-1
        assert value == (self.N-1)*(self.N-1)

    def test02_keyed_maptype(self):
        """Test access to a map<std::string,int>"""

        import cppyy
        std = cppyy.gbl.std

        a = std.map(std.string, int)()
        for i in range(self.N):
            a[str(i)] = i
            assert a[str(i)] == i

        assert len(a) == self.N

    def test03_empty_maptype(self):
        """Test behavior of empty map<int,int>"""

        import cppyy
        std = cppyy.gbl.std

        m = std.map(int, int)()
        for key, value in m:
            pass

    def test04_unsignedvalue_typemap_types(self):
        """Test assignability of maps with unsigned value types"""

        import cppyy, math, sys
        std = cppyy.gbl.std

        mui = std.map(str, 'unsigned int')()
        mui['one'] = 1
        assert mui['one'] == 1
        raises(TypeError, mui.__setitem__, 'minus one', -1)

        # UInt_t is always 32b, maxvalue is sys.maxint/maxsize and follows system int
        maxint32 = int(math.pow(2,31)-1)
        mui['maxint'] = maxint32 + 3
        assert mui['maxint'] == maxint32 + 3

        mul = std.map(str, 'unsigned long')()
        mul['two'] = 2
        assert mul['two'] == 2
        mul['maxint'] = maxvalue + 3
        assert mul['maxint'] == maxvalue + 3

        raises(TypeError, mul.__setitem__, 'minus two', -2)

    def test05_STL_like_class_indexing_overloads(self):
        """Test overloading of operator[] in STL like class"""

        import cppyy
        stl_like_class = cppyy.gbl.stl_like_class

        a = stl_like_class(int)()
        assert a["some string" ] == 'string'
        assert a[3.1415] == 'double'

    def test06_STL_like_class_iterators(self):
        """Test the iterator protocol mapping for an STL like class"""

        return

        # TODO: reconsider this (this is a case where there is no return
        # type available and the code should (?) fall back onto getitem
        # iteration. (Python does, and that would break this.)

        import cppyy
        stl_like_class = cppyy.gbl.stl_like_class

        a = stl_like_class(int)()
        for i in a:
            pass

        assert i == 3


class TestSTLITERATOR:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_builtin_vector_iterators(self):
        """Test iterator comparison with operator== reflected"""

        import cppyy
        from cppyy.gbl import std

        v = std.vector(int)()
        v.resize(1)

        b1, e1 = v.begin(), v.end()
        b2, e2 = v.begin(), v.end()

        assert b1 == b2
        assert not b1 != b2

        assert e1 == e2
        assert not e1 != e2

        assert not b1 == e1
        assert b1 != e1

        b1.__preinc__()
        assert not b1 == b2
        assert b1 == e2
        assert b1 != b2
        assert b1 == e2


class TestSTLARRAY:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_array_of_basic_types(self):
        """Usage of std::array of basic types"""

        import cppyy
        from cppyy.gbl import std

        a = std.array[int, 4]()
        assert len(a) == 4
        for i in range(len(a)):
            a[i] = i
            assert a[i] == i

    def test02_array_of_pods(self):
        """Usage of std::array of PODs"""

        import cppyy
        from cppyy import gbl
        from cppyy.gbl import std

        a = std.array[gbl.ArrayTest.Point, 4]()
        assert len(a) == 4
        for i in range(len(a)):
            a[i].px = i
            assert a[i].px == i
            a[i].py = i**2
            assert a[i].py == i**2

    def test03_array_of_pointer_to_pods(self):
        """Usage of std::array of pointer to PODs"""

        import cppyy
        from cppyy import gbl
        from cppyy.gbl import std

        ll = [gbl.ArrayTest.Point() for i in range(4)]
        for i in range(len(ll)):
            ll[i].px = 13*i
            ll[i].py = 42*i

        a = std.array['ArrayTest::Point*', 4]()
        assert len(a) == 4
        for i in range(len(a)):
            a[i] = ll[i]
            assert a[i].px == 13*i
            assert a[i].py == 42*i

        raises(TypeError, a.__setitem__, 1, 42)

        for i in range(len(a)):
            assert gbl.ArrayTest.get_pp_px(a.data(), i) == 13*i
            assert gbl.ArrayTest.get_pp_py(a.data(), i) == 42*i

            assert gbl.ArrayTest.get_pa_px(a.data(), i) == 13*i
            assert gbl.ArrayTest.get_pa_py(a.data(), i) == 42*i
