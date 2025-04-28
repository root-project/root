# File: roottest/python/stl/PyROOT_stltests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 10/25/05
# Last: 02/28/16

"""STL unit tests for PyROOT package."""

import sys, os, unittest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import *
from pytest import raises

from ROOT import kRed   # needed (?!) to load iterator comparison funcs on Mac

def setup_module(mod):
    if not os.path.exists('StlTypes.C'):
        os.chdir(os.path.dirname(__file__))
        err = os.system("make StlTypes_C")
        if err:
            raise OSError("'make' failed (see stderr)")


### STL vector test case =====================================================
class TestClasSTLVECTOR:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = 13

    def test01_builtin_vector_type(self):
        """Test access to a vector<int> (part of cintdlls)"""

        from cppyy.gbl import std

        a = std.vector(int)(self.N)
        assert len(a) == self.N

        for i in range(self.N):
            a[i] = i
            assert a[i] == i
            assert a.at(i) == i

        assert a.size() == self.N
        assert len(a) == self.N

    def test02_builtin_vector_type(self):
        """Test access to a vector<double> (part of cintdlls)"""

        from cppyy.gbl import std

        a = std.vector('double')()
        for i in range(self.N):
            a.push_back( i )
            assert a.size() == i+1
            assert a[i] == i
            assert a.at(i) == i

        assert a.size() == self.N
        assert len(a) == self.N

    def test03_generated_vector_type(self):
        """Test access to a ACLiC generated vector type"""

        from cppyy.gbl import std, JustAClass

        a = std.vector( JustAClass )()
        assert hasattr(a, 'size')
        assert hasattr(a, 'push_back')
        assert hasattr(a, '__getitem__')
        assert hasattr(a, 'begin')
        assert hasattr(a, 'end')

        assert a.size() == 0

        for i in range(self.N):
            a.push_back( JustAClass() )
            a[i].m_i = i
            assert a[i].m_i == i

        assert len(a) == self.N

    def test04_empty_vector(self):
        """Test behavior of empty vector<int> (part of cintdlls)"""

        from cppyy.gbl import std

        a = std.vector(int)()
        assert len(a) == 0
        for arg in a:
            pass

        # ROOT-10118
        # In current Cppyy, STL containers evaluate to True
        # if they contain at least one element
        assert not a
        a.push_back(0)
        assert a

    def test05_pushback_iterables_with_iadd(self):
        """Test usage of += of iterable on push_back-able container"""

        from cppyy.gbl import std

        a = std.vector(int)()
        a += [1, 2, 3]
        assert len(a) == 3

        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 3

        a += ( 4, 5, 6 )
        assert len(a) == 6

        assert a[3] == 4
        assert a[4] == 5
        assert a[5] == 6

        raises(TypeError, a.__iadd__, (7, '8'))

    def test06_vector_return_downcasting(self):
        """Pointer returns of vector indexing must be down cast"""

        from cppyy.gbl import std, PR_Test

        v = PR_Test.mkVect()
        assert type(v) == std.vector('PR_Test::Base*')
        assert len(v) == 1
        assert type(v[0]) == PR_Test.Derived
        assert PR_Test.checkType(v[0]) == PR_Test.checkType(PR_Test.Derived())

        p = PR_Test.check()
        assert type(p) == PR_Test.Derived
        assert PR_Test.checkType(p) == PR_Test.checkType(PR_Test.Derived())

    def test07_vector_bool_iter(self):
        """Iteration over a vector<bool>"""
        # ROOT-9397
        from cppyy.gbl import std
        v = std.vector[bool]()
        l = [True, False]
        for b in l:
            v.push_back(b)
        assert [ b for b in v ] == l


### STL list test case =======================================================
class TestClasSTLLIST:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = 13

    def test01_builtin_list_type(self):
        """Test access to a list<int> (part of cintdlls)"""

        from cppyy.gbl import std

        a = std.list(int)()
        for i in range(self.N):
            a.push_back( i )

        assert len(a) == self.N
        assert 11 in a

        ll = list(a)
        for i in range(self.N):
            assert ll[i] == i

        for val in a:
            assert ll[ ll.index(val) ] == val

    def test02_empty_list_type(self):
        """Test behavior of empty list<int> (part of cintdlls)"""

        from cppyy.gbl import std

        a = std.list(int)()
        for arg in a:
            pass


### STL map test case ========================================================
class TestClasSTLMAP:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = 13

    def test01_builtin_map_type(self):
        """Test access to a map<int,int> (part of cintdlls)"""

        from cppyy.gbl import std

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

    def test02_keyed_map_type(self):
        """Test access to a map<std::string,int> (part of cintdlls)"""

        from cppyy.gbl import std

        a = std.map(std.string, int)()
        for i in range(self.N):
            a[str(i)] = i
            assert a[str(i)] == i

        assert i == self.N-1
        assert len(a) == self.N

    def test03_empty_map_type(self):
        """Test behavior of empty map<int,int> (part of cintdlls)"""

        from cppyy.gbl import std

        m = std.map(int, int)()
        for key, value in m:
            pass

    def test04_unsigned_value_type_map_types(self):
        """Test assignability of maps with unsigned value types (not part of cintdlls)"""

        import math
        from cppyy.gbl import std

        mui = std.map(str, 'unsigned int')()
        mui['one'] = 1
        assert mui['one'] == 1
        raises(ValueError, mui.__setitem__, 'minus one', -1)

      # UInt_t is always 32b, sys.maxint follows system int
        maxint32 = int(math.pow(2,31)-1)
        mui['maxint'] = maxint32 + 3
        assert mui['maxint'] == maxint32 + 3

        mul = std.map(str, 'unsigned long')()
        mul['two'] = 2
        assert mul['two'] == 2
        mul['maxint'] = maxvalue + 3
        assert mul['maxint'] == maxvalue + 3
        raises(ValueError, mul.__setitem__, 'minus two', -2)

    def test05_freshly_instantiated_map_type(self):
        """Instantiate a map from a newly defined class"""

        import cppyy
        cppyy.cppdef('template<typename T> struct Data { T fVal; };')

        from cppyy.gbl import std, Data

        results = std.map( std.string, Data(int) )()
        d = Data(int)(); d.fVal = 42
        results['summary'] = d
        assert results.size() == 1
        for tag, data in results:
            assert data.fVal == 42


### Protocol mapping for an STL like class ===================================
class TestClasSTLLIKECLASS:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_STL_like_class_indexing_overloads(self):
        """Test overloading of operator[] in STL like class"""

        from cppyy.gbl import STLLikeClass

        a = STLLikeClass(int)()
        assert a["some string"] == 'string'
        assert a[3.1415] == 'double'

    def test02_STL_like_class_iterators(self):
        """Test the iterator protocol mapping for an STL like class"""

        from cppyy.gbl import STLLikeClass

        a = STLLikeClass(int)()
        for i in a:
            pass

        assert i == 3


### String handling ==========================================================
class TestClasSTLSTRINGHANDLING:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_string_argument_passing(self):
        """Test mapping of python strings and std::string"""

        from cppyy.gbl import std, StringyClass

        c, s = StringyClass(), std.string("test1")

      # pass through const std::string&
        c.SetString1(s)
        assert type(c.GetString1()) == str
        assert c.GetString1() == s

        c.SetString1("test2")
        assert c.GetString1() == "test2"

      # pass through std::string (by value)
        s = std.string("test3")
        c.SetString2(s)
        assert c.GetString1() == s

        c.SetString2("test4")
        assert c.GetString1() == "test4"

      # getting through std::string&
        s2 = std.string()
        c.GetString2(s2)
        assert s2 == "test4"

        raises(TypeError, c.GetString2, "temp string")

    def test02_string_data_access(self):
        """Test access to std::string object data members"""

        from cppyy.gbl import std, StringyClass

        c, s = StringyClass(), std.string("test string")
 
        c.m_string = s
        assert c.m_string == s
        assert c.GetString1() == s

        c.m_string = "another test"
        assert c.m_string == "another test"
        assert c.GetString1() == "another test"

    def test03_string_with_null_Character(self):
        """Test that strings with NULL do not get truncated"""

        from cppyy.gbl import std, StringyClass

        t0 = "aap\0noot"
        assert t0 == "aap\0noot"

        c, s = StringyClass(), std.string(t0, 0, len(t0))

        c.SetString1( s )
        assert t0 == c.GetString1()
        assert s == c.GetString1()

    def test04_string_hash(self):
        """Test that std::string has the same hash as the equivalent Python str"""
        # ROOT-10830

        from cppyy.gbl import std

        assert hash(std.string("test")) == hash("test")

        # Somehow redundant, but for completeness
        v = std.vector(std.string)()
        v.push_back('a'); v.push_back('b'); v.push_back('c')
        assert set(v) == set('abc')

    def test05_string_concat(self):
        """Test concatenation of std::string and Python str"""
        # ROOT-10830

        import cppyy
        s1 = cppyy.gbl.std.string("ying")
        s2 = "yang"
        assert s1 + s2 == "yingyang"


### Iterator comparison ======================================================
class TestClasSTLITERATORCOMPARISON:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)

    def __run_tests(self, container):
        assert len(container) == 1

        b1, e1 = container.begin(), container.end()
        b2, e2 = container.begin(), container.end()

        def pr_cmp(a, b):
           if a == b: return 0
           return 1

        assert b1.__eq__(b2)
        assert not b1.__ne__(b2)
        assert pr_cmp(b1, b2) == 0

        assert e1.__eq__(e2)
        assert not e1.__ne__(e2)
        assert pr_cmp(e1, e2) == 0

        assert not b1.__eq__(e1)
        assert b1.__ne__(e1)
        assert pr_cmp(b1, e1) != 0

        b1.__preinc__()
        assert not b1.__eq__(b2)
        assert b1.__eq__(e2)
        assert pr_cmp(b1, b2) != 0
        assert pr_cmp(b1, e1) == 0
        assert b1 != b2
        assert b1 == e2

    def test01_builtin_vector_iterators(self):
        """Test iterator comparison for vector"""

        from cppyy.gbl import std

        v = std.vector(int)()
        v.resize(1)

        self.__run_tests(v)

    def test02_builtin_list_iterators(self):
        """Test iterator comparison for list"""

        from cppyy.gbl import std

        l = std.list(int)()
        l.push_back(1)

        self.__run_tests(l)

    def test03_builtin_map_iterators(self):
        """Test iterator comparison for map"""

        from cppyy.gbl import std

        m = std.map(int, int)()
        m[1] = 1

        self.__run_tests(m)

    def test04_builtin_vector_iterators_bool(self):
        """Test iterator comparison for vector of bool"""

        from cppyy.gbl import std

        v = std.vector(bool)()
        v.resize(1)

        self.__run_tests(v)


### Stream usage =============================================================
class TestClasSTLSTREAM:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_pass_stringstream(self):
        """Pass stringstream through ostream&"""

        from cppyy.gbl import std, StringStreamUser

        s = std.stringstream()
        o = StringStreamUser()

        o.fillStream(s)

        assert "StringStreamUser Says Hello!" == s.str()
        
### Iteration with SET========================================================
class TestClasSTLSET:

    def setup_class(cls):
        import cppyy
        cls.test_dct = "StlTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)

    def getset(self):
        from cppyy.gbl import std
        s = std.set("int")()
        s.insert(1)
        s.insert(2)
        s.insert(3)
        return s

    def test01_iterate_set(self):
        """Test ref counter while looking over std::set (ROOT-8038)"""
        result = [elem for elem in self.getset()]
        assert result == [1, 2, 3]

## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
