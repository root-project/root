# -*- coding: UTF-8 -*-
import py, sys
from pytest import raises, skip
from .support import setup_make, pylong, pyunicode, maxvalue, ispypy

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("stltypesDict"))

def setup_module(mod):
    setup_make("stltypes")


# after CPython's Lib/test/seq_tests.py
def iterfunc(seqn):
    """Regular generator"""
    for i in seqn:
        yield i

class Sequence:
    """Sequence using __getitem__"""
    def __init__(self, seqn):
        self.seqn = seqn
    def __getitem__(self, i):
        return self.seqn[i]

class IterFunc:
    """Sequence using iterator protocol"""
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v
    next = __next__ # p2.7

class IterGen:
    """Sequence using iterator protocol defined with a generator"""
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        for val in self.seqn:
            yield val

class IterNextOnly:
    """Missing __getitem__ and __iter__"""
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __next__(self):
        if self.i >= len(self.seqn): raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v
    next = __next__ # p2.7

class IterNoNext:
    """Iterator missing __next__()"""
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self

class IterGenExc:
    """Test propagation of exceptions"""
    def __init__(self, seqn):
        self.seqn = seqn
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        3 // 0
    next = __next__ # p2.7

class IterFuncStop:
    """Test immediate stop"""
    def __init__(self, seqn):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration
    next = __next__ # p2.7

from itertools import chain
def itermulti(seqn):
    """Test multiple tiers of iterators"""
    return chain(map(lambda x:x, iterfunc(IterGen(Sequence(seqn)))))

class LyingTuple(tuple):
    def __iter__(self):
        yield 1

class LyingList(list):
    def __iter__(self):
        yield 1

def constructors_cpython_test(type2test):
    l0 = []
    l1 = [0]
    l2 = [0, 1]

    u = type2test()
    u0 = type2test(l0)
    u1 = type2test(l1)
    u2 = type2test(l2)

    uu = type2test(u)
    uu0 = type2test(u0)
    uu1 = type2test(u1)
    uu2 = type2test(u2)

    v = type2test(tuple(u))
    class OtherSeq:
        def __init__(self, initseq):
            self.__data = initseq
        def __len__(self):
            return len(self.__data)
        def __getitem__(self, i):
            return self.__data[i]
    s = OtherSeq(u0)
    v0 = type2test(s)
    assert len(v0) == len(s)

    # the following does not work for type-checked containers
    #s = "this is also a sequence"
    #vv = type2test(s)
    #assert len(vv) == len(s)

  # Create from various iteratables
    # as above, can not put strings in type-checked containers
    #for s in ("123", "", range(1000), ('do', 1.2), range(2000,2200,5)):
    for s in (range(1000), range(2000,2200,5)):
        for g in (Sequence, IterFunc, IterGen,
                  itermulti, iterfunc):
            assert type2test(g(s)) == type2test(s)
        assert type2test(IterFuncStop(s))  ==  type2test()
        # as above, no strings
        #assert type2test(c for c in "123") == type2test("123")
        raises(TypeError, type2test, IterNextOnly(s))
        raises(TypeError, type2test, IterNoNext(s))
        raises(ZeroDivisionError, type2test, IterGenExc(s))

  # Issue #23757 (in CPython)
    #assert type2test(LyingTuple((2,))) == type2test((1,))
    #assert type2test(LyingList([2]))   == type2test([1])

def getslice_cpython_test(type2test):
    """Detailed slicing tests from CPython"""

    l = [0, 1, 2, 3, 4]
    u = type2test(l)

    assert u[0:0]        == type2test()
    assert u[1:2]        == type2test([1])
    assert u[-2:-1]      == type2test([3])
    assert u[-1000:1000] == u
    assert u[1000:-1000] == type2test([])
    assert u[:]          == u
    assert u[1:None]     == type2test([1, 2, 3, 4])
    assert u[None:3]     == type2test([0, 1, 2])

  # Extended slices
    assert u[::]          == u
    assert u[::2]         == type2test([0, 2, 4])
    assert u[1::2]        == type2test([1, 3])
    assert u[::-1]        == type2test([4, 3, 2, 1, 0])
    assert u[::-2]        == type2test([4, 2, 0])
    assert u[3::-2]       == type2test([3, 1])
    assert u[3:3:-2]      == type2test([])
    assert u[3:2:-2]      == type2test([3])
    assert u[3:1:-2]      == type2test([3])
    assert u[3:0:-2]      == type2test([3, 1])
    assert u[::-100]      == type2test([4])
    assert u[100:-100:]   == type2test([])
    assert u[-100:100:]   == u
    assert u[100:-100:-1] == u[::-1]
    assert u[-100:100:-1] == type2test([])
    assert u[-pylong(100):pylong(100):pylong(2)] == type2test([0, 2, 4])

  # Test extreme cases with long ints
    a = type2test([0,1,2,3,4])
    # the following two fail b/c PySlice_GetIndices succeeds w/o error, while
    # returning an overflown value (list object uses different internal APIs)
    #assert a[ -pow(2,128): 3 ] == type2test([0,1,2])
    #assert a[ 3: pow(2,145) ]  == type2test([3,4])
    assert a[3::maxvalue]      == type2test([3])


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
            v = tv1()
            assert not v
            v += range(self.N)
            assert v
            if p_type == int:
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
            assert len(v.data()) == self.N

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
        assert hasattr(v, 'size')
        assert hasattr(v, 'push_back')
        assert hasattr(v, '__getitem__')
        assert hasattr(v, 'begin')
        assert hasattr(v, 'end')

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
        assert not v
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

        sz = len(v)
        v += []
        assert len(v) == sz

    def test06_vector_indexing(self):
        """Test python-style indexing to an std::vector<int>"""

        import cppyy

        v = cppyy.gbl.std.vector(int)()

        for i in range(self.N):
            v.push_back(i)

        with raises(IndexError):
            v[self.N]
        with raises(IndexError):
            v[self.N+1]

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

    def test08_vector_enum(self):
        """Usability of std::vector<> of some enums"""

        import cppyy

        assert cppyy.gbl.VecTestEnum
        for tp in ['VecTestEnum', cppyy.gbl.VecTestEnum]:
            ve = cppyy.gbl.std.vector[tp]()
            ve.push_back(cppyy.gbl.EVal1);
            assert ve[0] == 1
            ve[0] = cppyy.gbl.EVal2
            assert ve[0] == 3

        assert cppyy.gbl.VecTestEnumNS.VecTestEnum
        for tp in ['VecTestEnumNS::VecTestEnum', cppyy.gbl.VecTestEnumNS.VecTestEnum]:
            ve = cppyy.gbl.std.vector['VecTestEnumNS::VecTestEnum']()
            ve.push_back(cppyy.gbl.VecTestEnumNS.EVal1);
            assert ve[0] == 5
            ve[0] = cppyy.gbl.VecTestEnumNS.EVal2
            assert ve[0] == 42

    def test09_vector_of_string(self):
        """Adverse effect of implicit conversion on vector<string>"""

        import cppyy

        assert cppyy.gbl.vectest_ol1("")  == 2
        assert cppyy.gbl.vectest_ol1("a") == 2
        assert cppyy.gbl.vectest_ol2("")  == 2
        assert cppyy.gbl.vectest_ol2("a") == 2

        raises(TypeError, cppyy.gbl.std.vector["std::string"], "abc")

    def test10_vector_std_distance(self):
        """Use of std::distance with vector"""

        import cppyy
        from cppyy.gbl import std

        v = std.vector[int]([1, 2, 3])
        assert v.size() == 3
        assert std.distance(v.begin(), v.end()) == v.size()
        assert std.distance[type(v).iterator](v.begin(), v.end()) == v.size()

    def test11_vector_of_pair(self):
        """Use of std::vector<std::pair>"""

        import cppyy

      # after the original bug report
        cppyy.cppdef("""
        class PairVector {
        public:
            std::vector<std::pair<double, double>> vector_pair(const std::vector<std::pair<double, double>>& a) {
                return a;
            }
        };
        """)

        from cppyy.gbl import PairVector
        a = PairVector()
        ll = [[1., 2.], [2., 3.], [3., 4.], [4., 5.]]
        v = a.vector_pair(ll)

        assert len(v) == 4
        i = 0
        for p in v:
            p.first  == ll[i][0]
            p.second == ll[i][1]
            i += 1
        assert i == 4

      # TODO: nicer error handling for the following (current: template compilation failure trying
      # to assign a pair with <double, string> to <double, double>)
        # ll2 = ll[:]
        # ll2[2] = ll[2][:]
        # ll2[2][1] = 'a'
        # v = a.vector_pair(ll2)

        ll3 = ll[:]
        ll3[0] = 'a'
        raises(TypeError, a.vector_pair, ll3)

        ll4 = ll[:]
        ll4[1] = 'a'
        raises(TypeError, a.vector_pair, ll4)

    def test12_vector_lifeline(self):
        """Check lifeline setting on vectors of objects"""

        import cppyy

        cppyy.cppdef("""namespace Lifeline {
        static int count = 0;
        template <typename T>
        struct C {
            C(int x) : x(x) { ++count; }
            C(const C& c) : x(c.x) { ++count; }
            ~C() { --count; }
            int x;
        };
        auto foo() { return std::vector<C<int>>({C<int>(1337)}); }
        auto bar() { return std::vector<std::string>{1024, "hello"}; }
        }""")

        assert cppyy.gbl.Lifeline.count == 0
        assert not cppyy.gbl.Lifeline.foo()._getitem__unchecked.__set_lifeline__
        assert cppyy.gbl.Lifeline.foo()[0].x == 1337
        raises(IndexError, cppyy.gbl.Lifeline.foo().__getitem__, 1)
        assert cppyy.gbl.Lifeline.foo()._getitem__unchecked.__set_lifeline__

        import gc
        gc.collect()
        assert cppyy.gbl.Lifeline.count == 0

        l = list(cppyy.gbl.Lifeline.bar())
        for val in l:
            assert hasattr(val, '__lifeline')

    def test13_vector_smartptr_iteration(self):
        """Iteration over smart pointers"""

        import cppyy

        cppyy.cppdef("""namespace VectorOfShared {
        struct X {
            int y;
            X() : y(0) {}
            X(int y) : y(y) { }
            std::vector<std::shared_ptr<X>> gimeVec() {
                std::vector<std::shared_ptr<X>> result;
                for (int i = 0; i < 10; ++i) {
                    result.push_back(std::make_shared<X>(i));
                }
                return result;
            }
        }; }""")

        test = cppyy.gbl.VectorOfShared.X()
        result = test.gimeVec()
        assert 'shared' in type(result).__cpp_name__
        assert len(result) == 10

        for i in range(len(result)):
            assert result[i].y == i

        i = 0
        for res in result:
            assert res.y == i
            i += 1
        assert i == len(result)

    def test14_vector_of_vector_of_(self):
        """Nested vectors"""

        from cppyy.gbl.std import vector

        vv = vector[vector[int]](((1, 2), [3, 4]))

        assert len(vv) == 2
        assert list(vv[0]) == [1, 2]
        assert vv[0][0] == 1
        assert vv[0][1] == 2
        assert list(vv[1]) == [3, 4]
        assert vv[1][0] == 3
        assert vv[1][1] == 4

    def test15_vector_slicing(self):
        """Advanced test of vector slicing"""

        from cppyy.gbl.std import vector

        l = list(range(10))
        v = vector[int](range(10))

        assert list(v[2:2])    == l[2:2]
        assert list(v[2:2:-1]) == l[2:2:-1]
        assert list(v[2:5])    == l[2:5]
        assert list(v[5:2])    == l[5:2]
        assert list(v[2:5:-1]) == l[2:5:-1]
        assert list(v[5:2:-1]) == l[5:2:-1]
        assert list(v[2:5: 2]) == l[2:5: 2]
        assert list(v[5:2: 2]) == l[5:2: 2]
        assert list(v[2:5:-2]) == l[2:5:-2]
        assert list(v[5:2:-2]) == l[5:2:-2]
        assert list(v[2:5: 7]) == l[2:5: 7]
        assert list(v[5:2: 7]) == l[5:2: 7]
        assert list(v[2:5:-7]) == l[2:5:-7]
        assert list(v[5:2:-7]) == l[5:2:-7]

      # additional test from CPython's test suite
        getslice_cpython_test(vector[int])

    def test16_vector_construction(self):
        """Vector construction following CPython's sequence"""

        import cppyy

        constructors_cpython_test(cppyy.gbl.std.vector[int])

    def test17_vector_cpp17_style(self):
        """C++17 style initialization of std::vector"""

        import cppyy

        l = [1.0, 2.0, 3.0]
        v = cppyy.gbl.std.vector(l)
        assert list(l) == l

    def test18_array_interface(self):
        """Test usage of __array__ from numpy"""

        import cppyy

        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        a = cppyy.gbl.std.vector[int]((1, 2, 3))

        b = np.array(a)
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

        cppyy.cppdef("""\
        namespace ImplicitVector {
        int func(std::vector<int> v) {
            return std::accumulate(v.begin(), v.end(), 0);
        } }""")

        ns = cppyy.gbl.ImplicitVector

        v = np.array(range(10), dtype=np.intc)
        assert ns.func(v) == sum(v)

        v = np.array(range(10), dtype=np.uint8)
        with raises(TypeError):
            ns.func(v)

        v = np.array(v, dtype=np.intc)
        assert ns.func(v) == sum(v)

    def test19_vector_point3d(self):
        """Iteration over a vector of by-value objects"""

        import cppyy

        N = 10

        cppyy.cppdef("""namespace vector_point3d {
        class Point3D {
            double x, y, z;

        public:
            Point3D(double x, double y, double z) : x(x), y(y), z(z) {}
            double square() { return x*x+y*y+z*z; }
        }; }""")

        Point3D = cppyy.gbl.vector_point3d.Point3D
        v = cppyy.gbl.std.vector[Point3D]()
        for i in range(N):
            v.emplace_back(i, i*2, i*3)

        pysum = 0.
        for x in range(N):
            pysum += 14*x**2

        cppsum = 0.
        for p in v:
            cppsum += p.square()

        assert cppsum == pysum

    def test20_vector_cstring(self):
        """Usage of a vector of const char*"""

        import cppyy

        cppyy.cppdef("""\
        namespace VectorConstCharStar {
            std::vector<const char*> test = {"hello"};
        }""")

        ns = cppyy.gbl.VectorConstCharStar

        assert len(ns.test) == 1
        assert ns.test[0] == "hello"

        ns.test.push_back("world")
        assert len(ns.test) == 2
        assert ns.test[0] == "hello"
        assert ns.test[1] == "world"

    def test21_vector_of_structs_data(self):
        """Vector of structs data() should return array-like"""

        import cppyy
        import cppyy.ll

        cppyy.cppdef("""\
        namespace ArrayLike {
        struct __attribute__((__packed__)) Vector3f {
            float x, y, z;
        }; }""")

        N = 5

        v = cppyy.gbl.std.vector['ArrayLike::Vector3f'](N)

        for i in range(N):
            d = v[i]
            d.x, d.y, d.z = i, i*N, i*N**2

        data = v.data()
        for i in range(N):
            d = data[i]
            assert d.x == float(i)
            assert d.y == float(i*N)
            assert d.z == float(i*N**2)

      # the following should not raise
        mv = cppyy.ll.as_memoryview(data)

      # length of the view is in bytes
        assert len(mv) == len(v)
        assert mv.itemsize == cppyy.sizeof(cppyy.gbl.ArrayLike.Vector3f)
        assert mv.nbytes   == cppyy.sizeof(cppyy.gbl.ArrayLike.Vector3f) * len(v)

    def test22_polymorphic(self):
        """Vector of polymorphic types should auto-cast"""

        import cppyy

        cppyy.cppdef("""\
        namespace Polymorphic {
        class vertex {
        public:
          virtual ~vertex() {}
        };

        class Mvertex : public vertex {};

        class vCont {
        public:
          virtual ~vCont() { for (auto& v: verts) delete v; }
          std::vector<vertex*> verts { new vertex(), new Mvertex() };
          const std::vector<vertex*>& vertices() { return verts; }
        }; }""")

        ns = cppyy.gbl.Polymorphic
        cont = ns.vCont()
        verts = cont.vertices()

        assert len([x for x in verts if isinstance(x, ns.Mvertex)]) == 1

    def test23_copy_conversion(self):
        """Vector given an array of different type should copy convert"""

        import cppyy

        try:
            import numpy as np
        except ImportError:
            skip('numpy is not installed')

        x = np.array([5., 25., 125.])
        v = cppyy.gbl.std.vector('float')(x)

        for f, d in zip(x, v):
            assert f == d

    def test24_byte_vectors(self):
        """Vectors of "byte" types should return low level views"""

        import cppyy
        import cppyy.types

        vector = cppyy.gbl.std.vector

        for ctype in ('unsigned char', 'signed char', 'int8_t', 'uint8_t'):
            vc = vector[ctype](range(10))
            data = vc.data()

            assert type(data) == cppyy.types.LowLevelView
            assert len(data) == 10

            for i, d in enumerate(data):
                assert d == i

        for ctype in ('signed char', 'int8_t'):
            vc = vector[ctype](range(-5, 5, 1))
            data = vc.data()

            assert type(data) == cppyy.types.LowLevelView
            assert len(data) == 10

            for i, d in zip(range(-5, 5, 1), data):
                assert d == i


class TestSTLSTRING:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_string_argument_passing(self):
        """Test mapping of python strings and std::[w]string"""

        import cppyy
        std = cppyy.gbl.std

        for stp, pystp in [(std.string, str), (std.wstring, pyunicode)]:
            stringy_class = cppyy.gbl.stringy_class[stp]

            c, s = stringy_class(pystp("")), stp(pystp("test1"))

            # pass through const std::[w]string&
            c.set_string1(s)
            assert c.get_string1() == s

            c.set_string1(pystp("test2"))
            assert c.get_string1() == pystp("test2")

            # pass through std::string (by value)
            s = stp(pystp("test3"))
            c.set_string2(s)
            assert c.get_string1() == s

            c.set_string2(pystp("test4"))
            assert c.get_string1() == pystp("test4")

            # getting through std::[w]string&
            s2 = stp()
            c.get_string2(s2)
            assert s2 == "test4"

            raises(TypeError, c.get_string2, "temp string")

    def test02_string_data_access(self):
        """Test access to std::string object data members"""

        import cppyy
        std = cppyy.gbl.std

        for stp, pystp in [(std.string, str), (std.wstring, pyunicode)]:
            stringy_class = cppyy.gbl.stringy_class[stp]

            c, s = stringy_class(pystp("dummy")), stp(pystp("test string"))

            c.m_string = pystp("another test")
            assert c.m_string == pystp("another test")
            assert pystp(c.m_string) == c.m_string
            assert c.get_string1() == pystp("another test")

            c.m_string = s
            assert pystp(c.m_string) == s
            assert c.m_string == s
            assert c.get_string1() == s

    def test03_string_with_null_character(self):
        """Test that strings with NULL do not get truncated"""

        import cppyy
        std = cppyy.gbl.std
        stringy_class = cppyy.gbl.stringy_class["std::string"]

        t0 = "aap\0noot"
        assert t0 == "aap\0noot"

        c, s = stringy_class(""), std.string(t0)

        c.set_string1(s)
        assert t0 == c.get_string1()
        assert s == c.get_string1()

        assert std.string('ab\0c')       == 'ab\0c'
        assert repr(std.string('ab\0c')) == repr(b'ab\0c')
        assert str(std.string('ab\0c'))  == str('ab\0c')

    def test04_array_of_strings(self):
        """Access to global arrays of strings"""

        import cppyy

        assert tuple(cppyy.gbl.str_array_1) == ('a', 'b', 'c')
        str_array_2 = cppyy.gbl.str_array_2
        # fix up the size
        str_array_2.size = 4
        assert tuple(str_array_2) == ('d', 'e', 'f', 'g')
        assert tuple(str_array_2) == ('d', 'e', 'f', 'g')

        # multi-dimensional
        vals = ['a', 'b', 'c', 'd', 'e', 'f']
        str_array_3 = cppyy.gbl.str_array_3
        for i in range(3):
            for j in range(2):
                assert str_array_3[i][j] == vals[i*2+j]

        vals = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
        str_array_4 = cppyy.gbl.str_array_4
        for i in range(4):
            for j in range(2):
                for k in range(2):
                    assert str_array_4[i][j][k] == vals[i*4+j*2+k]

    def test05_stlstring_and_unicode(self):
        """Mixing unicode and std::string"""

        import cppyy

        uas = cppyy.gbl.UnicodeAndSTL

        actlen = len(u'ℕ'.encode(encoding='UTF-8'))
        assert uas.get_size('ℕ')    == actlen
        assert uas.get_size_cr('ℕ') == actlen
        assert uas.get_size_cc('ℕ') == actlen

        assert uas.get_size_w('ℕ')   == 1
        assert uas.get_size_wcr('ℕ') == 1

        assert str(uas.get_string('ℕ'))     == 'ℕ'
        assert str(uas.get_string_cr('ℕ'))  == 'ℕ'
        assert str(uas.get_string_cc('ℕ'))  == 'ℕ'

        if sys.hexversion >= 0x3000000:
            assert uas.get_string_w('ℕ')   == 'ℕ'
            assert uas.get_string_wcr('ℕ') == 'ℕ'
        else:
            assert uas.get_string_w('ℕ').encode(encoding='UTF-8')   == 'ℕ'
            assert uas.get_string_wcr('ℕ').encode(encoding='UTF-8') == 'ℕ'

        bval = u'ℕ'.encode(encoding='UTF-8')
        actlen = len(bval)
        assert uas.get_size(bval)    == actlen
        assert uas.get_size_cr(bval) == actlen
        assert uas.get_size_cc(bval) == actlen

        assert str(uas.get_string(bval))    == 'ℕ'
        assert str(uas.get_string_cr(bval)) == 'ℕ'
        assert str(uas.get_string_cc(bval)) == 'ℕ'

    def test06_stlstring_bytes_and_text(self):
        """Mixing of bytes and str"""

        import cppyy

        cppyy.cppdef("""\
        namespace PyBytesTest {
        std::string string_field = "";
        }""")

        ns = cppyy.gbl.PyBytesTest
        assert type(ns.string_field) == cppyy.gbl.std.string

        ns.string_field = b'\xe9'
        assert repr(ns.string_field) == repr(b'\xe9')
        assert str(ns.string_field)  == str(b'\xe9')       # b/c fails to decode

    def test07_stlstring_in_dictionaries(self):
        """Mixing str and std::string as dictionary keys"""

        import cppyy

        x = cppyy.gbl.std.string("x")
        d = { x : 0 }

        assert d[x] == 0
        assert d['x'] == 0

    def test08_string_operators(self):
        """Mixing of C++ and Python types in global operators"""

        import cppyy

      # note order in these checks: the first is str then unicode, the other is
      # the reverse; this exercises both paths (once resolved, the operator+ can
      # handle both str and unicde
        s1 = cppyy.gbl.std.string("Hello")
        s2 = ", World!"

        assert s1+s2 == "Hello, World!"
        assert s2+s1 == ", World!Hello"

        s2 = u", World!"
        assert s1+s2 == "Hello, World!"
        assert s2+s1 == ", World!Hello"

        s1 = cppyy.gbl.std.wstring("Hello")

        assert s1+s2 == "Hello, World!"
        assert s2+s1 == ", World!Hello"

        s2 = ", World!"
        assert s1+s2 == "Hello, World!"
        assert s2+s1 == ", World!Hello"

    def test09_string_as_str_bytes(self):
        """Python-style methods of str/bytes on std::string"""

        import cppyy

        S = cppyy.gbl.std.string

      # check that object.method(*args) returns result
        def EQ(result, init, methodname, *args):
            assert getattr(S(init), methodname)(*args) == result

      # npos plays a dual role: both C++ and Python type checking
        assert S.npos == -1
        assert S.npos !=  0
        assert S.npos == S.size_type(-1)

      # -- method decode
        s = S(u'\xe9')
        assert s.decode('utf-8')           == u'\xe9'
        assert s.decode('utf-8', "strict") == u'\xe9'
        assert s.decode(encoding='utf-8')  == u'\xe9'

      # -- method split (only Python)
        assert S("a b c").split() == ['a', 'b', 'c']

      # -- method replace (from Python's string tests)

      # Operations on the empty string
        EQ("", "", "replace", "", "")
        EQ("A", "", "replace", "", "A")
        EQ("", "", "replace", "A", "")
        EQ("", "", "replace", "A", "A")
        EQ("", "", "replace", "", "", 100)
        EQ("", "", "replace", "", "", sys.maxsize)

      # interleave (from=="", 'to' gets inserted everywhere)
        EQ("A", "A", "replace", "", "")
        EQ("*A*", "A", "replace", "", "*")
        EQ("*1A*1", "A", "replace", "", "*1")
        EQ("*-#A*-#", "A", "replace", "", "*-#")
        EQ("*-A*-A*-", "AA", "replace", "", "*-")
        EQ("*-A*-A*-", "AA", "replace", "", "*-", -1)
        EQ("*-A*-A*-", "AA", "replace", "", "*-", sys.maxsize)
        EQ("*-A*-A*-", "AA", "replace", "", "*-", 4)
        EQ("*-A*-A*-", "AA", "replace", "", "*-", 3)
        EQ("*-A*-A", "AA", "replace", "", "*-", 2)
        EQ("*-AA", "AA", "replace", "", "*-", 1)
        EQ("AA", "AA", "replace", "", "*-", 0)

      # -- methods find and rfind
        s = S('aap')

      # Python style
        assert s.find('a')  == 0
        assert s.find('a')  != s.npos
        assert s.rfind('a') == 1
        assert s.rfind('a') != s.npos
        assert s.find('c')   < 0
        assert s.find('c')  == s.npos
        assert s.rfind('c')  < 0
        assert s.rfind('c') == s.npos

    def test10_string_in_repr_and_str_bytes(self):
        """Special cases for __str__/__repr__"""

        import cppyy

        cppyy.cppdef(r"""\
        namespace ReprAndStr {

        struct Test1 {
            const char* __repr__() const { return "Test1"; }
            const char* __str__()  const { return "Test1"; }
        };

        struct Test2 {
            std::string __repr__() const { return "Test2"; }
            std::string __str__()  const { return "Test2"; }
        };

        struct Test3 {
            std::wstring __repr__() const { return L"Test3"; }
            std::wstring __str__()  const { return L"Test3"; }
        };
        }""")

        ns = cppyy.gbl.ReprAndStr

        assert str (ns.Test1()) == "Test1"
        assert repr(ns.Test1()) == "Test1"

        assert str (ns.Test2()) == "Test2"
        assert repr(ns.Test2()) == "Test2"

        assert str (ns.Test3()) == "Test3"
        assert repr(ns.Test3()) == "Test3"


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
            assert not a
            for i in range(self.N):
                a.push_back(i)
            assert a

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
        assert not a
        for arg in a:
            pass

    def test03_replacement_of_eq(self):
        """A global templated function should work as a method"""

        import cppyy

        cppyy.cppdef("""template<class C>
            bool cont_eq(const typename C::iterator& a, const typename C::iterator& b) {
                return a != b;
            }""")

        a = cppyy.gbl.std.list[int]()
        icls = a.begin().__class__
        oldeq = icls.__eq__
        icls.__eq__ = cppyy.gbl.cont_eq[cppyy.gbl.std.list[int]]
        assert not (a.begin() == a.end())

        a = cppyy.gbl.std.list[float]()
        assert a.begin() == a.end()
        assert not cppyy.gbl.cont_eq[cppyy.gbl.std.list[float]](a.begin(), a.begin())
        a.push_back(1)
        assert     cppyy.gbl.cont_eq[cppyy.gbl.std.list[float]](a.begin(), a.end())

        icls.__eq__ = oldeq

    def test04_iter_of_iter(self):
        """Iteration using iter()"""

        import cppyy

        l = cppyy.gbl.std.list['int']((1, 2, 3))
        assert [x for x in l] == [1, 2, 3]

        i = 1
        for a in iter(l):
            assert a == i
            i += 1

    def test05_list_cpp17_style(self):
        """C++17 style initialization of std::list"""

        import cppyy

        l = [1.0, 2.0, 3.0]
        v = cppyy.gbl.std.list(l)
        assert list(l) == l

    def test06_convert_list_of_strings(self):
        """Convert list of strings from C++ to Python types"""

        import cppyy

        contents = ["aap", "noot", "mies"]

        l = cppyy.gbl.std.list[str]()
        l += contents

      # the following used to fail on Windows (TODO: currently worked around in
      # cppyy-backend/clingwrapper; need to see whether Clang9 solves the issue)
        assert [str(x) for x in l] == contents


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

        for mtype in (std.map, std.unordered_map):
            a = mtype(int, int)()
            for i in range(self.N):
                a[i] = i
                assert a[i] == i

            assert len(a) == self.N

            itercount = 0
            for key, value in a:
                assert key == value
                itercount += 1
            assert itercount == len(a)
            if mtype == std.map:            # ordered
                assert key   == self.N-1
                assert value == self.N-1

            # add a variation, just in case
            m = mtype(int, int)()
            for i in range(self.N):
                m[i] = i*i
                assert m[i] == i*i

            itercount = 0
            for key, value in m:
                assert key*key == value
                itercount += 1
            assert itercount == len(m)
            if mtype == std.map:            # ordered
                assert key   == self.N-1
                assert value == (self.N-1)*(self.N-1)

    def test02_keyed_maptype(self):
        """Test access to a map<std::string,int>"""

        import cppyy
        std = cppyy.gbl.std

        for mtype in (std.map, std.unordered_map):
            a = mtype(std.string, int)()
            assert not a
            for i in range(self.N):
                a[str(i)] = i
                assert a[str(i)] == i
            assert a

        assert len(a) == self.N

    def test03_empty_maptype(self):
        """Test behavior of empty map<int,int>"""

        import cppyy
        std = cppyy.gbl.std

        for mtype in (std.map, std.unordered_map):
            m = mtype(int, int)()
            assert not m
            for key, value in m:
                pass

    def test04_unsignedvalue_typemap_types(self):
        """Test assignability of maps with unsigned value types"""

        import cppyy, math, sys
        std = cppyy.gbl.std

        for mtype in (std.map, std.unordered_map):
            mui = mtype(str, 'unsigned int')()
            mui['one'] = 1
            assert mui['one'] == 1
            raises(ValueError, mui.__setitem__, 'minus one', -1)

            # UInt_t is always 32b, maxvalue is sys.maxint/maxsize and follows system int
            maxint32 = int(math.pow(2,31)-1)
            mui['maxint'] = maxint32 + 3
            assert mui['maxint'] == maxint32 + 3

            mul = mtype(str, 'unsigned long')()
            mul['two'] = 2
            assert mul['two'] == 2
            mul['maxint'] = maxvalue + 3
            assert mul['maxint'] == maxvalue + 3

            raises(ValueError, mul.__setitem__, 'minus two', -2)

    def test05_STL_like_class_indexing_overloads(self):
        """Test overloading of operator[] in STL like class"""

        import cppyy
        stl_like_class = cppyy.gbl.stl_like_class

        a = stl_like_class(int)()
        assert a["some string" ] == 'string'
        assert a[3.1415] == 'double'

    def test06_initialize_from_dict(self):
        """Test std::map initializion from Python dict"""

        import cppyy
        std = cppyy.gbl.std

        for mtype in (std.map, std.unordered_map):
            m = mtype[str, int]({'1' : 1, '2' : 2})

            assert m['1'] == 1
            assert m['2'] == 2

            with raises(TypeError):
                m = mtype[int, str]({'1' : 1, '2' : 2})

    def test07_map_cpp17_style(self):
        """C++17 style initialization of std::map"""

        if ispypy:
            skip('emulated class crash')

        import cppyy
        std = cppyy.gbl.std

        for mtype in (std.map, std.unordered_map):
            m = mtype({'1': 2, '2':1})
            assert m['1'] == 2
            assert m['2'] == 1

    def test08_map_derived_objects(self):
        """Enter derived objects through an initializer list"""

        import cppyy
        std = cppyy.gbl.std

        cppyy.cppdef("""\
        namespace MapInitializer {
        class Base {
        public:
            virtual ~Base() {}
        };

        class Derived : public Base { };
        } """)

        ns = cppyy.gbl.MapInitializer

        for mtype in (std.map, std.unordered_map):
          # dictionary style initializer; allow derived through assignment (this may slice
          # but that is the choice of the program; in this case it's fine as both are the
          # same size
            m = mtype['std::string', ns.Base]({"aap": ns.Base(), "noot": ns.Base()})
            assert len(m) == 2

            m = mtype['std::string', ns.Base]({"aap": ns.Derived(), "noot": ns.Derived()})
            assert len(m) == 2

          # similar but now initialize through the initializer_list of pairs style
            m = mtype['std::string', ns.Base]((("aap", ns.Base()),))
            assert len(m) == 1
            m = mtype['std::string', ns.Base]([("aap", ns.Base()),])   # list instead of tuple
            assert len(m) == 1

            m = mtype['std::string', ns.Base]((("aap", ns.Base()), ("noot", ns.Base())))
            assert len(m) == 2

            m = mtype['std::string', ns.Base]((("aap", ns.Derived()), ("noot", ns.Derived())))
            assert len(m) == 2


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

    def test02_STL_like_class_iterators(self):
        """Test the iterator protocol mapping for an STL like class"""

        import cppyy

        a = cppyy.gbl.stl_like_class(int)()
        assert len(a) == 4
        for i, j in enumerate(a):
            assert i == j

        assert i == len(a)-1

        for cls in [cppyy.gbl.stl_like_class2, cppyy.gbl.stl_like_class3]:
            b = cls[float, 2]()
            b[0] = 27; b[1] = 42
            limit = len(b)+1
            for x in b:
                limit -= 1
                assert limit and "iterated too far!"
                assert x in [27, 42]
            assert x == 42
            del x, b

        for num in [4, 5, 6, 7]:
            cls = getattr(cppyy.gbl, 'stl_like_class%d' % num)
            count = 0
            for i in cls():
                count += 1
            assert count == 10

        for cls in [cppyy.gbl.stl_like_class8, cppyy.gbl.stl_like_class9]:
            b = cls()
            for i in [1, 2, 3]:
                b.push_back(i)

            assert len(b) == 3
            assert sum(b) == 6

    def test03_stllike_preinc(self):
        """STL-like class with preinc by-ref returns"""

        import cppyy

        cppyy.cppdef("""\
        namespace PreIncrement {
        struct Token {
            int value;
        };

        struct Iterator {
            Token current;

            bool operator!=(const Iterator& rhs) {
                return rhs.current.value != current.value; }
            const Token& operator*() { return current; }
            Iterator& operator++() {
                current.value++;
                return *this;
            }
        };

        struct Stream {
            Iterator begin() { return Iterator(); }
            Iterator end() { return Iterator{10}; }
        }; }""")

        ns = cppyy.gbl.PreIncrement

        stream = ns.Stream()
        assert [x.value for x in stream] == list(range(10))

        stream = ns.Stream()
        it = iter(stream)
        assert next(it).value == 0
        assert next(it).value == 1
        assert next(it).value == 2

    def test04_stllike_confusing_name(self):
        """Having "iterator" in the container name used to fail"""

        import cppyy

        cppyy.cppdef("""\
        namespace ConstainerOfIterators {
        template<typename TI>
        class iterator_range {
        public:
            iterator_range(TI b, TI e) : m_begin(b), m_end(e) {}

            TI begin() const { return m_begin; }
            TI end() const { return m_end; }

        private:
            TI m_begin;
            TI m_end;
        };

        class iterator {
        public:
            iterator(const int* first) : current(first) {}

            bool operator==(const iterator& other) const { return current == other.current; }
            const int& operator*() { return *current; }
            iterator& operator++() { ++current; return *this; }

        private:
            const int* current;
        };

        class A {
        public:
            A() : fArray{1, 2, 3, 4} {}

            iterator_range<iterator> members() {
                return iterator_range<iterator>(&fArray[0], &fArray[3]+1);
            }

        private:
            std::array<int, 4> fArray;
        }; }""")

        ns = cppyy.gbl.ConstainerOfIterators

        a = ns.A()
        m = a.members()

        assert [x for x in m] == [1, 2, 3, 4]


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

        if ispypy:
            raise RuntimeError("test fails with crash")
        # test assignment
        assert a[2]
        a[2] = gbl.ArrayTest.Point(6, 7)
        assert a[2].px == 6
        assert a[2].py == 7

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
        if ispypy:
            raise RuntimeError("test fails with crash")
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

    def test04_array_from_aggregate(self):
        """Initialize an array from an aggregate contructor"""

        import cppyy

        l = [1.0, 1.0, 1.0]
        t = cppyy.gbl.std.array["double",3](l)
        assert list(t) == l

        with raises(ValueError):
            cppyy.gbl.std.array["double",3]([1.0, 1.0, 1.0, 1.0])

        with raises(TypeError):
            cppyy.gbl.std.array["double",3](['a', 1.0, 1.0])


class TestSTLSTRING_VIEW:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_string_through_string_view(self):
        """Usage of std::string_view as formal argument"""

        import cppyy
        if cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;") <= 201402:
            # string_view exists as of C++17
            return
        countit = cppyy.gbl.StringViewTest.count
        countit_cr = cppyy.gbl.StringViewTest.count_cr

        assert countit("aap")    == 3
        assert countit_cr("aap") == 3
        s = cppyy.gbl.std.string("noot")
        assert countit(s)    == 4
        assert countit_cr(s) == 4
        v = cppyy.gbl.std.string_view(s.data(), s.size())
        assert v[0] == 'n'
        assert countit(v)    == 4
        assert countit_cr(v) == 4

    def test02_string_view_from_unicode(self):
        """Life-time management of converted unicode strings"""

        import cppyy, gc

        # view on (converted) unicode
        text = cppyy.gbl.std.string_view('''\
        The standard Lorem Ipsum passage, used since the 1500s

        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
         tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
         quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
         consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
         cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
         non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."\
        ''')

        gc.collect()    # likely erases Python-side temporary

        assert "Lorem ipsum dolor sit amet" in str(text)

        # view on bytes
        text = cppyy.gbl.std.string_view(b'''\
        The standard Lorem Ipsum passage, used since the 1500s

        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
         tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
         quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
         consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
         cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
         non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."\
        ''')

        gc.collect()    # id.

        assert "Lorem ipsum dolor sit amet" in str(text)


class TestSTLDEQUE:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

    def test01_deque_byvalue_regression(self):
        """Return by value of a deque used to crash"""

        import cppyy
        assert cppyy.cppdef("""std::deque<long double> emptyf() {
            std::deque<long double> d; d.push_back(0); return d ; }""")
        x = cppyy.gbl.emptyf()
        assert x
        del x

    def test02_deque_cpp17_style(self):
        """C++17 style initialization of std::deque"""

        import cppyy

        l = [1.0, 2.0, 3.0]
        v = cppyy.gbl.std.deque(l)
        assert list(l) == l


class TestSTLSET:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

    def test01_set_iteration(self):
        """Iterate over a set"""

        import cppyy

        s = cppyy.gbl.std.set[int]()
        r = range(self.N)
        for i in r:
            s.insert(i)

        assert len(s) == len(r)
        assert sum(s) == sum(r)

        for i in s:
            assert i in s
            assert i in r

    def test02_set_iterators(self):
        """Access to set iterators and their comparisons"""

        import cppyy

        cppyy.include("iterator")
        s = cppyy.gbl.std.set[int]()

        assert s.begin()  == s.end()
        assert s.rbegin() == s.rend()

        val = 42
        s.insert(val)

        assert len(s) == 1
        assert s.begin().__deref__()  == val
        assert s.rbegin().__deref__() == val

        assert s.begin()  != s.end()
        assert s.begin().__preinc__()  == s.end()
        assert s.rbegin() != s.rend()
        assert s.rbegin().__preinc__() == s.rend()

    def test03_initialize_from_set(self):
        """Test std::set initializion from Python set"""

        import cppyy

        N = 10

        s = cppyy.gbl.std.set[int](set(range(N)))
        assert list(range(N)) == list(s)

        s = cppyy.gbl.std.set[int](range(10))
        assert list(range(N)) == list(s)

        with raises(TypeError):
            s = cppyy.gbl.std.set[int](set([1, "2"]))

        with raises(TypeError):
            s = cppyy.gbl.std.set[int](set(["aap", "noot", "mies"]))

    def test04_set_cpp17_style(self):
        """C++17 style initialization of std::set"""

        import cppyy

        l = [1.0, 2.0, 3.0]
        v = cppyy.gbl.std.set(l)
        assert list(l) == l

    def test05_contains(self):
        """Contains check should not iterate and compare"""

        import cppyy

        assert '__contains__' in cppyy.gbl.std.set[int].__dict__

        S = cppyy.gbl.std.set[int](range(2**20))

        assert 1337 in S
        assert not (2**30 in S)

      # not a true test, but this'll take a noticable amount of time (>1min) if
      # there is a regression somehow
        for i in range(100):
            assert not (2**30 in S)


class TestSTLTUPLE:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

    def test01_tuple_creation_and_access(self):
        """Create tuples and access their elements"""

        import cppyy
        std = cppyy.gbl.std

        t1 = std.make_tuple(1, 'a')
        assert t1
        assert std.get[0](t1) == 1
        assert std.get[1](t1) == 'a'

        t2 = std.make_tuple(1, 'a')
        assert t1 == t2

        t3 = std.make_tuple[int, 'char'](1, 'a')
        assert t3
        assert std.get[0](t3) == 1
        assert std.get[1](t3) == 'a'

        # assert t1 != t3     # fails to link (?!)

        t4 = std.make_tuple(7., 1, 'b')
        assert t4
        assert std.get[0](t4) == 7.
        assert std.get[1](t4) == 1
        assert std.get[2](t4) == 'b'

        v = std.vector[int](range(self.N))
        t5 = std.make_tuple(v, False)
        assert std.get[0](t5).size() == self.N
        assert not std.get[1](t5)

        # TODO: should be easy enough to add iterators over std::tuple?

    def test02_tuple_size(self):
        """Usage of tuple_size helper class"""

        import cppyy
        std = cppyy.gbl.std

        t = std.make_tuple("aap", 42, 5.)
        assert std.tuple_size(type(t)).value == 3

    def test03_tuple_iter(self):
        """Pack/unpack tuples"""

        import cppyy, ctypes
        std = cppyy.gbl.std

        t = std.make_tuple(1, '2', 5.)
        assert len(t) == 3

        a, b, c = t
        assert a == 1
        assert b == '2'
        assert c == 5.

    def test04_tuple_lifeline(self):
        """Tuple memory management"""

        import cppyy, gc
        std = cppyy.gbl.std

        cppyy.cppdef("""\
        namespace TupleLifeLine {
        struct Simple {
          Simple() : fInt(42) {}
          Simple(const Simple&) = default;
          Simple& operator=(const Simple&) = default;
          ~Simple() { fInt = -1; }
          int fInt;
        }; }""")

        Simple = cppyy.gbl.TupleLifeLine.Simple

        s1, s2 = std.make_tuple(Simple(), Simple())

        gc.collect()

        assert s1.fInt == 42
        assert s2.fInt == 42


class TestSTLPAIR:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N

    def test01_pair_pack_unpack(self):
        """Pack/unpack pairs"""

        import cppyy
        std = cppyy.gbl.std

        p = std.make_pair(1, 2)
        a, b = p

        assert a == 1
        assert b == 2


class TestSTLEXCEPTION:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.stltypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_basics(self):
        """Test behavior of std::exception derived classes"""

        import cppyy

        assert issubclass(cppyy.gbl.std.exception, BaseException)
        assert cppyy.gbl.std.exception is cppyy.gbl.std.exception

        MyError = cppyy.gbl.MyError
        assert MyError is cppyy.gbl.MyError
        assert issubclass(MyError, BaseException)
        assert issubclass(MyError, cppyy.gbl.std.exception)
        assert MyError.__name__     == 'MyError'
        assert MyError.__cpp_name__ == 'MyError'
        assert MyError.__module__   == 'cppyy.gbl'

        YourError = cppyy.gbl.YourError
        assert YourError is cppyy.gbl.YourError
        assert issubclass(YourError, MyError)
        assert YourError.__name__     == 'YourError'
        assert YourError.__cpp_name__ == 'YourError'
        assert YourError.__module__   == 'cppyy.gbl'

        MyError = cppyy.gbl.ErrorNamespace.MyError
        assert MyError is not cppyy.gbl.MyError
        assert MyError is cppyy.gbl.ErrorNamespace.MyError
        assert issubclass(MyError, BaseException)
        assert issubclass(MyError, cppyy.gbl.std.exception)
        assert MyError.__name__     == 'MyError'
        assert MyError.__cpp_name__ == 'ErrorNamespace::MyError'
        assert MyError.__module__   == 'cppyy.gbl.ErrorNamespace'

        YourError = cppyy.gbl.ErrorNamespace.YourError
        assert YourError is not cppyy.gbl.YourError
        assert YourError is cppyy.gbl.ErrorNamespace.YourError
        assert issubclass(YourError, MyError)
        assert YourError.__name__     == 'YourError'
        assert YourError.__cpp_name__ == 'ErrorNamespace::YourError'
        assert YourError.__module__   == 'cppyy.gbl.ErrorNamespace'

    def test02_raising(self):
        """Raise a C++ std::exception derived class as a Python excption"""

        import cppyy

        assert issubclass(cppyy.gbl.MyError, BaseException)

        def raiseit(cls):
            raise cls('Oops')

        with raises(Exception):
            raiseit(cppyy.gbl.MyError)

        with raises(cppyy.gbl.MyError):
            raiseit(cppyy.gbl.MyError)

        try:
            raiseit(cppyy.gbl.MyError)
        except Exception as e:
            assert e.what() == 'Oops'

        try:
            raiseit(cppyy.gbl.MyError)
        except cppyy.gbl.MyError as e:
            assert e.what() == 'Oops'

        try:
            raiseit(cppyy.gbl.YourError)
        except cppyy.gbl.MyError as e:
            assert e.what() == 'Oops'

        try:
            raiseit(cppyy.gbl.YourError)
        except cppyy.gbl.YourError as e:
            assert e.what() == 'Oops'

    def test03_memory(self):
        """Memory handling of C++ c// helper for exception base class testing"""

        import cppyy, gc

        MyError   = cppyy.gbl.MyError
        YourError = cppyy.gbl.YourError

        gc.collect()
        assert cppyy.gbl.GetMyErrorCount() == 0

        m = MyError('Oops')
        assert cppyy.gbl.GetMyErrorCount() == 1
        del m
        gc.collect()
        assert cppyy.gbl.GetMyErrorCount() == 0

        def raiseit(cls):
            raise cls('Oops')

        def run_raiseit(t1, t2):
            try:
                raiseit(t1)
            except t2 as e:
                assert e.what() == 'Oops'
                return
            assert not "should not reach this point"

        for t1, t2 in [(MyError,   Exception),
                       (MyError,   MyError),
                       (YourError, MyError),
                       (YourError, YourError)]:
            with raises(t2):
                raiseit(t1)
            gc.collect()
            assert cppyy.gbl.GetMyErrorCount() == 0

            run_raiseit(t1, t2)
            gc.collect()
            assert cppyy.gbl.GetMyErrorCount() == 0

        gc.collect()
        assert cppyy.gbl.GetMyErrorCount() == 0

    def test04_from_cpp(self):
        """Catch C++ exceptiosn from C++"""

        if ispypy:
            skip('currently terminates')

        import cppyy, gc

        gc.collect()
        assert cppyy.gbl.GetMyErrorCount() == 0

        with raises(cppyy.gbl.MyError):
            cppyy.gbl.ErrorNamespace.throw_error(0)

        with raises(cppyy.gbl.MyError):
            cppyy.gbl.ErrorNamespace.throw_error(1)

        with raises(cppyy.gbl.YourError):
            cppyy.gbl.ErrorNamespace.throw_error(1)

        with raises(cppyy.gbl.ErrorNamespace.MyError):
            cppyy.gbl.ErrorNamespace.throw_error(2)

        with raises(cppyy.gbl.ErrorNamespace.MyError):
            cppyy.gbl.ErrorNamespace.throw_error(3)

        with raises(cppyy.gbl.ErrorNamespace.YourError):
            cppyy.gbl.ErrorNamespace.throw_error(3)

        gc.collect()
        assert cppyy.gbl.GetMyErrorCount() == 0
