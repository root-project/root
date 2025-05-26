import py, pytest, os
from pytest import raises
from support import setup_make

currpath = os.getcwd()
test_dct = currpath + "/libconversionsDict"


class TestCONVERSIONS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.conversion = cppyy.load_reflection_info(cls.test_dct)

    def test01_implicit_vector_conversions(self):
        """Test implicit conversions of std::vector"""

        import cppyy
        CNS = cppyy.gbl.CNS

        N = 10
        total = float(sum(range(N)))

        v = cppyy.gbl.std.vector['double'](range(N))
        assert CNS.sumit(v) == total
        assert sum(v) == total
        assert CNS.sumit(range(N)) == total

        M = 5
        total = float(sum(range(N)) + sum(range(M, N)))
        v1 = cppyy.gbl.std.vector['double'](range(N))
        v2 = cppyy.gbl.std.vector['double'](range(M, N))
        assert CNS.sumit(v1, v2) == total
        assert sum(v1)+sum(v2)   == total
        assert CNS.sumit(v1, range(M, N))       == total
        assert CNS.sumit(range(N), v2)          == total
        assert CNS.sumit(range(N), range(M, N)) == total

    def test02_memory_handling_of_temporaries(self):
        """Verify that memory of temporaries is properly cleaned up"""

        import cppyy, gc
        CNS, CC = cppyy.gbl.CNS, cppyy.gbl.CNS.Counter

        assert CC.s_count == 0
        c = CC()
        assert c.__python_owns__
        assert CC.s_count == 1
        del c; gc.collect()
        assert CC.s_count == 0

        assert CNS.myhowmany((CC(), CC(), CC())) == 3
        gc.collect()
        assert CC.s_count == 0

        assert CNS.myhowmany((CC(), CC(), CC()), [CC(), CC()]) == 5
        gc.collect()
        assert CC.s_count == 0

    def test03_error_handling(self):
        """Verify error handling"""

        import cppyy, gc
        CNS, CC = cppyy.gbl.CNS, cppyy.gbl.CNS.Counter

        N = 13
        total = sum(range(N))
        assert CNS.sumints(range(N)) == total
        assert CNS.sumit([float(x) for x in range(N)]) == float(total)
        raises(TypeError, CNS.sumints, [float(x) for x in range(N)])
        raises(TypeError, CNS.sumints, list(range(N))+[0.])

        assert CC.s_count == 0

        raises(TypeError, CNS.sumints, list(range(N))+[CC()])
        gc.collect()
        assert CC.s_count == 0

        raises(TypeError, CNS.sumints, range(N), [CC()])
        gc.collect()

        assert CC.s_count == 0
        raises(TypeError, CNS.sumints, [CC()], range(N))
        gc.collect()
        assert CC.s_count == 0

    def test04_implicit_conversion_from_tuple(self):
        """Allow implicit conversions from tuples as arguments {}-like"""

        # Note: fails on windows b/c the assignment operator for strings is
        # template, which ("operator=(std::string)") doesn't instantiate
        import cppyy

        m = cppyy.gbl.std.map[str, str]()
        m.insert(('a', 'b'))      # implicit conversion to std::pair

        assert m['a'] == 'b'

    def test05_bool_conversions(self):
        """Test operator bool() and null pointer behavior"""

        import cppyy

        cppyy.cppdef("""\
        namespace BoolConversions {
        struct Test1 {};
        struct Test2 {
            Test2(bool b) : m_b(b) {}
            explicit operator bool() const { return m_b; }
            bool m_b;
        };

        Test1* CreateNullTest1() { return nullptr; }
        Test2* CreateNullTest2() { return nullptr; }
        }""")

        ns = cppyy.gbl.BoolConversions

        for t in [ns.CreateNullTest1(), ns.CreateNullTest2()]:
            assert not t

        assert     ns.Test1()
        assert     ns.Test2(True)
        assert not ns.Test2(False)


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
