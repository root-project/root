import py
from pytest import mark, raises
from support import IS_CLANG_REPL, IS_CLING, IS_MAC, setup_make

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/conversionsDict"))


def setup_module(mod):
    setup_make("conversions")


class TestCONVERSIONS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.conversion = cppjit.load_reflection_info(cls.test_dct)

    def test01_implicit_vector_conversions(self):
        """Test implicit conversions of std::vector"""

        import cppjit

        CNS = cppjit.gbl.CNS

        N = 10
        total = float(sum(range(N)))

        v = cppjit.gbl.std.vector["double"](range(N))
        assert CNS.sumit(v) == total
        assert sum(v) == total
        assert CNS.sumit(range(N)) == total

        M = 5
        total = float(sum(range(N)) + sum(range(M, N)))
        v1 = cppjit.gbl.std.vector["double"](range(N))
        v2 = cppjit.gbl.std.vector["double"](range(M, N))
        assert CNS.sumit(v1, v2) == total
        assert sum(v1) + sum(v2) == total
        assert CNS.sumit(v1, range(M, N)) == total
        assert CNS.sumit(range(N), v2) == total
        assert CNS.sumit(range(N), range(M, N)) == total

    def test02_memory_handling_of_temporaries(self):
        """Verify that memory of temporaries is properly cleaned up"""

        import gc

        import cppjit

        CNS, CC = cppjit.gbl.CNS, cppjit.gbl.CNS.Counter

        assert CC.s_count == 0
        c = CC()
        assert c.__python_owns__
        assert CC.s_count == 1
        del c
        gc.collect()
        assert CC.s_count == 0

        assert CNS.myhowmany((CC(), CC(), CC())) == 3
        gc.collect()
        assert CC.s_count == 0

        assert CNS.myhowmany((CC(), CC(), CC()), [CC(), CC()]) == 5
        gc.collect()
        assert CC.s_count == 0

    def test03_error_handling(self):
        """Verify error handling"""

        import gc

        import cppjit

        CNS, CC = cppjit.gbl.CNS, cppjit.gbl.CNS.Counter

        N = 13
        total = sum(range(N))
        assert CNS.sumints(range(N)) == total
        assert CNS.sumit([float(x) for x in range(N)]) == float(total)
        raises(TypeError, CNS.sumints, [float(x) for x in range(N)])
        raises(TypeError, CNS.sumints, list(range(N)) + [0.0])

        assert CC.s_count == 0

        raises(TypeError, CNS.sumints, list(range(N)) + [CC()])
        gc.collect()
        assert CC.s_count == 0

        raises(TypeError, CNS.sumints, range(N), [CC()])
        gc.collect()

        assert CC.s_count == 0
        raises(TypeError, CNS.sumints, [CC()], range(N))
        gc.collect()
        assert CC.s_count == 0

    @mark.xfail(
        condition=IS_MAC or IS_CLING, run=IS_CLANG_REPL, reason="Crashes on Cling"
    )
    def test04_implicit_conversion_from_tuple(self):
        """Allow implicit conversions from tuples as arguments {}-like"""

        # Note: fails on windows b/c the assignment operator for strings is
        # template, which ("operator=(std::string)") doesn't instantiate
        import cppjit

        m = cppjit.gbl.std.map[str, str]()
        m.insert(("a", "b"))  # implicit conversion to std::pair

        assert m["a"] == "b"

    def test05_bool_conversions(self):
        """Test operator bool() and null pointer behavior"""

        import cppjit

        cppjit.cppdef("""\
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

        ns = cppjit.gbl.BoolConversions

        for t in [ns.CreateNullTest1(), ns.CreateNullTest2()]:
            assert not t

        assert ns.Test1()
        assert ns.Test2(True)
        assert not ns.Test2(False)

    def test07_mutable_voidp_reference(self):
        """An object can be passed through a non-const void*& argument"""

        import cppjit

        cppjit.cppdef("""\
        namespace VoidPtrRef {
            struct Obj { int v = 5; };
            bool is_same(void*& p, Obj* o) { return p == (void*)o; }
        }""")

        ns = cppjit.gbl.VoidPtrRef
        o = ns.Obj()

        assert ns.is_same(o, o)
