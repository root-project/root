import py
from pytest import mark, raises, skip
from support import (
    IS_MAC,
    ispypy,
    pylong,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/example01Dict"))


def setup_module(mod):
    setup_make("example01")


class TestPYTHONIFY:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.example01 = cppjit.load_reflection_info(cls.test_dct)

    def test01_load_dictionary_cache(self):
        """Test whether loading a dictionary twice results in the same object"""

        import cppjit

        lib2 = cppjit.load_reflection_info(self.test_dct)
        assert self.example01 is lib2

    def test02_finding_classes(self):
        """Test the lookup of a class, and its caching"""

        import cppjit

        example01_class = cppjit.gbl.example01
        cl2 = cppjit.gbl.example01
        assert example01_class is cl2

        with raises(AttributeError):
            cppjit.gbl.nonexistingclass

    def test03_calling_static_functions(self):
        """Test calling of static methods"""

        import cppjit

        example01_class = cppjit.gbl.example01
        res = example01_class.staticAddOneToInt(1)
        assert res == 2

        res = example01_class.staticAddOneToInt(pylong(1))
        assert res == 2
        res = example01_class.staticAddOneToInt(1, 2)
        assert res == 4
        res = example01_class.staticAddOneToInt(-1)
        assert res == 0
        maxint32 = int(2**31 - 1)
        res = example01_class.staticAddOneToInt(maxint32 - 1)
        assert res == maxint32
        res = example01_class.staticAddOneToInt(maxint32)
        assert res == -maxint32 - 1

        raises(TypeError, example01_class.staticAddOneToInt, 1, [])
        raises(TypeError, example01_class.staticAddOneToInt, 1.0)
        raises(TypeError, example01_class.staticAddOneToInt, maxint32 + 1)
        res = example01_class.staticAddToDouble(0.09)
        assert res == 0.09 + 0.01

        res = example01_class.staticAtoi("1")
        assert res == 1

        res = example01_class.staticStrcpy("aap")  # TODO: this leaks
        assert res == "aap"
        res = example01_class.staticStrcpy("aap")  # TODO: id.
        assert res == "aap"
        raises(TypeError, example01_class.staticStrcpy, 1.0)  # TODO: id.

    def test04_constructing_and_calling(self):
        """Test object and method calls"""

        import cppjit

        example01_class = cppjit.gbl.example01
        assert example01_class.getCount() == 0
        instance = example01_class(7)
        assert example01_class.getCount() == 1
        res = instance.addDataToInt(4)
        assert res == 11
        res = instance.addDataToInt(-4)
        assert res == 3
        instance.__destruct__()
        assert example01_class.getCount() == 0
        raises(ReferenceError, instance.addDataToInt, 4)

        instance = example01_class(7)
        instance2 = example01_class(8)
        assert example01_class.getCount() == 2
        instance.__destruct__()
        assert example01_class.getCount() == 1
        instance2.__destruct__()
        assert example01_class.getCount() == 0

        instance = example01_class(13)
        res = instance.addDataToDouble(16)
        assert round(res - 29, 8) == 0.0
        instance.__destruct__()
        instance = example01_class(-13)
        res = instance.addDataToDouble(16)
        assert round(res - 3, 8) == 0.0
        instance.__destruct__()

        instance = example01_class(42)
        assert example01_class.getCount() == 1

        res = instance.addDataToAtoi("13")
        assert res == 55

        res = instance.addToStringValue("12")  # TODO: this leaks
        assert res == "54"
        res = instance.addToStringValue("-12")  # TODO: this leaks
        assert res == "30"

        res = instance.staticAddOneToInt(pylong(1))
        assert res == 2

        instance.__destruct__()
        assert example01_class.getCount() == 0

    def test05_passing_object_by_pointer(self):
        """Pass object by pointer"""

        import cppjit

        example01_class = cppjit.gbl.example01
        payload_class = cppjit.gbl.payload

        e = example01_class(14)
        pl = payload_class(3.14)
        assert round(pl.getData() - 3.14, 8) == 0

        example01_class.staticSetPayload(pl, 41.0)
        assert pl.getData() == 41.0
        example01_class.staticSetPayload(pl, 43.0)
        assert pl.getData() == 43.0
        e.staticSetPayload(pl, 45.0)
        assert pl.getData() == 45.0

        e.setPayload(pl)
        assert round(pl.getData() - 14.0, 8) == 0

        pl.__destruct__()
        e.__destruct__()
        assert example01_class.getCount() == 0

    def test06_returning_object_by_pointer(self):
        """Return an object py pointer"""

        import cppjit

        example01_class = cppjit.gbl.example01
        payload_class = cppjit.gbl.payload

        pl = payload_class(3.14)
        assert round(pl.getData() - 3.14, 8) == 0

        pl2 = example01_class.staticCyclePayload(pl, 38.0)
        assert pl2.getData() == 38.0

        e = example01_class(14)

        pl2 = e.cyclePayload(pl)
        assert round(pl2.getData() - 14.0, 8) == 0

        pl.__destruct__()
        e.__destruct__()
        assert example01_class.getCount() == 0

    def test07_returning_object_by_value(self):
        """Return an object by value"""

        import cppjit

        example01_class = cppjit.gbl.example01
        payload_class = cppjit.gbl.payload

        pl = payload_class(3.14)
        assert round(pl.getData() - 3.14, 8) == 0

        pl2 = example01_class.staticCopyCyclePayload(pl, 38.0)
        assert pl2.getData() == 38.0
        pl2.__destruct__()

        e = example01_class(14)

        pl2 = e.copyCyclePayload(pl)
        assert round(pl2.getData() - 14.0, 8) == 0
        pl2.__destruct__()

        pl.__destruct__()
        e.__destruct__()
        assert example01_class.getCount() == 0

    def test08_global_functions(self):
        """Call a global function"""

        import cppjit

        assert cppjit.gbl.globalAddOneToInt(3) == 4  # creation lookup
        assert cppjit.gbl.globalAddOneToInt(3) == 4  # cached lookup

        assert cppjit.gbl.ns_example01.globalAddOneToInt(4) == 5
        assert cppjit.gbl.ns_example01.globalAddOneToInt(4) == 5

    def test09_memory(self):
        """Test proper C++ destruction by the garbage collector"""

        import gc

        import cppjit

        example01_class = cppjit.gbl.example01
        payload_class = cppjit.gbl.payload

        pl = payload_class(3.14)
        assert payload_class.count == 1
        assert round(pl.getData() - 3.14, 8) == 0

        pl2 = example01_class.staticCopyCyclePayload(pl, 38.0)
        assert payload_class.count == 2
        assert pl2.getData() == 38.0
        pl2 = None
        gc.collect()
        assert payload_class.count == 1

        e = example01_class(14)

        pl2 = e.copyCyclePayload(pl)
        assert payload_class.count == 2
        assert round(pl2.getData() - 14.0, 8) == 0
        pl2 = None
        gc.collect()
        assert payload_class.count == 1

        pl = None
        e = None
        gc.collect()
        assert payload_class.count == 0
        assert example01_class.getCount() == 0

        pl = payload_class(3.14)
        pl_a = example01_class.staticCyclePayload(pl, 66.0)
        pl_a.getData() == 66.0
        assert payload_class.count == 1
        pl_a = None
        pl = None
        gc.collect()
        assert payload_class.count == 0

        # TODO: need ReferenceError on touching pl_a

    @mark.xfail(condition=IS_MAC, reason="Fails in OSX")
    def test10_default_arguments(self):
        """Test propagation of default function arguments"""

        import cppjit

        a = cppjit.gbl.ArgPasser()

        # NOTE: when called through the stub, default args are fine
        f = a.stringRef
        s = cppjit.gbl.std.string
        assert f(s("aap"), 0, s("noot")) == "aap"
        assert f(s("noot"), 1) == "default"
        assert f(s("mies")) == "mies"

        for itype in ["short", "ushort", "int", "uint", "long", "ulong"]:
            g = getattr(a, "%sValue" % itype)
            raises(TypeError, g, 1, 2, 3, 4, 6)
            assert g(11, 0, 12, 13) == 11
            assert g(11, 1, 12, 13) == 12
            assert g(11, 1, 12) == 12
            assert g(11, 2, 12) == 2
            assert g(11, 1) == 1
            assert g(11, 2) == 2
            assert g(11) == 11

        for ftype in ["float", "double"]:
            g = getattr(a, "%sValue" % ftype)
            raises(TypeError, g, 1.0, 2, 3.0, 4.0, 6.0)
            assert g(11.0, 0, 12.0, 13.0) == 11.0
            assert g(11.0, 1, 12.0, 13.0) == 12.0
            assert g(11.0, 1, 12.0) == 12.0
            assert g(11.0, 2, 12.0) == 2.0
            assert g(11.0, 1) == 1.0
            assert g(11.0, 2) == 2.0
            assert g(11.0) == 11.0

    def test11_overload_on_arguments(self):
        """Test functions overloaded on arguments"""

        import cppjit

        e = cppjit.gbl.example01(1)

        assert e.addDataToInt(2) == 3
        assert e.overloadedAddDataToInt(3) == 4
        assert e.overloadedAddDataToInt(4, 5) == 10
        assert e.overloadedAddDataToInt(6, 7, 8) == 22

    def test12_typedefs(self):
        """Test access and use of typedefs"""

        import cppjit

        assert cppjit.gbl.example01 == cppjit.gbl.example01_t

    def test13_underscore_in_class_name(self):
        """Test recognition of '_' as part of a valid class name"""

        import cppjit

        assert cppjit.gbl.z_ == cppjit.gbl.z_

        z = cppjit.gbl.z_()

        assert hasattr(z, "myint")
        assert z.gime_z_(z)

    def test14_bound_unbound_calls(self):
        """Test (un)bound method calls"""

        if ispypy:
            skip("segfaults in pypy")

        import cppjit

        raises(TypeError, cppjit.gbl.example01.addDataToInt, 1)

        meth = cppjit.gbl.example01.addDataToInt
        raises(TypeError, meth)
        raises(TypeError, meth, 1)

        e = cppjit.gbl.example01(2)
        assert 5 == meth(e, 3)

    def test15_installable_function(self):
        """Test installing and calling global C++ function as python method"""

        import cppjit

        cppjit.gbl.example01.fresh = cppjit.gbl.installableAddOneToInt

        e = cppjit.gbl.example01(0)
        assert 2 == e.fresh(1)
        assert 3 == e.fresh(2)

    def test16_subclassing(self):
        """A sub-class on the python side should have that class as type"""

        import gc

        import cppjit

        gc.collect()

        example01 = cppjit.gbl.example01

        assert example01.getCount() == 0

        o = example01()
        assert type(o) == example01
        assert example01.getCount() == 1
        o.__destruct__()
        assert example01.getCount() == 0

        class MyClass1(example01):
            def myfunc(self):
                return 1

        o = MyClass1()
        assert type(o) == MyClass1
        assert isinstance(o, example01)
        assert example01.getCount() == 1
        assert o.myfunc() == 1
        o.__destruct__()
        assert example01.getCount() == 0

        class MyClass2(example01):
            def __init__(self, what):
                example01.__init__(self)
                self.what = what

        o = MyClass2("hi")
        assert type(o) == MyClass2
        assert example01.getCount() == 1
        assert o.what == "hi"
        o.__destruct__()

        assert example01.getCount() == 0

    def test17_chaining(self):
        """Respective return values of temporaries should not go away"""

        import cppjit

        cppjit.cppdef("""namespace Lifeline {
        struct A1 { A1(int x) : x(x) {} int x; };
        struct A2 { A2(int x) { v.emplace_back(x); } std::vector<A1> v; std::vector<A1>& get() { return v; } };
        struct A3 { A3(int x) { v.emplace_back(x); } std::vector<A2> v; std::vector<A2>& get() { return v; } };
        struct A4 { A4(int x) { v.emplace_back(x); } std::vector<A3> v; std::vector<A3>& get() { return v; } };
        struct A5 { A5(int x) { v.emplace_back(x); } std::vector<A4> v; std::vector<A4>& get() { return v; } };

        A5 gime(int i) { return A5(i); }
        }""")

        assert cppjit.gbl.Lifeline.gime(42).get()[0].get()[0].get()[0].get()[0].x == 42

    def test18_keywords(self):
        """Use of keyword arguments"""

        import cppjit

        cppjit.cppdef("""namespace KeyWords {
        struct A {
            A(std::initializer_list<int> vals) : fVals(vals) {}
            std::vector<int> fVals;
        };

        struct B {
            B() = delete;
            B(const A& in_A, const A& out_A) : fVal(42), fIn(in_A), fOut(out_A) {}
            B(int val, const A& in_A, const A& out_A) : fVal(val), fIn(in_A), fOut(out_A) {}
            int fVal;
            A fIn, fOut;
        };

        int callme(int choice, int a, int b, int c) {
            if (choice == 0) return a;
            if (choice == 1) return b;
            return c;
        }

        struct C {
            int fChoice;
        };

        int callme_c(const C& o, int a, int b, int c) {
            return callme(o.fChoice, a, b, c);
        } }""")

        # constructor and implicit conversion with keywords
        A = cppjit.gbl.KeyWords.A
        B = cppjit.gbl.KeyWords.B

        def verify_b(b, val, ti, to):
            assert b.fVal == val
            assert b.fIn.fVals.size() == len(ti)
            assert tuple(b.fIn.fVals) == ti
            assert b.fOut.fVals.size() == len(to)
            assert tuple(b.fOut.fVals) == to

        b = B(in_A=(256,), out_A=(32,))
        verify_b(b, 42, (256,), (32,))

        b = B(out_A=(32,), in_A=(256,))
        verify_b(b, 42, (256,), (32,))

        with raises(TypeError):
            b = B(in_B=(256,), out_A=(32,))

        b = B(17, in_A=(23,), out_A=(78,))
        verify_b(b, 17, (23,), (78,))

        with raises(TypeError):
            b = B(17, val=23, out_A=(78,))

        with raises(TypeError):
            b = B(17, out_A=(78,))

        # global function with keywords
        callme = cppjit.gbl.KeyWords.callme
        for i in range(3):
            assert callme(i, a=1, b=2, c=3) == i + 1
            assert callme(i, b=2, c=3, a=1) == i + 1
            assert callme(i, c=3, a=1, b=2) == i + 1

        with raises(TypeError):
            callme(0, a=1, b=2, d=3)

        with raises(TypeError):
            callme(0, 1, a=2, c=3)

        with raises(TypeError):
            callme(0, a=1, b=2)

        # global function as method with keywords
        c = cppjit.gbl.KeyWords.C()
        cppjit.gbl.KeyWords.C.callme = cppjit.gbl.KeyWords.callme_c

        for i in range(3):
            c.fChoice = i
            assert c.callme(a=1, b=2, c=3) == i + 1
            assert c.callme(b=2, c=3, a=1) == i + 1
            assert c.callme(c=3, a=1, b=2) == i + 1

        c.fChoice = 0
        with raises(TypeError):
            c.callme(a=1, b=2, d=3)

        with raises(TypeError):
            c.callme(1, a=2, c=3)

        with raises(TypeError):
            c.callme(a=1, b=2)

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test19_keywords_and_defaults(self):
        """Use of keyword arguments mixed with defaults"""

        import cppjit

        cppjit.cppdef("""namespace KeyWordsAndDefaults {
        int foo(int a=10, int b=20, int c=5, int d=4) {
            return a-b/c*d;
        }

        std::string bar(const std::string& a = "a", const std::string& b = "b") {
            return a+b;
        }

        class MyClass {};

        void foobar(const MyClass& m1 = MyClass(), const MyClass& m2 = MyClass()) {
            /* empty */
        }

        bool barfoo(bool opt1=false, bool opt2=true) {
            return opt1 && opt2;
        } }""")

        def pyfoo(a=10, b=20, c=5, d=4):
            return a - b // c * d

        ns = cppjit.gbl.KeyWordsAndDefaults

        assert ns.foo() == pyfoo()
        assert ns.foo(a=100) == pyfoo(a=100)
        assert ns.foo(b=100) == pyfoo(b=100)
        assert ns.foo(a=100, b=200) == pyfoo(a=100, b=200)
        assert ns.foo(a=100, b=200, d=0) == pyfoo(a=100, b=200, d=0)
        assert ns.foo(b=100, a=200) == pyfoo(b=100, a=200)

        with raises(TypeError):
            ns.foo(1, 2, 3, 4, b=5)

        assert ns.bar() == "ab"
        assert ns.bar(b=" greeting") == "a greeting"

        ns.foobar(m2=ns.MyClass())

        assert not ns.barfoo()
        assert not ns.barfoo(opt2=True)
        assert not ns.barfoo(opt2=False)
        assert ns.barfoo(opt1=True, opt2=True)
        assert not ns.barfoo(opt1=True, opt2=False)


class TestPYTHONIFY_UI:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.example01 = cppjit.load_reflection_info(cls.test_dct)

    def test01_pythonizations(self):
        """Test addition of user-defined pythonizations"""

        import cppjit

        def example01a_pythonize(pyclass, pyname):
            if pyname == "example01a":

                def getitem(self, idx):
                    return self.addDataToInt(idx)

                pyclass.__getitem__ = getitem

        cppjit.py.add_pythonization(example01a_pythonize)

        e = cppjit.gbl.example01a(1)

        assert e[0] == 1
        assert e[1] == 2
        assert e[5] == 6

    def test02_fragile_pythonizations(self):
        """Test pythonizations error reporting"""

        import cppjit

        example01_pythonize = 1
        raises(TypeError, cppjit.py.add_pythonization, example01_pythonize)

    def test03_write_access_to_globals(self):
        """Test overwritability of globals"""

        import cppjit

        oldval = cppjit.gbl.ns_example01.gMyGlobalInt
        assert oldval == 99

        proxy = cppjit.gbl.ns_example01.__class__.__dict__["gMyGlobalInt"]
        cppjit.gbl.ns_example01.gMyGlobalInt = 3
        assert proxy.__get__(proxy, None) == 3

        cppjit.gbl.ns_example01.gMyGlobalInt = oldval


class TestPINNEDCOMPARISON:
    def test01_pinned_base_compares_equal(self):
        """Comparison downcasts to the actual class before comparing addresses"""

        import cppjit
        from cppjit._pythonization import pin_type

        cppjit.cppdef("""\
        namespace PinnedCmp {
            struct B1 { virtual ~B1() {} int a = 1; };
            struct B2 { virtual ~B2() {} int b = 2; };
            struct D : B1, B2 {};
            D  g_d;
            D*  get_d()  { return &g_d; }
            B2* get_b2() { return static_cast<B2*>(&g_d); }
        }""")

        ns = cppjit.gbl.PinnedCmp

        # pinning keeps the B2 proxy from being downcast on creation, so the
        # B2 subobject address is what reaches the comparison
        pin_type(ns.B2)
        d, b2 = ns.get_d(), ns.get_b2()
        assert type(b2).__cpp_name__ == "PinnedCmp::B2"

        assert d == b2
        assert not (d != b2)
