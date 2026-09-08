import py
from pytest import mark, raises, skip
from support import IS_MAC, IS_MAC_ARM, IS_WINDOWS, ispypy, setup_make

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/overloadsDict"))


def setup_module(mod):
    setup_make("overloads")


class TestOVERLOADS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.overloads = cppjit.load_reflection_info(cls.test_dct)

    def test01_class_based_overloads(self):
        """Test functions overloaded on different C++ clases"""

        import cppjit

        a_overload = cppjit.gbl.a_overload
        b_overload = cppjit.gbl.b_overload
        c_overload = cppjit.gbl.c_overload
        d_overload = cppjit.gbl.d_overload

        ns_a_overload = cppjit.gbl.ns_a_overload
        ns_b_overload = cppjit.gbl.ns_b_overload

        assert c_overload().get_int(a_overload()) == 42
        assert c_overload().get_int(b_overload()) == 13
        assert d_overload().get_int(a_overload()) == 42
        assert d_overload().get_int(b_overload()) == 13

        assert c_overload().get_int(ns_a_overload.a_overload()) == 88
        assert c_overload().get_int(ns_b_overload.a_overload()) == -33

        assert d_overload().get_int(ns_a_overload.a_overload()) == 88
        assert d_overload().get_int(ns_b_overload.a_overload()) == -33

    def test02_class_based_overloads_explicit_resolution(self):
        """Test explicitly resolved function overloads"""

        import cppjit

        a_overload = cppjit.gbl.a_overload
        b_overload = cppjit.gbl.b_overload
        c_overload = cppjit.gbl.c_overload
        d_overload = cppjit.gbl.d_overload

        ns_a_overload = cppjit.gbl.ns_a_overload

        c = c_overload()
        raises(TypeError, c.__dispatch__, "get_int", 12)
        raises(LookupError, c.__dispatch__, "get_int", "does_not_exist")
        assert c.__dispatch__("get_int", "a_overload*")(a_overload()) == 42
        assert c_overload.get_int.__overload__("a_overload*")(c, a_overload()) == 42
        assert c.__dispatch__("get_int", "b_overload*")(b_overload()) == 13
        assert c_overload.get_int.__overload__("b_overload*")(c, b_overload()) == 13

        assert c_overload().__dispatch__("get_int", "a_overload*")(a_overload()) == 42
        # TODO: #assert c_overload.__dispatch__('get_int', 'b_overload*')(c, b_overload()) == 13

        d = d_overload()
        assert d.__dispatch__("get_int", "a_overload*")(a_overload()) == 42
        assert d_overload.get_int.__overload__("a_overload*")(d, a_overload()) == 42
        assert d.__dispatch__("get_int", "b_overload*")(b_overload()) == 13
        assert d_overload.get_int.__overload__("b_overload*")(d, b_overload()) == 13

        nb = ns_a_overload.b_overload()
        raises(TypeError, nb.f, c_overload())

    def test03_fragile_class_based_overloads(self):
        """Test functions overloaded on void* and non-existing classes"""

        import cppjit

        more_overloads = cppjit.gbl.more_overloads
        aa_ol = cppjit.gbl.aa_ol
        bb_ol = cppjit.gbl.bb_ol
        cc_ol = cppjit.gbl.cc_ol
        dd_ol = cppjit.gbl.dd_ol

        assert more_overloads().call(aa_ol()) == "aa_ol"
        bb = cppjit.gbl.get_bb_ol()
        assert more_overloads().call(bb) == "bb_ol"
        assert more_overloads().call(cc_ol()) == "cc_ol"
        dd = cppjit.bind_object(cppjit.nullptr, dd_ol)
        with raises(TypeError):
            more_overloads().call(dd)
        dd = cppjit.gbl.get_dd_ol()
        assert more_overloads().call(dd) == "dd_ol"

    def test04_fully_fragile_overloads(self):
        """Test that unknown* is preferred over unknown&"""

        import cppjit

        more_overloads2 = cppjit.gbl.more_overloads2

        bb = cppjit.bind_object(cppjit.nullptr, cppjit.gbl.bb_ol)
        assert more_overloads2().call(bb) == "bb_olptr"

        dd = cppjit.bind_object(cppjit.nullptr, cppjit.gbl.dd_ol)
        assert more_overloads2().call(dd, 1) == "dd_olptr"

    def test05_array_overloads(self):
        """Test functions overloaded on different arrays"""

        import cppjit

        c_overload = cppjit.gbl.c_overload
        d_overload = cppjit.gbl.d_overload

        from array import array

        ai = array("i", [525252])
        assert c_overload().get_int(ai) == 525252
        assert d_overload().get_int(ai) == 525252

        ah = array("h", [25])
        assert c_overload().get_int(ah) == 25
        assert d_overload().get_int(ah) == 25

    def test06_double_int_overloads(self):
        """Test overloads on int/doubles"""

        import cppjit

        more_overloads = cppjit.gbl.more_overloads

        assert more_overloads().call(1) == "int"
        assert more_overloads().call(1.0) == "double"
        assert more_overloads().call1(1) == "int"
        assert more_overloads().call1(1.0) == "double"

    def test07_mean_overloads(self):
        """Adapted test for array overloading"""

        import array

        import cppjit

        cmean = cppjit.gbl.calc_mean

        numbers = [8, 2, 4, 2, 4, 2, 4, 4, 1, 5, 6, 3, 7]
        mean, median = 4.0, 4.0

        for l in ["f", "d", "i", "h", "l"]:
            a = array.array(l, numbers)
            assert round(cmean(len(a), a) - mean, 8) == 0

    def test08_const_non_const_overloads(self):
        """Check selectability of const/non-const overloads"""

        import cppjit

        m = cppjit.gbl.more_overloads3()

        assert m.slice.__overload__(":any:", True)(0) == "const"
        assert m.slice.__overload__(":any:", False)(0) == "non-const"

        allmeths = cppjit.gbl.more_overloads3.slice.__overload__(":any:")
        cppjit.gbl.more_overloads3.slice = allmeths.__overload__(":any:", False)
        cppjit.gbl.more_overloads3.slice_const = allmeths.__overload__(":any:", True)
        del allmeths

        assert m.slice(0) == "non-const"
        assert m, slice(0, 0) == "non-const"
        assert m.slice_const(0) == "const"
        assert m, slice_const(0, 0) == "const"  # noqa: F821

    def test09_bool_int_overloads(self):
        """Check bool/int overloaded calls"""

        import cppjit

        cpp = cppjit.gbl

        cppjit.cppdef("namespace BoolInt1 { int  fff(int i)  { return i; } }")
        cppjit.cppdef("namespace BoolInt1 { bool fff(bool i) { return i; } }")

        assert type(cpp.BoolInt1.fff(0)) == int
        assert type(cpp.BoolInt1.fff(1)) == int
        assert type(cpp.BoolInt1.fff(2)) == int

        assert type(cpp.BoolInt1.fff(True)) == bool
        assert type(cpp.BoolInt1.fff(False)) == bool

        cppjit.cppdef("namespace BoolInt2 { int  fff(int i)  { return i; } }")
        cppjit.cppdef("namespace BoolInt2 { bool fff(bool i) { return i; } }")

        assert type(cpp.BoolInt2.fff(True)) == bool
        assert type(cpp.BoolInt2.fff(False)) == bool

        assert type(cpp.BoolInt2.fff(0)) == int
        assert type(cpp.BoolInt2.fff(1)) == int
        assert type(cpp.BoolInt2.fff(2)) == int

        cppjit.cppdef("namespace BoolInt3 { int  fff(int i)  { return i; } }")
        assert type(cpp.BoolInt3.fff(True)) == int
        assert type(cpp.BoolInt3.fff(False)) == int

        cppjit.cppdef("namespace BoolInt4 { bool fff(bool i) { return i; } }")

        assert type(cpp.BoolInt4.fff(0)) == bool
        assert type(cpp.BoolInt4.fff(1)) == bool
        with raises(ValueError):
            cpp.BoolInt4.fff(2)

    @mark.xfail(condition=IS_MAC, run=not IS_MAC_ARM, reason="Seg Faults")
    def test10_overload_and_exceptions(self):
        """Prioritize reporting C++ exceptions from callee"""

        if ispypy or IS_WINDOWS:
            skip("throwing exceptions from the JIT terminates the process")

        import cppjit

        cppjit.cppdef("""\
        namespace ExceptionTypeTest {

        class ConfigFileNotFoundError : public std::exception {
            std::string fMsg;
        public:
            ConfigFileNotFoundError(const std::string& msg) : fMsg(msg) {}
            const char* what() const throw() { return fMsg.c_str(); }
        };

        class MyClass1 {
        public:
            MyClass1(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            MyClass1(const MyClass1& other) {}
        };

        class MyClass2 {
        public:
            MyClass2(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            MyClass2(const char* configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            MyClass2(const MyClass2& other) {}
        };

        class MyClass3 {
        public:
            MyClass3(int) {}
            MyClass3(const MyClass3& other) {}
        }; }""")

        ns = cppjit.gbl.ExceptionTypeTest

        with raises(ns.ConfigFileNotFoundError):
            ns.MyClass1("some_file")

        with raises(TypeError):
            ns.MyClass2("some_file")

        with raises(TypeError):
            ns.MyClass3("some_file")

    def test11_deep_inheritance(self):
        """Prioritize expected most derived class"""

        import cppjit

        cppjit.cppdef("""\
        namespace DeepInheritance {
        class A {};
        class B: public A {};
        class C: public B {};

        class D: public A {};
        class E: public D {};

        std::string myfunc1(const B&) { return "B"; }
        std::string myfunc1(const C&) { return "C"; }
        std::string myfunc2(const E&) { return "E"; }
        std::string myfunc2(const D&) { return "D"; }
        }""")

        ns = cppjit.gbl.DeepInheritance

        assert ns.myfunc1(ns.B()) == "B"
        assert ns.myfunc1(ns.C()) == "C"

        assert ns.myfunc2(ns.E()) == "E"
        assert ns.myfunc2(ns.D()) == "D"

    def test12_static_call_from_derived_instance(self):
        """Test calling a static member function via a derived instance."""

        import cppjit

        cppjit.cppdef("""
            class Base {
            public:
                static int StaticMethod() {
                    return 42;
                }
            };

            class Derived : public Base {
            };
        """)

        d = cppjit.gbl.Derived()

        # Call static method through base class directly
        result_direct = cppjit.gbl.Base.StaticMethod()

        # Call static method through instance
        result_instance = d.StaticMethod()

        assert result_instance == result_direct

    def test13_disallow_functor_to_function_pointer(self):
        """Make sure we're no allowing to convert C++ functors to function
        pointers, extending the C++ language in an unnatural way that can lead
        to wrong overload resolutions."""
        import cppjit

        cppjit.cppdef("""
        class Test14Functor {
        public:
            double operator () (double* args, double*) {
                return 4.0 * args[0];
            }
        };

        int test14_foo(double (*fcn)(double*, double*)) {
            return 0;
        }

        template<class T>
        int test14_foo(T fcn) {
            return 1;
        }

        int test14_bar(double (*fcn)(double*, double*)) {
            return 0;
        }

        int test14_baz(double (*fcn)(double*, double*)) {
            return 0;
        }

        int test14_baz(std::function<double(double*, double*)> const &fcn) {
            return 2;
        }
        """)

        functor = cppjit.gbl.Test14Functor()
        assert cppjit.gbl.test14_foo(functor) == 1  # should resolve to foo(T fcn)
        # not allowed, because there is only an overload taking a function pointer
        raises(TypeError, cppjit.gbl.test14_bar, functor)
        # The "baz" function has a std::function overload, which should be selected
        assert (
            cppjit.gbl.test14_baz(functor) == 2
        )  # should resolve to baz(std::function)

    def test14_explicit_constructor_in_implicit_conversion(self):
        """Check that explicit constructors are not used in implicit conversion."""

        import cppjit

        cppjit.cppdef("""struct Test12Class {
          explicit Test12Class(int arg) {}
        };
        int test12_foo(Test12Class const&) { return 0; }
        int test12_foo(bool) { return 1; }
        int test12_bar(Test12Class const&) { return 0; }
        int test12_bar(bool = true) { return 1; }
        int call_test12_foo() { return test12_foo(1); }
        int call_test12_bar() { return test12_bar(1); }
        """)

        # Check that the cppjit overload resolution figures out the right
        # overload when calling the functions with an integer. In the past,
        # this used to go wrong for the "bar" function with the default bool
        # argument: cppjit went for the overload that takes the test class, even
        # though implicit construction of the test class is forbidden.
        assert cppjit.gbl.test12_foo(1) == cppjit.gbl.call_test12_foo()
        assert cppjit.gbl.test12_bar(1) == cppjit.gbl.call_test12_bar()

    def test15_disallow_mutable_pointer_references(self):
        """Verify that mutable pointer references (T*&) are not allowed as arguments."""

        import cppjit

        cppjit.cppdef("""
        struct MyClass {
           int val = 0;
        };

        void changePtr(MyClass *& ptr) {}
        """)

        ptr = cppjit.gbl.MyClass()

        raises(TypeError, cppjit.gbl.changePtr, ptr)

    def test16_voidp_does_not_outrank_conversion(self):
        """Verify that a const void* overload does not shadow a converting one."""

        import cppjit

        cppjit.cppdef("""
        namespace VoidPPriority {
        struct Handle {
            void* data;
            Handle() : data(nullptr) {}
            Handle(void* p) : data(p) {}
        };
        struct ConstHandle {
            const void* data;
            ConstHandle() : data(nullptr) {}
            ConstHandle(const void* p) : data(p) {}   // declared first on purpose
            ConstHandle(Handle h) : data(h.data) {}
        };
        Handle make_handle() { return Handle((void*)0xABCD1234); }
        bool kept_value(ConstHandle c) { return c.data == (const void*)0xABCD1234; }
        }""")

        ns = cppjit.gbl.VoidPPriority

        # taking ConstHandle(const void*) would pass the proxy's address instead
        h = ns.make_handle()
        assert ns.kept_value(h)
        assert ns.kept_value(ns.make_handle())

    def test17_func_overloads_types_reports_constness(self):
        """Verify func_overloads_types carries per-overload const-qualification."""

        import cppjit

        cppjit.cppdef("""
        namespace OverloadConstness {
        struct Probe {
            int v = 0;
            int get_const() const { return v; }
            void set_nonconst(int x) { v = x; }
            int mixed(int x) const { return x + v; }
            double mixed(double x) { return x; }
            static int static_fn(int x) { return x; }
        };
        }""")

        cls = cppjit.gbl.OverloadConstness.Probe

        def constness(name):
            return {
                sig: info["is_const"]
                for sig, info in cls.__dict__[name].func_overloads_types.items()
            }

        assert constness("get_const") == {
            "int OverloadConstness::Probe::get_const()": True
        }
        assert constness("set_nonconst") == {
            "void OverloadConstness::Probe::set_nonconst(int x)": False
        }
        # constness is per overload, not per method name
        assert constness("mixed") == {
            "int OverloadConstness::Probe::mixed(int x)": True,
            "double OverloadConstness::Probe::mixed(double x)": False,
        }
        assert constness("static_fn") == {
            "static int OverloadConstness::Probe::static_fn(int x)": False
        }
