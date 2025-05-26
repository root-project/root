import py, pytest, os
from pytest import raises, skip, mark
from support import setup_make, ispypy, IS_WINDOWS, IS_MAC_ARM


currpath = os.getcwd()
test_dct = currpath + "/liboverloadsDict"


class TestOVERLOADS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.overloads = cppyy.load_reflection_info(cls.test_dct)

    def test01_class_based_overloads(self):
        """Test functions overloaded on different C++ clases"""

        import cppyy
        a_overload = cppyy.gbl.a_overload
        b_overload = cppyy.gbl.b_overload
        c_overload = cppyy.gbl.c_overload
        d_overload = cppyy.gbl.d_overload

        ns_a_overload = cppyy.gbl.ns_a_overload
        ns_b_overload = cppyy.gbl.ns_b_overload

        assert c_overload().get_int(a_overload()) == 42
        assert c_overload().get_int(b_overload()) == 13
        assert d_overload().get_int(a_overload()) == 42
        assert d_overload().get_int(b_overload()) == 13

        assert c_overload().get_int(ns_a_overload.a_overload()) ==  88
        assert c_overload().get_int(ns_b_overload.a_overload()) == -33

        assert d_overload().get_int(ns_a_overload.a_overload()) ==  88
        assert d_overload().get_int(ns_b_overload.a_overload()) == -33

    def test02_class_based_overloads_explicit_resolution(self):
        """Test explicitly resolved function overloads"""

        import cppyy
        a_overload = cppyy.gbl.a_overload
        b_overload = cppyy.gbl.b_overload
        c_overload = cppyy.gbl.c_overload
        d_overload = cppyy.gbl.d_overload

        ns_a_overload = cppyy.gbl.ns_a_overload

        c = c_overload()
        raises(TypeError, c.__dispatch__, 'get_int', 12)
        raises(LookupError, c.__dispatch__, 'get_int', 'does_not_exist')
        assert c.__dispatch__('get_int', 'a_overload*')(a_overload())             == 42
        assert c_overload.get_int.__overload__('a_overload*')(c, a_overload())    == 42
        assert c.__dispatch__('get_int', 'b_overload*')(b_overload())             == 13
        assert c_overload.get_int.__overload__('b_overload*')(c, b_overload())    == 13

        assert c_overload().__dispatch__('get_int', 'a_overload*')(a_overload())  == 42
        # TODO: #assert c_overload.__dispatch__('get_int', 'b_overload*')(c, b_overload()) == 13

        d = d_overload()
        assert d.__dispatch__('get_int', 'a_overload*')(a_overload())             == 42
        assert d_overload.get_int.__overload__('a_overload*')(d, a_overload())    == 42
        assert d.__dispatch__('get_int', 'b_overload*')(b_overload())             == 13
        assert d_overload.get_int.__overload__('b_overload*')(d, b_overload())    == 13

        nb = ns_a_overload.b_overload()
        raises(TypeError, nb.f, c_overload())

    def test03_fragile_class_based_overloads(self):
        """Test functions overloaded on void* and non-existing classes"""

        import cppyy
        more_overloads = cppyy.gbl.more_overloads
        aa_ol = cppyy.gbl.aa_ol
        bb_ol = cppyy.gbl.bb_ol
        cc_ol = cppyy.gbl.cc_ol
        dd_ol = cppyy.gbl.dd_ol

        assert more_overloads().call(aa_ol()) == "aa_ol"
        bb = cppyy.gbl.get_bb_ol()
        assert more_overloads().call(bb     ) == "bb_ol"
        assert more_overloads().call(cc_ol()) == "cc_ol"
        dd = cppyy.bind_object(cppyy.nullptr, dd_ol)
        with raises(TypeError):
            more_overloads().call(dd)
        dd = cppyy.gbl.get_dd_ol()
        assert more_overloads().call(dd     ) == "dd_ol"

    def test04_fully_fragile_overloads(self):
        """Test that unknown* is preferred over unknown&"""

        import cppyy
        more_overloads2 = cppyy.gbl.more_overloads2

        bb = cppyy.bind_object(cppyy.nullptr, cppyy.gbl.bb_ol)
        assert more_overloads2().call(bb)    == "bb_olptr"

        dd = cppyy.bind_object(cppyy.nullptr, cppyy.gbl.dd_ol)
        assert more_overloads2().call(dd, 1) == "dd_olptr"

    def test05_array_overloads(self):
        """Test functions overloaded on different arrays"""

        import cppyy
        c_overload = cppyy.gbl.c_overload
        d_overload = cppyy.gbl.d_overload

        from array import array

        ai = array('i', [525252])
        assert c_overload().get_int(ai) == 525252
        assert d_overload().get_int(ai) == 525252

        ah = array('h', [25])
        assert c_overload().get_int(ah) == 25
        assert d_overload().get_int(ah) == 25

    def test06_double_int_overloads(self):
        """Test overloads on int/doubles"""

        import cppyy
        more_overloads = cppyy.gbl.more_overloads

        assert more_overloads().call(1)   == "int"
        assert more_overloads().call(1.)  == "double"
        assert more_overloads().call1(1)  == "int"
        assert more_overloads().call1(1.) == "double"

    def test07_mean_overloads(self):
        """Adapted test for array overloading"""

        import cppyy, array
        cmean = cppyy.gbl.calc_mean

        numbers = [8, 2, 4, 2, 4, 2, 4, 4, 1, 5, 6, 3, 7]
        mean, median = 4.0, 4.0

        for l in ['f', 'd', 'i', 'h', 'l']:
            a = array.array(l, numbers)
            assert round(cmean(len(a), a) - mean, 8) == 0

    def test08_const_non_const_overloads(self):
        """Check selectability of const/non-const overloads"""

        import cppyy

        m = cppyy.gbl.more_overloads3()

        assert m.slice.__overload__(':any:', True)(0)  == 'const'
        assert m.slice.__overload__(':any:', False)(0) == 'non-const'

        allmeths = cppyy.gbl.more_overloads3.slice.__overload__(':any:')
        cppyy.gbl.more_overloads3.slice        = allmeths.__overload__(':any:', False)
        cppyy.gbl.more_overloads3.slice_const =  allmeths.__overload__(':any:', True)
        del allmeths

        assert m.slice(0)          == 'non-const'
        assert m,slice(0, 0)       == 'non-const'
        assert m.slice_const(0)    ==     'const'
        assert m,slice_const(0, 0) ==     'const'

    def test09_bool_int_overloads(self):
        """Check bool/int overloaded calls"""

        import cppyy

        cpp = cppyy.gbl

        cppyy.cppdef("namespace BoolInt1 { int  fff(int i)  { return i; } }");
        cppyy.cppdef("namespace BoolInt1 { bool fff(bool i) { return i; } }")

        assert type(cpp.BoolInt1.fff(0)) == int
        assert type(cpp.BoolInt1.fff(1)) == int
        assert type(cpp.BoolInt1.fff(2)) == int

        assert type(cpp.BoolInt1.fff(True))  == bool
        assert type(cpp.BoolInt1.fff(False)) == bool

        cppyy.cppdef("namespace BoolInt2 { int  fff(int i)  { return i; } }");
        cppyy.cppdef("namespace BoolInt2 { bool fff(bool i) { return i; } }")

        assert type(cpp.BoolInt2.fff(True))  == bool
        assert type(cpp.BoolInt2.fff(False)) == bool

        assert type(cpp.BoolInt2.fff(0)) == int
        assert type(cpp.BoolInt2.fff(1)) == int
        assert type(cpp.BoolInt2.fff(2)) == int

        cppyy.cppdef("namespace BoolInt3 { int  fff(int i)  { return i; } }");

        assert type(cpp.BoolInt3.fff(True))  == int
        assert type(cpp.BoolInt3.fff(False)) == int

        cppyy.cppdef("namespace BoolInt4 { bool fff(bool i) { return i; } }")

        assert type(cpp.BoolInt4.fff(0)) == bool
        assert type(cpp.BoolInt4.fff(1)) == bool
        with raises(ValueError):
            cpp.BoolInt4.fff(2)

    @mark.xfail(run=False, condition=IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test10_overload_and_exceptions(self):
        """Prioritize reporting C++ exceptions from callee"""

        if ispypy or IS_WINDOWS:
            skip('throwing exceptions from the JIT terminates the process')

        import cppyy

        cppyy.cppdef("""\
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

        class MyClass1a {
        public:
            MyClass1a() {}
            void initialize(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
        };

        class MyClass2 {
        public:
            MyClass2(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            MyClass2(const char* configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
        };

        class MyClass2a {
        public:
            MyClass2a() {}
            void initialize(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            void initialize(const char* configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
        };

        class MyClass3 {
        public:
            MyClass3(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            MyClass3(const char* configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            MyClass3(const MyClass3& other) = delete;
            MyClass3(const MyClass3&& other) = delete;
        };

        class MyClass4 {
        public:
            MyClass4(int) {}
            MyClass4(const MyClass3& other) {}
        };

        class MyClass4a {
        public:
            MyClass4a() {}
            void initialize(const std::string& configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            void initialize(const char* configfilename) {
                throw ConfigFileNotFoundError{configfilename};
            }
            void initialize(int) {}
        }; }""")

        ns = cppyy.gbl.ExceptionTypeTest

        # single C++ exception during overload selection: assumes this is a logic
        # error and prioritizes the C++ exception
        with raises(ns.ConfigFileNotFoundError):
            ns.MyClass1("some_file")

        with raises(ns.ConfigFileNotFoundError):
            m = ns.MyClass1a()
            m.initialize("some_file")

        with raises(TypeError):   # special case b/c of copy constructor
            ns.MyClass2("some_file")

        # multiple C++ exceptions are considered argument conversion errors and
        # only result in the same exception type if they are all the same
        with raises(ns.ConfigFileNotFoundError):
            m = ns.MyClass2a()
            m.initialize("some_file")

            ns.MyClass2("some_file")   # special case b/c of copy constructor

        with raises(ns.ConfigFileNotFoundError):
            ns.MyClass3("some_file")   # copy constructor deleted

        # a mix of exceptions becomes a TypeError
        with raises(TypeError):
            ns.MyClass4("some_file")

        with raises(TypeError):
            m = ns.MyClass4a()
            m.initialize("some_file")

    def test11_deep_inheritance(self):
        """Prioritize expected most derived class"""

        import cppyy

        cppyy.cppdef("""\
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

        ns = cppyy.gbl.DeepInheritance

        assert ns.myfunc1(ns.B()) == "B"
        assert ns.myfunc1(ns.C()) == "C"

        assert ns.myfunc2(ns.E()) == "E"
        assert ns.myfunc2(ns.D()) == "D"


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
