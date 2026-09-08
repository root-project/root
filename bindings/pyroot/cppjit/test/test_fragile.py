import os
import sys

import py
from pytest import mark, raises, skip
from support import (
    IS_CLANG_REPL,
    IS_LINUX,
    IS_MAC,
    IS_MAC_ARM,
    IS_WINDOWS,
    ispypy,
    setup_make,
)

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("cpp/fragileDict"))


def setup_module(mod):
    setup_make("fragile")


def has_asan_interface():
    import cppjit

    return (
        cppjit.evaluate("""#if __has_include(<sanitizer/asan_interface.h>)
                        true
                        #else
                        false
                        #endif\n""")
        == 1
    )


class TestFRAGILE:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.fragile = cppjit.load_reflection_info(cls.test_dct)

    def test01_load_failure(self):
        """Test failure to load dictionary"""

        import cppjit

        raises(RuntimeError, cppjit.load_reflection_info, "does_not_exist")

        try:
            cppjit.load_reflection_info("does_not_exist")
        except RuntimeError as e:
            assert "does_not_exist" in str(e)

    def test02_missing_classes(self):
        """Test (non-)access to missing classes"""

        import cppjit

        raises(AttributeError, getattr, cppjit.gbl, "no_such_class")

        assert cppjit.gbl.fragile is cppjit.gbl.fragile
        assert cppjit.gbl.fragile == cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile

        # FIXME: Should be fixed with root dictionary
        #        look at https://github.com/wlav/cppyy/discussions/306
        # raises(AttributeError, getattr, fragile, "no_such_class")

        assert fragile.C is fragile.C
        assert fragile.C == fragile.C
        assert fragile.C().check() == ord("C")

        assert fragile.B is fragile.B
        assert fragile.B == fragile.B
        assert fragile.B().check() == ord("B")
        assert not fragile.B().gime_no_such()

        assert fragile.C is fragile.C
        assert fragile.C == fragile.C
        assert fragile.C().check() == ord("C")
        raises(TypeError, fragile.C().use_no_such, None)

    def test03_arguments(self):
        """Test reporting when providing wrong arguments"""

        import cppjit

        assert cppjit.gbl.fragile == cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile

        assert fragile.D == fragile.D
        assert fragile.D().check() == ord("D")

        d = fragile.D()
        raises(TypeError, d.overload, None)
        raises(TypeError, d.overload, None, None, None)

        d.overload("a")
        d.overload(1)

    def test04_unsupported_arguments(self):
        """Test arguments that are yet unsupported"""

        import cppjit

        assert cppjit.gbl.fragile == cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile

        assert fragile.E == fragile.E
        assert fragile.E().check() == ord("E")

        e = fragile.E()
        raises(TypeError, e.overload, None)
        # allowing access to e.m_pp_no_such is debatable, but it allows a typed address
        # to be passed back into C++, which may be useful ...
        assert cppjit.addressof(e.m_pp_no_such[0]) == 0xDEAD

    def test05_wrong_arg_addressof(self):
        """Test addressof() error reporting"""

        import cppjit

        assert cppjit.gbl.fragile == cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile

        assert fragile.F == fragile.F
        assert fragile.F().check() == ord("F")

        f = fragile.F()
        o = object()

        cppjit.addressof(f)
        raises(TypeError, cppjit.addressof, o)
        raises(TypeError, cppjit.addressof, 1)

        # regression (m_int is 0 by default, but its address is not)
        assert cppjit.addressof(f, "m_int")

        # see also test08_void_pointer_passing in test_advancedcpp.py

    def test06_wrong_this(self):
        """Test that using an incorrect self argument raises"""

        import cppjit

        assert cppjit.gbl.fragile == cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile

        a = fragile.A()
        assert fragile.A.check(a) == ord("A")

        b = fragile.B()
        assert fragile.B.check(b) == ord("B")
        raises(TypeError, fragile.A.check, b)
        raises(TypeError, fragile.B.check, a)

        assert not a.gime_null()

        assert isinstance(a.gime_null(), fragile.A)
        raises(ReferenceError, fragile.A.check, a.gime_null())

    def test07_unnamed_enum(self):
        """Test that an unnamed enum does not cause infinite recursion"""

        import cppjit

        assert cppjit.gbl.fragile is cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile
        assert cppjit.gbl.fragile is fragile

        g = fragile.G()

    def test08_unhandled_scoped_datamember(self):
        """Test that an unhandled scoped data member does not cause infinite recursion"""

        import cppjit

        assert cppjit.gbl.fragile is cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile
        assert cppjit.gbl.fragile is fragile

        h = fragile.H()

    def test09_operator_bool(self):
        """Access to global vars with an operator bool() returning False"""

        import cppjit

        i = cppjit.gbl.fragile.I()
        assert not i

        g = cppjit.gbl.fragile.gI
        assert not g

    def test10_documentation(self):
        """Check contents of documentation"""

        import cppjit

        assert cppjit.gbl.fragile == cppjit.gbl.fragile
        fragile = cppjit.gbl.fragile

        d = fragile.D()
        try:
            d.check(None)  # raises TypeError
            assert 0
        except TypeError as e:
            assert "fragile::D::check()" in str(e)
            assert "TypeError: takes at most 0 arguments (1 given)" in str(e)
            assert "TypeError: takes at least 2 arguments (1 given)" in str(e)

        try:
            d.overload(None)  # raises TypeError
            assert 0
        except TypeError as e:
            # TODO: pypy-c does not indicate which argument failed to convert, CPython does
            # likewise there are still minor differences in descriptiveness of messages
            err_msg = str(e).replace(" ", "")
            assert "fragile::D::overload()" in err_msg
            assert "TypeError: takes at most 0 arguments (1 given)" in str(e)
            assert "fragile::D::overload(fragile::no_such_class*)" in err_msg
            # assert "no converter available for 'fragile::no_such_class*'" in str(e)
            assert (
                "void fragile::D::overload(char, int i = 0)".replace(" ", "") in err_msg
            )
            # assert "char or small int type expected" in str(e)
            assert (
                "void fragile::D::overload(int, fragile::no_such_class * p = 0)".replace(
                    " ", ""
                )
                in err_msg
            )
            # assert "int/long conversion expects an integer object" in str(e)

        j = fragile.J()
        assert fragile.J.method1.__doc__ == j.method1.__doc__
        assert j.method1.__doc__ == "int fragile::J::method1(int, double)"

        f = fragile.fglobal
        assert f.__doc__ == "void fragile::fglobal(int, double, char)"

        try:
            o = fragile.O()  # raises TypeError
            assert 0
        except TypeError as e:
            assert "cannot instantiate abstract class 'fragile::O'" in str(e)

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test11_dir(self):
        """Test __dir__ method"""

        import cppjit

        members = dir(cppjit.gbl.fragile)
        assert "A" in members
        assert "B" in members
        assert "C" in members
        assert "D" in members  # classes

        assert "nested1" in members  # namespace

        # TODO: think this through ... probably want this, but interferes with
        # the (new) policy of lazy lookups
        # assert 'fglobal' in members         # function
        assert "gI" in members  # variable

        # GetAllCppNames() behaves differently from python dir() but providing the full
        # set, which is then filtered in dir(); check both
        cppjit.cppdef("""\
        #ifdef _MSC_VER
        #define CPPJIT_IMPORT extern __declspec(dllimport)
        #else
        #define CPPJIT_IMPORT extern
        #endif
        
        namespace Cpp {
        struct DeclRef;
        }  // namespace Cpp

        namespace cppjit::interop {
        typedef Cpp::DeclRef TCppScope_t;

        CPPJIT_IMPORT TCppScope_t GetScope(const std::string& scope_name, TCppScope_t parent_scope = TCppScope_t());
        CPPJIT_IMPORT void GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames);

        }""")

        cppjit.cppdef("""\
        namespace GG {
        struct S {
          int _a;
          int _c;
          S(int a, int c): _a{a}, _c{c} { }
          S(): _a{0}, _c{0} { }
          bool operator<(int i) { return i < (_a+_c); }
        }; }""")
        assert "S" in dir(cppjit.gbl.GG)

        handle = cppjit.gbl.cppjit.interop.GetScope("GG::S")
        assert handle

        cppnames = cppjit.gbl.std.set[str]()
        cppjit.gbl.cppjit.interop.GetAllCppNames(handle, cppnames)

        assert "S" in cppnames
        assert "_a" in cppnames
        assert "_c" in cppnames

        assert "operator<" in cppnames

        dirS = dir(cppjit.gbl.GG.S)

        assert "S" not in dirS  # is __init__
        assert "_a" in dirS
        assert "_c" in dirS

        assert "operator<" not in dirS

    def test12_imports(self):
        """Test ability to import from namespace (or fail with ImportError)"""

        import cppjit

        # TODO: namespaces aren't loaded (and thus not added to sys.modules)
        # with just the from ... import statement; actual use is needed
        from cppjit.gbl import fragile  # noqa: F401

        def fail_import():
            from cppjit.gbl import does_not_exist  # noqa: F401

        raises(ImportError, fail_import)

        from cppjit.gbl.fragile import A, B, C, D

        assert cppjit.gbl.fragile.A is A
        assert cppjit.gbl.fragile.B is B
        assert cppjit.gbl.fragile.C is C
        assert cppjit.gbl.fragile.D is D

        # according to warnings, can't test "import *" ...

        from cppjit.gbl.fragile import nested1

        assert cppjit.gbl.fragile.nested1 is nested1
        assert nested1.__name__ == "nested1"
        assert nested1.__module__ == "cppjit.gbl.fragile"
        assert nested1.__cpp_name__ == "fragile::nested1"

        from cppjit.gbl.fragile.nested1 import A, nested2

        assert cppjit.gbl.fragile.nested1.A is A
        assert A.__name__ == "A"
        assert A.__module__ == "cppjit.gbl.fragile.nested1"
        assert A.__cpp_name__ == "fragile::nested1::A"
        assert cppjit.gbl.fragile.nested1.nested2 is nested2
        assert nested2.__name__ == "nested2"
        assert nested2.__module__ == "cppjit.gbl.fragile.nested1"
        assert nested2.__cpp_name__ == "fragile::nested1::nested2"

        from cppjit.gbl.fragile.nested1.nested2 import A, nested3

        assert cppjit.gbl.fragile.nested1.nested2.A is A
        assert A.__name__ == "A"
        assert A.__module__ == "cppjit.gbl.fragile.nested1.nested2"
        assert A.__cpp_name__ == "fragile::nested1::nested2::A"
        assert cppjit.gbl.fragile.nested1.nested2.nested3 is nested3
        assert nested3.__name__ == "nested3"
        assert nested3.__module__ == "cppjit.gbl.fragile.nested1.nested2"
        assert nested3.__cpp_name__ == "fragile::nested1::nested2::nested3"

        from cppjit.gbl.fragile.nested1.nested2.nested3 import A

        assert cppjit.gbl.fragile.nested1.nested2.nested3.A is nested3.A
        assert A.__name__ == "A"
        assert A.__module__ == "cppjit.gbl.fragile.nested1.nested2.nested3"
        assert A.__cpp_name__ == "fragile::nested1::nested2::nested3::A"

        # test writability of __module__
        nested3.__module__ = "peanut butter"
        assert nested3.__module__ == "peanut butter"

        # classes in namespace should inherit
        assert A.__module__ == "peanut butter.nested3"
        assert "peanut butter" in repr(A)
        assert "class" in repr(A)
        assert "peanut butter" in repr(nested3)
        assert "namespace" in repr(nested3)

        # as should objects
        a = A()
        assert "peanut butter" in repr(a)
        assert "object" in repr(a)

    def test13_missing_casts(self):
        """Test proper handling when a hierarchy is not fully available"""

        import cppjit

        k = cppjit.gbl.fragile.K()

        assert k is k.GimeK(False)
        assert k is not k.GimeK(True)

        kd = k.GimeK(True)
        assert kd is k.GimeK(True)
        assert kd is not k.GimeK(False)

        l = k.GimeL()
        assert l is k.GimeL()

    def test14_double_enum_trouble(self):
        """Test a redefinition of enum in a derived class"""

        import cppjit

        M = cppjit.gbl.fragile.M
        N = cppjit.gbl.fragile.N

        assert M.kOnce == N.kOnce
        assert M.kTwice == N.kTwice

    def test15_const_in_name(self):
        """Make sure 'const' is not erased when part of a name"""

        import cppjit

        cppjit.cppdef("""
            struct Some0Class {}        myvar0;
            struct constSome1Class {}   myvar1;
            struct Some2Classconst {}   myvar2;
            struct Some_const_Class3 {} myvar3;
            struct SomeconstClass4 {}   myvar4;
        """)

        assert cppjit.gbl.myvar0
        assert cppjit.gbl.myvar1
        assert cppjit.gbl.myvar2
        assert cppjit.gbl.myvar3
        assert cppjit.gbl.myvar4

    def test16_opaque_handle(self):
        """Support use of opaque handles"""

        import cppjit

        assert cppjit.gbl.fragile.OpaqueType
        assert cppjit.gbl.fragile.OpaqueHandle_t

        handle = cppjit.gbl.fragile.OpaqueHandle_t(0x42)
        assert handle
        assert cppjit.addressof(handle) == 0x42

        raises(TypeError, cppjit.gbl.fragile.OpaqueType)
        assert "OpaqueType" not in cppjit.gbl.fragile.__dict__

        handle = cppjit.gbl.fragile.OpaqueHandle_t()
        assert not handle

        addr = cppjit.gbl.fragile.create_handle(handle)
        assert addr
        assert not not handle

        assert cppjit.gbl.fragile.destroy_handle(handle, addr)
        # now define OpaqueType
        cppjit.cppdef("namespace fragile { class OpaqueType { public: int m_int; }; }")

        # get fresh (should not have been cached while incomplete)
        o = cppjit.gbl.fragile.OpaqueType()
        assert hasattr(o, "m_int")

        assert "OpaqueType" in cppjit.gbl.fragile.__dict__

    def test17_interactive(self):
        """Test the usage of 'from cppjit.interactive import *'"""

        if 0x030B0000 <= sys.hexversion:
            skip('"from cppjit.interactive import *" is no longer supported')

        oldsp = sys.path[:]
        sys.path.append(".")
        try:
            import assert_interactive  # noqa: F401
        finally:
            sys.path = oldsp

    def test18_overload(self):
        """Test usage of __overload__"""

        import cppjit

        cppjit.cppdef("""struct Variable {
            Variable(double lb, double ub, double value, bool binary, bool integer, const std::string& name) {}
            Variable(int) {}
        };""")

        for sig in [
            "double, double, double, bool, bool, const std::string&",
            "double,double,double,bool,bool,const std::string&",
            "double lb, double ub, double value, bool binary, bool integer, const std::string& name",
        ]:
            assert cppjit.gbl.Variable.__init__.__overload__(sig)

    def test19_gbl_contents(self):
        """Assure cppjit.gbl is mostly devoid of ROOT thingies"""

        import cppjit

        dd = dir(cppjit.gbl)

        assert "TCanvasImp" not in dd
        assert "ESysConstants" not in dd
        assert "kDoRed" not in dd

    def test20_capture_output(self):
        """Capture cerr into a string"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace capture {
        void say_hello() {
           std::cerr << "Hello, World\n";
        }

        void rdbuf_wa(std::ostream& o, std::basic_stringbuf<char>* b) {
           o.rdbuf(b);
        } }""")

        capture = cppjit.gbl.std.ostringstream()
        oldbuf = cppjit.gbl.std.cerr.rdbuf()

        try:
            cppjit.gbl.capture.rdbuf_wa(cppjit.gbl.std.cerr, capture.rdbuf())
            cppjit.gbl.capture.say_hello()
        finally:
            cppjit.gbl.std.cerr.rdbuf(oldbuf)

        assert capture.str() == "Hello, World\n"

    @mark.xfail(condition=IS_CLANG_REPL, reason="Fails with ClangRepl")
    def test21_failing_cppcode(self):
        """Check error behavior of failing C++ code"""

        import re

        import cppjit

        allspace = re.compile(r"\s+")

        def get_errmsg(exc, allspace=allspace):
            err = str(exc.value)
            return re.sub(allspace, "", err)

        with raises(ImportError) as include_exc:
            cppjit.include("doesnotexist.h")
        err = get_errmsg(include_exc)
        assert 'Failedtoloadheaderfile"doesnotexist.h"' in err
        assert "fatalerror:" in err
        assert "'doesnotexist.h'filenotfound" in err

        with raises(ImportError) as c_include_exc:
            cppjit.c_include("doesnotexist.h")
        err = get_errmsg(c_include_exc)
        assert 'Failedtoloadheaderfile"doesnotexist.h"' in err
        assert "fatalerror:" in err
        assert "'doesnotexist.h'filenotfound" in err

        with raises(SyntaxError) as cppdef_exc:
            cppjit.cppdef("1aap = 42;")
        err = get_errmsg(cppdef_exc)
        assert "FailedtoparsethegivenC++code" in err
        assert "error:" in err

        # On older cling versions this error used to contain "expectedunqualified-id"
        # With Cling based on LLVM 18 and Clang-REPL based on LLVM 20, this is consistently
        # an invalid digit error
        assert "invaliddigit" in err
        assert "1aap=42;" in err

    def test22_cppexec(self):
        """Interactive access to the Cling global scope"""

        import cppjit

        cppjit.cppexec("int interactive_b = 4")
        assert cppjit.gbl.interactive_b == 4

        with raises(SyntaxError):
            cppjit.cppexec("doesnotexist")

    def test23_set_debug(self):
        """Setting of debugging facilities"""

        import cppjit

        cppjit.set_debug()
        assert cppjit.gbl.Cpp.IsDebugOutputEnabled() == True
        cppjit.set_debug(False)
        assert cppjit.gbl.Cpp.IsDebugOutputEnabled() == False
        cppjit.set_debug(True)
        assert cppjit.gbl.Cpp.IsDebugOutputEnabled() == True
        cppjit.set_debug(False)
        assert cppjit.gbl.Cpp.IsDebugOutputEnabled() == False

    @mark.xfail(
        condition=IS_LINUX and not has_asan_interface(),
        reason="sanitizer/asan_interface.h not available",
    )
    def test24_asan(self):
        """Check availability of ASAN with gcc"""

        import cppjit

        if "linux" not in sys.platform:
            return

        cppjit.include("sanitizer/asan_interface.h")

    @mark.xfail(reason="cppdef of invalid code does not raise SyntaxError")
    def test25_cppdef_error_reporting(self):
        """Check error reporting of cppjit.cppdef"""

        import warnings

        import cppjit

        assert cppjit.gbl.fragile.add42(1) == 43  # brings in symbol from library

        with raises(SyntaxError):
            # redefine symbol, leading to duplicate
            cppjit.cppdef("""\
            namespace fragile {
                int add42(int i) { return i + 42; }
            }""")

        # isolate the warning configuration
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn warnings into errors

            # missing return statement
            with raises(SyntaxWarning) as exc:
                cppjit.cppdef("""\
                namespace fragile {
                    void add42d(double d) {
                    #warning return plastic for recycling!
                    }
                }""")

            assert "return" in str(exc.value)

        # mix of error and warning
        with raises(SyntaxError):
            # redefine symbol, leading to duplicate
            cppjit.cppdef("""\
            namespace fragile {
                float add42f(float d) { d + 42.f; }
                int add42(int i) { return i + 42; }
            }""")

    @mark.xfail(condition=IS_CLANG_REPL, reason="Fails on ClangRepl")
    def test26_macro(self):
        """Test access to C++ pre-processor macro's"""

        import cppjit

        cppjit.cppdef('#define HELLO "Hello, World!"')
        assert cppjit.macro("HELLO") == "Hello, World!"

        with raises(ValueError):
            cppjit.macro("SOME_INT")

        cppjit.cppdef("#define SOME_INT 42")
        assert cppjit.macro("SOME_INT") == 42

    def test27_pickle_enums(self):
        """Pickling of enum types"""

        import pickle

        import cppjit

        cppjit.cppdef("""
        enum MyPickleEnum { PickleFoo, PickleBar };
        namespace MyPickleNamespace {
          enum MyPickleEnum { PickleFoo, PickleBar };
        }""")

        e1 = cppjit.gbl.MyPickleEnum
        assert e1.__module__ == "cppjit.gbl"
        assert pickle.dumps(e1.PickleFoo)

        e2 = cppjit.gbl.MyPickleNamespace.MyPickleEnum
        assert e2.__module__ == "cppjit.gbl.MyPickleNamespace"
        assert pickle.dumps(e2.PickleBar)

    def test28_memoryview_of_empty(self):
        """memoryview of an empty array"""

        import array

        import cppjit

        cppjit.cppdef("void f(unsigned char const *buf) {}")
        try:
            cppjit.gbl.f(memoryview(array.array("B", [])))
        except TypeError:
            pass  # used to crash in PyObject_CheckBuffer on Linux

    def test29_vector_datamember(self):
        """Offset calculation of vector datamember"""

        import cppjit

        cppjit.cppdef("struct VectorDatamember { std::vector<unsigned> v; };")
        cppjit.gbl.VectorDatamember  # used to crash on Mac arm64

    def test30_two_nested_ambiguity(self):
        """Nested class ambiguity in older Clangs"""

        import cppjit

        cppjit.cppdef("""\
        #include <vector>

        namespace Test {
        struct Common { std::string name; };
        struct Family1 {
            struct Parent : Common {
                struct Child : Common { };
                std::vector<Child> children;
            };
        };

        struct Family2 {
            struct Parent : Common {
                struct Child : Common { };
                std::vector<Child> children;
            };
        }; }""")

        from cppjit.gbl import Test

        p = Test.Family1.Parent()
        p.children  # used to crash

    def test31_template_with_class_enum(self):
        """Template instantiated with class enum"""

        import cppjit

        cppjit.cppdef("""\
        enum class ClassEnumA { A, };

        template<ClassEnumA T>
        struct EnumTemplate {
          int foo();
        };

        template<> int EnumTemplate<ClassEnumA::A>::foo() { return 42; }
        template class EnumTemplate<ClassEnumA::A>;

        namespace ClassEnumNS {
          enum class ClassEnumA { A, };

          template<ClassEnumA T>
          struct EnumTemplate {
            int foo();
          };

          template<> int EnumTemplate<ClassEnumA::A>::foo() { return 37; }
          template class EnumTemplate<ClassEnumA::A>;
        }""")

        for ns, val in [(cppjit.gbl, 42), (cppjit.gbl.ClassEnumNS, 37)]:
            assert ns.EnumTemplate[ns.ClassEnumA.A]().foo() == val

    def test32_overloaded_method_error_with_null_object(self):
        """Check exception type and message when method invoked on instance without C++ object"""

        import cppjit
        from cppjit import gbl

        cppjit.cppdef(r"""\
        using fragile::D;
        D *something = new D;
        D *nothing = nullptr;
        """)

        assert gbl.something.check() == gbl.something.check(0, 1)
        with raises(ReferenceError, match=r"^no C\+\+ object available$"):
            gbl.nothing.check()  # raises error
        with raises(ReferenceError, match=r"^no C\+\+ object available$"):
            gbl.nothing.check(0, 1)  # raises error


class TestSIGNALS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppjit

        cls.fragile = cppjit.load_reflection_info(cls.test_dct)

    @mark.xfail(run=False, reason="Crashes")
    def test01_abortive_signals(self):
        """Conversion from abortive signals to Python exceptions"""

        if ispypy:
            skip("signals not yet implemented")

        if IS_MAC_ARM:
            skip("JIT exceptions from signals not supported on Mac ARM")

        import cppjit
        import cppjit.ll

        f = cppjit.gbl.fragile

        assert issubclass(cppjit.ll.BusError, cppjit.ll.FatalError)
        assert issubclass(cppjit.ll.SegmentationViolation, cppjit.ll.FatalError)
        assert issubclass(cppjit.ll.IllegalInstruction, cppjit.ll.FatalError)
        assert issubclass(cppjit.ll.AbortSignal, cppjit.ll.FatalError)

        os.putenv("CPPJIT_CRASH_QUIET", "1")

        with raises((cppjit.ll.SegmentationViolation, cppjit.ll.IllegalInstruction)):
            with cppjit.ll.signals_as_exception():
                f.segfault()

        with raises(cppjit.ll.AbortSignal):
            with cppjit.ll.signals_as_exception():
                f.sigabort()

        # can only recover once from each error on Windows, which is functionally
        # enough, but precludes further testing here
        if not IS_WINDOWS:
            cppjit.ll.set_signals_as_exception(True)
            with raises(
                (cppjit.ll.SegmentationViolation, cppjit.ll.IllegalInstruction)
            ):
                f.segfault()
            with raises(cppjit.ll.AbortSignal):
                f.sigabort()
            cppjit.ll.set_signals_as_exception(False)

            f.segfault.__sig2exc__ = True
            with raises(
                (cppjit.ll.SegmentationViolation, cppjit.ll.IllegalInstruction)
            ):
                f.segfault()

            f.sigabort.__sig2exc__ = True
            with raises(cppjit.ll.AbortSignal):
                f.sigabort()


class TestSTDNOTINGLOBAL:
    def setup_class(cls):
        pass

    def test01_stl_in_std(self):
        """STL classes should live in std:: only"""

        import cppjit

        names = ["array", "function", "list", "set", "vector", "byte"]

        for name in names:
            getattr(cppjit.gbl.std, name)
            with raises(AttributeError):
                getattr(cppjit.gbl, name)

        # inject a vector in the global namespace
        cppjit.cppdef("class vector{};")
        v = cppjit.gbl.vector()
        assert cppjit.gbl.vector is not cppjit.gbl.std.vector

    def test02_ctypes_in_both(self):
        """Standard int types live in both global and std::"""

        import cppjit

        for name in ["int8_t", "uint8_t"]:
            getattr(cppjit.gbl.std, name)
            getattr(cppjit.gbl, name)

        # TODO: get the types to match exactly as well
        assert cppjit.gbl.std.int8_t(-42) == cppjit.gbl.int8_t(-42)
        assert cppjit.gbl.std.uint8_t(42) == cppjit.gbl.uint8_t(42)

    def test03_clashing_using_in_global(self):
        """Redefines of std:: typedefs should be possible in global"""

        import cppjit

        cppjit.cppdef("""
            using uint   = unsigned int;
            using ushort = unsigned short;
            using uchar  = unsigned char;
            using byte   = unsigned char;
        """)

        for name in ["int", "uint", "ushort", "uchar", "byte"]:
            getattr(cppjit.gbl, name)

    def test04_no_legacy(self):
        """Test some functions that previously crashed"""

        import cppjit

        cppjit.cppdef("""
        namespace CppyyLegacy {
          enum ELogLevel {
            kLogEmerg          = 0,
            kLogAlert          = 1,
            kLogCrit           = 2,
            kLogErr            = 3,
            kLogWarning        = 4,
            kLogNotice         = 5,
            kLogInfo           = 6,
            kLogDebug          = 7
          };
        }""")

        cppjit.cppdef("""
        enum ELogLevel {
          kLogEmerg          = 0,
          kLogAlert          = 1,
          kLogCrit           = 2,
          kLogErr            = 3,
          kLogWarning        = 4,
          kLogNotice         = 5,
          kLogInfo           = 6,
          kLogDebug          = 7
        };""")

        assert cppjit.gbl.ELogLevel != cppjit.gbl.CppyyLegacy.ELogLevel

    def test05_span_compatibility(self):
        """Test compatibility of span under C++2a compilers that support it"""

        import cppjit

        cppjit.cppdef("""\
        #if __cplusplus >= 202002 && __has_include(<span>)
        #include <span>
        std::span<int> my_test_span1;
        #endif
        """)
