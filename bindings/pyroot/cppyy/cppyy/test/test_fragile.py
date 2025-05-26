import py, os, sys, pytest
from pytest import mark, raises, skip
from support import setup_make, ispypy, IS_WINDOWS, IS_MAC_ARM


currpath = os.getcwd()
test_dct = currpath + "/libfragileDict"


class TestFRAGILE:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.fragile = cppyy.load_reflection_info(cls.test_dct)

    def test01_load_failure(self):
        """Test failure to load dictionary"""

        import cppyy
        raises(RuntimeError, cppyy.load_reflection_info, "does_not_exist")

        try:
            cppyy.load_reflection_info("does_not_exist")
        except RuntimeError as e:
            assert "does_not_exist" in str(e)

    def test02_missing_classes(self):
        """Test (non-)access to missing classes"""

        import cppyy

        raises(AttributeError, getattr, cppyy.gbl, "no_such_class")

        assert cppyy.gbl.fragile is cppyy.gbl.fragile
        assert cppyy.gbl.fragile == cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile

        raises(AttributeError, getattr, fragile, "no_such_class")

        assert fragile.C is fragile.C
        assert fragile.C == fragile.C
        assert fragile.C().check() == ord('C')

        assert fragile.B is fragile.B
        assert fragile.B == fragile.B
        assert fragile.B().check() == ord('B')
        assert not fragile.B().gime_no_such()

        assert fragile.C is fragile.C
        assert fragile.C == fragile.C
        assert fragile.C().check() == ord('C')
        raises(TypeError, fragile.C().use_no_such, None)

    def test03_arguments(self):
        """Test reporting when providing wrong arguments"""

        import cppyy

        assert cppyy.gbl.fragile == cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile

        assert fragile.D == fragile.D
        assert fragile.D().check() == ord('D')

        d = fragile.D()
        raises(TypeError, d.overload, None)
        raises(TypeError, d.overload, None, None, None)

        d.overload('a')
        d.overload(1)

    def test04_unsupported_arguments(self):
        """Test arguments that are yet unsupported"""

        import cppyy

        assert cppyy.gbl.fragile == cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile

        assert fragile.E == fragile.E
        assert fragile.E().check() == ord('E')

        e = fragile.E()
        raises(TypeError, e.overload, None)
        # allowing access to e.m_pp_no_such is debatable, but it allows a typed address
        # to be passed back into C++, which may be useful ...
        assert cppyy.addressof(e.m_pp_no_such[0]) == 0xdead

    def test05_wrong_arg_addressof(self):
        """Test addressof() error reporting"""

        import cppyy

        assert cppyy.gbl.fragile == cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile

        assert fragile.F == fragile.F
        assert fragile.F().check() == ord('F')

        f = fragile.F()
        o = object()

        cppyy.addressof(f)
        raises(TypeError, cppyy.addressof, o)
        raises(TypeError, cppyy.addressof, 1)

        # regression (m_int is 0 by default, but its address is not)
        assert cppyy.addressof(f, 'm_int')

        # see also test08_void_pointer_passing in test_advancedcpp.py

    def test06_wrong_this(self):
        """Test that using an incorrect self argument raises"""

        import cppyy

        assert cppyy.gbl.fragile == cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile

        a = fragile.A()
        assert fragile.A.check(a) == ord('A')

        b = fragile.B()
        assert fragile.B.check(b) == ord('B')
        raises(TypeError, fragile.A.check, b)
        raises(TypeError, fragile.B.check, a)

        assert not a.gime_null()

        assert isinstance(a.gime_null(), fragile.A)
        raises(ReferenceError, fragile.A.check, a.gime_null())

    def test07_unnamed_enum(self):
        """Test that an unnamed enum does not cause infinite recursion"""

        import cppyy

        assert cppyy.gbl.fragile is cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile
        assert cppyy.gbl.fragile is fragile

        g = fragile.G()

    def test08_unhandled_scoped_datamember(self):
        """Test that an unhandled scoped data member does not cause infinite recursion"""

        import cppyy

        assert cppyy.gbl.fragile is cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile
        assert cppyy.gbl.fragile is fragile

        h = fragile.H()

    def test09_operator_bool(self):
        """Access to global vars with an operator bool() returning False"""

        import cppyy

        i = cppyy.gbl.fragile.I()
        assert not i

        g = cppyy.gbl.fragile.gI
        assert not g

    @mark.xfail
    def test10_documentation(self):
        """Check contents of documentation"""

        import cppyy

        assert cppyy.gbl.fragile == cppyy.gbl.fragile
        fragile = cppyy.gbl.fragile

        d = fragile.D()
        try:
            d.check(None)         # raises TypeError
            assert 0
        except TypeError as e:
            assert "fragile::D::check()" in str(e)
            assert "TypeError: takes at most 0 arguments (1 given)" in str(e)
            assert "TypeError: takes at least 2 arguments (1 given)" in str(e)

        try:
            d.overload(None)      # raises TypeError
            assert 0
        except TypeError as e:
            # TODO: pypy-c does not indicate which argument failed to convert, CPython does
            # likewise there are still minor differences in descriptiveness of messages
            assert "fragile::D::overload()" in str(e)
            assert "TypeError: takes at most 0 arguments (1 given)" in str(e)
            assert "fragile::D::overload(fragile::no_such_class*)" in str(e)
            #assert "no converter available for 'fragile::no_such_class*'" in str(e)
            assert "void fragile::D::overload(char, int i = 0)" in str(e)
            #assert "char or small int type expected" in str(e)
            assert "void fragile::D::overload(int, fragile::no_such_class* p = 0)" in str(e)
            #assert "int/long conversion expects an integer object" in str(e)

        j = fragile.J()
        assert fragile.J.method1.__doc__ == j.method1.__doc__
        assert j.method1.__doc__ == "int fragile::J::method1(int, double)"

        f = fragile.fglobal
        assert f.__doc__ == "void fragile::fglobal(int, double, char)"

        try:
            o = fragile.O()       # raises TypeError
            assert 0
        except TypeError as e:
            assert "cannot instantiate abstract class 'fragile::O'" in str(e)

    @mark.xfail()
    def test11_dir(self):
        """Test __dir__ method"""

        import cppyy

        members = dir(cppyy.gbl.fragile)
        assert 'A' in members
        assert 'B' in members
        assert 'C' in members
        assert 'D' in members                # classes

        assert 'nested1' in members          # namespace

        # TODO: think this through ... probably want this, but interferes with
        # the (new) policy of lazy lookups
        #assert 'fglobal' in members         # function
        assert 'gI'in members                # variable

      # GetAllCppNames() behaves differently from python dir() but providing the full
      # set, which is then filtered in dir(); check both
        cppyy.cppdef("""\
        #ifdef _MSC_VER
        #define CPPYY_IMPORT extern __declspec(dllimport)
        #else
        #define CPPYY_IMPORT extern
        #endif

        namespace Cppyy {

        typedef size_t TCppScope_t;

        CPPYY_IMPORT TCppScope_t GetScope(const std::string& scope_name);
        CPPYY_IMPORT void GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames);

        }""")

        cppyy.cppdef("""\
        namespace GG {
        struct S {
          int _a;
          int _c;
          S(int a, int c): _a{a}, _c{c} { }
          S(): _a{0}, _c{0} { }
          bool operator<(int i) { return i < (_a+_c); }
        }; }""");


        assert 'S' in dir(cppyy.gbl.GG)

        handle = cppyy.gbl.Cppyy.GetScope("GG::S")
        assert handle

        cppnames = cppyy.gbl.std.set[str]()
        cppyy.gbl.Cppyy.GetAllCppNames(handle, cppnames)

        assert 'S' in cppnames
        assert '_a' in cppnames
        assert '_c' in cppnames

        assert 'operator<' in cppnames

        dirS = dir(cppyy.gbl.GG.S)

        assert 'S' not in dirS # is __init__
        assert '_a' in dirS
        assert '_c' in dirS

        assert 'operator<' not in dirS

    def test12_imports(self):
        """Test ability to import from namespace (or fail with ImportError)"""

        import cppyy

        # TODO: namespaces aren't loaded (and thus not added to sys.modules)
        # with just the from ... import statement; actual use is needed
        from cppyy.gbl import fragile

        def fail_import():
            from cppyy.gbl import does_not_exist
        raises(ImportError, fail_import)

        from cppyy.gbl.fragile import A, B, C, D
        assert cppyy.gbl.fragile.A is A
        assert cppyy.gbl.fragile.B is B
        assert cppyy.gbl.fragile.C is C
        assert cppyy.gbl.fragile.D is D

        # according to warnings, can't test "import *" ...

        from cppyy.gbl.fragile import nested1
        assert cppyy.gbl.fragile.nested1 is nested1
        assert nested1.__name__ == 'nested1'
        assert nested1.__module__ == 'cppyy.gbl.fragile'
        assert nested1.__cpp_name__ == 'fragile::nested1'

        from cppyy.gbl.fragile.nested1 import A, nested2
        assert cppyy.gbl.fragile.nested1.A is A
        assert A.__name__ == 'A'
        assert A.__module__ == 'cppyy.gbl.fragile.nested1'
        assert A.__cpp_name__ == 'fragile::nested1::A'
        assert cppyy.gbl.fragile.nested1.nested2 is nested2
        assert nested2.__name__ == 'nested2'
        assert nested2.__module__ == 'cppyy.gbl.fragile.nested1'
        assert nested2.__cpp_name__ == 'fragile::nested1::nested2'

        from cppyy.gbl.fragile.nested1.nested2 import A, nested3
        assert cppyy.gbl.fragile.nested1.nested2.A is A
        assert A.__name__ == 'A'
        assert A.__module__ == 'cppyy.gbl.fragile.nested1.nested2'
        assert A.__cpp_name__ == 'fragile::nested1::nested2::A'
        assert cppyy.gbl.fragile.nested1.nested2.nested3 is nested3
        assert nested3.__name__ == 'nested3'
        assert nested3.__module__ == 'cppyy.gbl.fragile.nested1.nested2'
        assert nested3.__cpp_name__ == 'fragile::nested1::nested2::nested3'

        from cppyy.gbl.fragile.nested1.nested2.nested3 import A
        assert cppyy.gbl.fragile.nested1.nested2.nested3.A is nested3.A
        assert A.__name__ == 'A'
        assert A.__module__ == 'cppyy.gbl.fragile.nested1.nested2.nested3'
        assert A.__cpp_name__ == 'fragile::nested1::nested2::nested3::A'

        # test writability of __module__
        nested3.__module__ = "peanut butter"
        assert nested3.__module__ == "peanut butter"

        # classes in namespace should inherit
        assert A.__module__ == 'peanut butter.nested3'
        assert 'peanut butter' in repr(A)
        assert 'class' in repr(A)
        assert 'peanut butter' in repr(nested3)
        assert 'namespace' in repr(nested3)

        # as should objects
        a = A()
        assert 'peanut butter' in repr(a)
        assert 'object' in repr(a)

    def test13_missing_casts(self):
        """Test proper handling when a hierarchy is not fully available"""

        import cppyy

        k = cppyy.gbl.fragile.K()

        assert k is k.GimeK(False)
        assert k is not k.GimeK(True)

        kd = k.GimeK(True)
        assert kd is k.GimeK(True)
        assert kd is not k.GimeK(False)

        l = k.GimeL()
        assert l is k.GimeL()

    def test14_double_enum_trouble(self):
        """Test a redefinition of enum in a derived class"""

        import cppyy

        M = cppyy.gbl.fragile.M
        N = cppyy.gbl.fragile.N

        assert M.kOnce == N.kOnce
        assert M.kTwice == N.kTwice

    def test15_const_in_name(self):
        """Make sure 'const' is not erased when part of a name"""

        import cppyy

        cppyy.cppdef("""
            struct Some0Class {}        myvar0;
            struct constSome1Class {}   myvar1;
            struct Some2Classconst {}   myvar2;
            struct Some_const_Class3 {} myvar3;
            struct SomeconstClass4 {}   myvar4;
        """)

        assert cppyy.gbl.myvar0
        assert cppyy.gbl.myvar1
        assert cppyy.gbl.myvar2
        assert cppyy.gbl.myvar3
        assert cppyy.gbl.myvar4

    @mark.xfail(run=False, reason="Crashes with \"alma10\"")
    def test16_opaque_handle(self):
        """Support use of opaque handles"""

        import cppyy

        assert cppyy.gbl.fragile.OpaqueType
        assert cppyy.gbl.fragile.OpaqueHandle_t

        handle = cppyy.gbl.fragile.OpaqueHandle_t(0x42)
        assert handle
        assert cppyy.addressof(handle) == 0x42

        raises(TypeError, cppyy.gbl.fragile.OpaqueType)
        assert not 'OpaqueType' in cppyy.gbl.fragile.__dict__

        handle = cppyy.gbl.fragile.OpaqueHandle_t()
        assert not handle

        addr = cppyy.gbl.fragile.create_handle(handle);
        assert addr
        assert not not handle

        assert cppyy.gbl.fragile.destroy_handle(handle, addr);

        # now define OpaqueType
        cppyy.cppdef("namespace fragile { class OpaqueType { public: int m_int; }; }")

        # get fresh (should not have been cached while incomplete)
        o = cppyy.gbl.fragile.OpaqueType()
        assert hasattr(o, 'm_int')

        assert 'OpaqueType' in cppyy.gbl.fragile.__dict__

    def test17_interactive(self):
        """Test the usage of 'from cppyy.interactive import *'"""

        import sys

        if 0x030b0000 <= sys.hexversion:
            skip('"from cppyy.interactive import *" is no longer supported')

        oldsp = sys.path[:]
        sys.path.append('.')
        try:
            import assert_interactive
        finally:
            sys.path = oldsp

    @mark.xfail()
    def test18_overload(self):
        """Test usage of __overload__"""

        import cppyy

        cppyy.cppdef("""struct Variable {
            Variable(double lb, double ub, double value, bool binary, bool integer, const std::string& name) {}
            Variable(int) {}
        };""")

        for sig in ['double, double, double, bool, bool, const std::string&',
                    'double,double,double,bool,bool,const std::string&',
                    'double lb, double ub, double value, bool binary, bool integer, const std::string& name']:
            assert cppyy.gbl.Variable.__init__.__overload__(sig)

    @mark.xfail(reason="Fails on \"alma9 modules_off runtime_cxxmodules=Off\"")
    def test19_gbl_contents(self):
        """Assure cppyy.gbl is mostly devoid of ROOT thingies"""

        import cppyy

        dd = dir(cppyy.gbl)

        assert not 'TCanvasImp' in dd
        assert not 'ESysConstants' in dd
        assert not 'kDoRed' in dd

    def test20_capture_output(self):
        """Capture cerr into a string"""

        if IS_MAC_ARM:
            skip("crashes in clang::Sema::FindInstantiatedDecl for rdbuf()")

        import cppyy

        cppyy.cppdef(r"""\
        namespace capture {
        void say_hello() {
           std::cerr << "Hello, World\n";
        }

        void rdbuf_wa(std::ostream& o, std::basic_stringbuf<char>* b) {
           o.rdbuf(b);
        } }""")

        capture = cppyy.gbl.std.ostringstream()
        oldbuf = cppyy.gbl.std.cerr.rdbuf()

        try:
            cppyy.gbl.capture.rdbuf_wa(cppyy.gbl.std.cerr, capture.rdbuf())
            cppyy.gbl.capture.say_hello()
        finally:
            cppyy.gbl.std.cerr.rdbuf(oldbuf)

        assert capture.str() == "Hello, World\n"

    def test21_failing_cppcode(self):
        """Check error behavior of failing C++ code"""

        import cppyy, string, re

        allspace = re.compile(r'\s+')
        def get_errmsg(exc, allspace=allspace):
            err = str(exc.value)
            return re.sub(allspace, '', err)

        with raises(ImportError) as include_exc:
            cppyy.include("doesnotexist.h")
        err = get_errmsg(include_exc)
        assert "Failedtoloadheaderfile\"doesnotexist.h\"" in err
        assert "fatalerror:" in err
        assert "\'doesnotexist.h\'filenotfound" in err

        with raises(ImportError) as c_include_exc:
            cppyy.c_include("doesnotexist.h")
        err = get_errmsg(c_include_exc)
        assert "Failedtoloadheaderfile\"doesnotexist.h\"" in err
        assert "fatalerror:" in err
        assert "\'doesnotexist.h\'filenotfound" in err

        with raises(SyntaxError) as cppdef_exc:
            cppyy.cppdef("1aap = 42;")
        err = get_errmsg(cppdef_exc)
        assert "FailedtoparsethegivenC++code" in err
        assert "error:" in err
        assert "invaliddigit" in err
        assert "1aap=42;" in err

    @mark.xfail()
    def test22_cppexec(self):
        """Interactive access to the Cling global scope"""

        import cppyy

        cppyy.cppexec("int interactive_b = 4")
        assert cppyy.gbl.interactive_b == 4

        with raises(SyntaxError):
            cppyy.cppexec("doesnotexist");

    # This test is very verbose since it sets gDebugo to true
    @mark.skip()
    def test23_set_debug(self):
        """Setting of global gDebug variable"""

        import cppyy

        cppyy.set_debug()
        assert cppyy.gbl.CppyyLegacy.gDebug == 10
        cppyy.set_debug(False)
        assert cppyy.gbl.CppyyLegacy.gDebug ==  0
        cppyy.set_debug(True)
        assert cppyy.gbl.CppyyLegacy.gDebug == 10
        cppyy.set_debug(False)
        assert cppyy.gbl.CppyyLegacy.gDebug ==  0

    @mark.xfail()
    def test24_asan(self):
        """Check availability of ASAN with gcc"""

        import cppyy
        import sys

        if not 'linux' in sys.platform:
            return

        cppyy.include('sanitizer/asan_interface.h')

    @mark.xfail()
    def test25_cppdef_error_reporting(self):
        """Check error reporting of cppyy.cppdef"""

        import cppyy, warnings

        assert cppyy.gbl.fragile.add42(1) == 43     # brings in symbol from library

        with raises(SyntaxError):
          # redefine symbol, leading to duplicate
            cppyy.cppdef("""\
            namespace fragile {
                int add42(int i) { return i + 42; }
            }""")

        with warnings.catch_warnings(record=True) as w:
          # missing return statement
            cppyy.cppdef("""\
            namespace fragile {
                double add42d(double d) { d + 42.; return d; }
            }""")

        assert len(w) == 1
        assert issubclass(w[-1].category, SyntaxWarning)
        assert "return" in str(w[-1].message)

      # mix of error and warning
        with raises(SyntaxError):
          # redefine symbol, leading to duplicate
            cppyy.cppdef("""\
            namespace fragile {
                float add42f(float d) { d + 42.f; }
                int add42(int i) { return i + 42; }
            }""")

    @mark.skip()
    def test26_macro(self):
        """Test access to C++ pre-processor macro's"""

        import cppyy

        cppyy.cppdef('#define HELLO "Hello, World!"')
        assert cppyy.macro("HELLO") == "Hello, World!"

        with raises(ValueError):
            cppyy.macro("SOME_INT")

        cppyy.cppdef('#define SOME_INT 42')
        assert cppyy.macro("SOME_INT") == 42

    def test27_pickle_enums(self):
        """Pickling of enum types"""

        import cppyy
        import pickle

        cppyy.cppdef("""
        enum MyPickleEnum { PickleFoo, PickleBar };
        namespace MyPickleNamespace {
          enum MyPickleEnum { PickleFoo, PickleBar };
        }""")

        e1 = cppyy.gbl.MyPickleEnum
        assert e1.__module__ == 'cppyy.gbl'
        assert pickle.dumps(e1.PickleFoo)

        e2 = cppyy.gbl.MyPickleNamespace.MyPickleEnum
        assert e2.__module__ == 'cppyy.gbl.MyPickleNamespace'
        assert pickle.dumps(e2.PickleBar)

    def test28_memoryview_of_empty(self):
        """memoryview of an empty array"""

        import cppyy, array

        cppyy.cppdef("void f(unsigned char const *buf) {}")
        try:
            cppyy.gbl.f(memoryview(array.array('B', [])))
        except TypeError:
            pass        # used to crash in PyObject_CheckBuffer on Linux

    def test29_vector_datamember(self):
        """Offset calculation of vector datamember"""

        import cppyy

        cppyy.cppdef("struct VectorDatamember { std::vector<unsigned> v; };")
        cppyy.gbl.VectorDatamember     # used to crash on Mac arm64

    @mark.skip()
    def test30_two_nested_ambiguity(self):
        """Nested class ambiguity in older Clangs"""

        import cppyy

        cppyy.cppdef("""\
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

        from cppyy.gbl import Test

        p = Test.Family1.Parent()
        p.children                          # used to crash

    @mark.xfail()
    def test31_template_with_class_enum(self):
        """Template instantiated with class enum"""

        import cppyy

        cppyy.cppdef("""\
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

        for ns, val in [(cppyy.gbl, 42),
                        (cppyy.gbl.ClassEnumNS, 37)]:
            assert ns.EnumTemplate[ns.ClassEnumA.A]().foo() == val


class TestSIGNALS:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.fragile = cppyy.load_reflection_info(cls.test_dct)

    @mark.xfail
    def test01_abortive_signals(self):
        """Conversion from abortive signals to Python exceptions"""

        if ispypy:
            skip('signals not yet implemented')

        if IS_MAC_ARM:
            skip("JIT exceptions from signals not supported on Mac ARM")

        if IS_WINDOWS:
            skip("abortive signals crash on most Windows platforms")

        import cppyy
        import cppyy.ll

        f = cppyy.gbl.fragile

        assert issubclass(cppyy.ll.BusError,               cppyy.ll.FatalError)
        assert issubclass(cppyy.ll.SegmentationViolation,  cppyy.ll.FatalError)
        assert issubclass(cppyy.ll.IllegalInstruction,     cppyy.ll.FatalError)
        assert issubclass(cppyy.ll.AbortSignal,            cppyy.ll.FatalError)

        import os
        os.putenv('CPPYY_CRASH_QUIET', '1')

        with raises((cppyy.ll.SegmentationViolation, cppyy.ll.IllegalInstruction)):
            with cppyy.ll.signals_as_exception():
                f.segfault()

        with raises(cppyy.ll.AbortSignal):
            with cppyy.ll.signals_as_exception():
                f.sigabort()

      # can only recover once from each error on Windows, which is functionally
      # enough, but precludes further testing here (change: now drop all, see above,
      # as on some MSVC builds, no signals are caught ??)
        if not IS_WINDOWS:
            cppyy.ll.set_signals_as_exception(True)
            with raises((cppyy.ll.SegmentationViolation, cppyy.ll.IllegalInstruction)):
                f.segfault()
            with raises(cppyy.ll.AbortSignal):
                f.sigabort()
            cppyy.ll.set_signals_as_exception(False)

            f.segfault.__sig2exc__ = True
            with raises((cppyy.ll.SegmentationViolation, cppyy.ll.IllegalInstruction)):
                f.segfault()

            f.sigabort.__sig2exc__ = True
            with raises(cppyy.ll.AbortSignal):
                f.sigabort()


class TestSTDNOTINGLOBAL:
    def setup_class(cls):
        import cppyy
        cls.has_byte = 201402 < cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;")

    @mark.xfail()
    def test01_stl_in_std(self):
        """STL classes should live in std:: only"""

        import cppyy

        names = ['array', 'function', 'list', 'set', 'vector']
        if self.has_byte:
            names.append('byte')

        for name in names:
            getattr(cppyy.gbl.std, name)
            with raises(AttributeError):
                getattr(cppyy.gbl, name)

      # inject a vector in the global namespace
        cppyy.cppdef("class vector{};")
        v = cppyy.gbl.vector()
        assert cppyy.gbl.vector is not cppyy.gbl.std.vector

    def test02_ctypes_in_both(self):
        """Standard int types live in both global and std::"""

        import cppyy

        for name in ['int8_t', 'uint8_t']:
            getattr(cppyy.gbl.std, name)
            getattr(cppyy.gbl, name)

        # TODO: get the types to match exactly as well
        assert cppyy.gbl.std.int8_t(-42) == cppyy.gbl.int8_t(-42)
        assert cppyy.gbl.std.uint8_t(42) == cppyy.gbl.uint8_t(42)

    @mark.xfail()
    def test03_clashing_using_in_global(self):
        """Redefines of std:: typedefs should be possible in global"""

        import cppyy

        cppyy.cppdef("""
            using uint   = unsigned int;
            using ushort = unsigned short;
            using uchar  = unsigned char;
            using byte   = unsigned char;
        """ )

        for name in ['int', 'uint', 'ushort', 'uchar', 'byte']:
            getattr(cppyy.gbl, name)

    @mark.xfail()
    def test04_no_legacy(self):
        """Test some functions that previously crashed"""

        import cppyy

        cppyy.cppdef("""
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

        assert cppyy.gbl.ELogLevel != cppyy.gbl.CppyyLegacy.ELogLevel

    @mark.xfail()
    def test05_span_compatibility(self):
        """Test compatibility of span under C++2a compilers that support it"""

        import cppyy

        cppyy.cppdef("""\
        #if __has_include(<span>)
        #include <span>
        std::span<int> my_test_span1;
        #endif
        """)


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
