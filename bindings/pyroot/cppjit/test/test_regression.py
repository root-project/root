import os
import sys

from pytest import mark, raises, skip
from support import (
    CAN_JIT_STD_FILESYSTEM,
    IS_CLANG_REPL,
    IS_CLING,
    IS_MAC,
    IS_MAC_ARM,
    IS_WINDOWS,
    ispypy,
)


class TestREGRESSION:
    helpout = []

    def setup_class(cls):

        if sys.hexversion < 0x30D0000:

            def stringpager(text, cls=cls):
                cls.helpout.append(text)
        else:

            def stringpager(text, title="", cls=cls):
                cls.helpout.append(text)

        import pydoc

        pydoc.pager = stringpager

    @mark.xfail(reason="pydoc rendering of KDcrawIface fails")
    def test01_kdcraw(self):
        """Doc strings for KDcrawIface (used to crash)."""

        import pydoc

        import cppjit

        # TODO: run a find for these paths
        qtpath = "/usr/include/qt5"
        kdcraw_h = "/usr/include/KF5/KDCRAW/kdcraw/kdcraw.h"
        if not os.path.isdir(qtpath) or not os.path.exists(kdcraw_h):
            skip("no KDE/Qt found, skipping test01_kdcraw")

        # need to resolve qt_version_tag for the incremental compiler; since
        # it's not otherwise used, just make something up
        cppjit.cppdef("int qt_version_tag = 42;")
        cppjit.add_include_path(qtpath)
        cppjit.include(kdcraw_h)

        # bring in some symbols to resolve the class
        cppjit.load_library("libQt5Core.so")
        cppjit.load_library("libKF5KDcraw.so")

        from cppjit.gbl import KDcrawIface

        self.__class__.helpout = []
        pydoc.doc(KDcrawIface.KDcraw)
        helptext = "".join(self.__class__.helpout)
        assert "KDcraw" in helptext
        assert "CPPInstance" in helptext

    def test02_dir(self):
        """For the same reasons as test01_kdcraw, this used to crash."""

        if ispypy:
            skip("hangs (??) in pypy")

        import pydoc

        import cppjit

        cppjit.cppdef("""
            namespace docs {
            struct MyDocs {
                void fn() {}
            } s;
            MyDocs *ptrDocs = &s;
            }
        """)

        assert "__abstractmethods__" not in dir(cppjit.gbl.docs.ptrDocs)
        assert "__class__" in dir(cppjit.gbl.docs.ptrDocs)

        self.__class__.helpout = []
        pydoc.doc(cppjit.gbl.docs.ptrDocs)
        helptext = "".join(self.__class__.helpout)
        assert "MyDocs" in helptext
        assert "CPPInstance" in helptext
        assert "fn" in helptext

        cppjit.cppdef("namespace cppjit_regression_test { void iii() {}; }")

        assert "iii" not in cppjit.gbl.cppjit_regression_test.__dict__
        assert "__abstractmethods__" not in dir(cppjit.gbl.cppjit_regression_test)
        assert "__class__" in dir(cppjit.gbl.cppjit_regression_test)
        assert "iii" in dir(cppjit.gbl.cppjit_regression_test)

        assert "iii" not in cppjit.gbl.cppjit_regression_test.__dict__
        assert cppjit.gbl.cppjit_regression_test.iii
        assert "iii" in cppjit.gbl.cppjit_regression_test.__dict__

        self.__class__.helpout = []
        pydoc.doc(cppjit.gbl.cppjit_regression_test)
        helptext = "".join(self.__class__.helpout)
        # TODO: it's deeply silly that namespaces inherit from CPPInstance (in cpyrt)
        assert "CPPInstance" in helptext or "CPPNamespace" in helptext

    def test03_pyfunc_doc(self):
        """Help on a generated pyfunc used to crash."""

        import pydoc
        import sys
        import sysconfig as sc

        import cppjit

        cppjit.add_include_path(sc.get_config_var("INCLUDEPY"))
        if sys.hexversion < 0x3000000:
            cppjit.cppdef("#undef _POSIX_C_SOURCE")
            cppjit.cppdef("#undef _XOPEN_SOURCE")
        else:
            cppjit.cppdef("#undef slots")  # potentially pulled in by Qt/xapian.h

        cppjit.cppdef("""#include "Python.h"
           long py2long(PyObject* obj) { return PyLong_AsLong(obj); }""")

        pydoc.doc(cppjit.gbl.py2long)

        assert 1 == cppjit.gbl.py2long(1)

    def test04_avx(self):
        """Test usability of AVX by default."""

        import subprocess

        import cppjit

        has_avx = False
        try:
            f = open("/proc/cpuinfo", "r")
            for line in f.readlines():
                if "avx" in line:
                    has_avx = True
                    break
            f.close()
        except Exception:
            try:
                cli_arg = subprocess.check_output(["sysctl", "machdep.cpu.features"])
                has_avx = "avx" in cli_arg.decode("utf-8").strip().lower()
            except Exception:
                pass

        if has_avx:
            assert cppjit.cppdef("int check_avx() { return (int) __AVX__; }")
            assert cppjit.gbl.check_avx()  # attribute error if compilation failed

    def test05_default_template_arguments(self):
        """Calling a templated method on a templated class with all defaults used to crash."""

        import cppjit

        cppjit.cppdef("""\
        template<typename T>
        class AllDefault {
        public:
            AllDefault(int val) : m_t(val) {}
            template<int aap=1, int noot=2>
               int do_stuff() { return m_t+aap+noot; }

        public:
            T m_t;
        };""")

        a = cppjit.gbl.AllDefault[int](24)
        a.m_t = 21
        assert a.do_stuff() == 24

    def test06_default_float_or_unsigned_argument(self):
        """Calling with default argument for float or unsigned, which not parse"""

        import cppjit

        cppjit.cppdef("""\
        namespace Defaulters {
            float take_float(float a=5.f, int b=2) { return a*b; }
            double take_double(double a=5.f, int b=2) { return a*b; }
            long take_long(long a=5l, int b=2) { return a*b; }
            unsigned long take_ulong(unsigned long a=5ul, int b=2) { return a*b; }
        }""")

        ns = cppjit.gbl.Defaulters

        # the following default argument used to fail to parse
        assert ns.take_float() == 10.0
        assert ns.take_float(b=2) == 10.0
        assert ns.take_double() == 10.0
        assert ns.take_double(b=2) == 10.0
        assert ns.take_long() == 10
        assert ns.take_long(b=2) == 10
        assert ns.take_ulong() == 10
        assert ns.take_ulong(b=2) == 10

    def test07_class_refcounting(self):
        """The memory regulator would leave an additional refcount on classes"""

        import gc
        import sys

        import cppjit

        x = cppjit.gbl.std.vector["float"]
        old_refcnt = sys.getrefcount(x)

        y = x()
        del y
        gc.collect()

        assert sys.getrefcount(x) == old_refcnt

    @mark.xfail(condition=IS_MAC and IS_CLING, run=False, reason="Crahes on OSX-Cling")
    def test08_typedef_identity(self):
        """Nested typedefs should retain identity"""

        import cppjit

        cppjit.cppdef("""
        namespace PyABC {
        struct S1 {};
        struct S2 {
            typedef std::vector<const PyABC::S1*> S1_coll;
        }; }""")

        from cppjit.gbl import PyABC

        assert PyABC.S2.S1_coll
        assert "S1_coll" in dir(PyABC.S2)
        assert "vector<const PyABC::S1*>" not in dir(PyABC.S2)
        assert PyABC.S2.S1_coll is cppjit.gbl.std.vector("const PyABC::S1*")

    def test09_gil_not_released(self):
        """GIL was released by accident for by-value returns"""

        import cppjit

        something = 5.0

        code = """
            #include "Python.h"

            std::vector<float> some_foo_calling_python() {
              auto pyobj = reinterpret_cast<PyObject*>(ADDRESS);
              float f = (float)PyFloat_AsDouble(pyobj);
              std::vector<float> v;
              v.push_back(f);
              return v;
            }
            """.replace("ADDRESS", str(id(something)))

        cppjit.cppdef(code)
        cppjit.gbl.some_foo_calling_python()

    @mark.xfail(condition=IS_CLING, run=False, reason="Crashes on Cling")
    def test10_enum_in_global_space(self):
        """Enum declared in search.h did not appear in global space"""

        if IS_WINDOWS:
            return  # no such enum in MSVC's search.h

        import cppjit

        cppjit.include("search.h")

        assert cppjit.gbl.ACTION
        assert hasattr(cppjit.gbl, "ENTER")
        assert hasattr(cppjit.gbl, "FIND")

    def test11_cobject_addressing(self):
        """AsCObject (now as_cobject) had a deref too many"""

        import cppjit
        import cppjit.ll

        cppjit.cppdef("struct CObjA { CObjA() : m_int(42) {} int m_int; };")
        a = cppjit.gbl.CObjA()
        co = cppjit.ll.as_cobject(a)

        assert a is cppjit.bind_object(co, "CObjA")
        assert a.m_int == 42
        assert cppjit.bind_object(co, "CObjA").m_int == 42

    def test12_exception_while_exception(self):
        """Exception from SetDetailedException during exception handling used to crash"""

        import cppjit

        cppjit.cppdef("namespace AnExceptionNamespace { }")

        try:
            cppjit.gbl.blabla
        except AttributeError:
            try:
                cppjit.gbl.AnExceptionNamespace.blabla
            except AttributeError:
                pass

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test13_char_star_over_char(self):
        """Map str to const char* over char"""

        # This is debatable, but although a single character string passes through char,
        # it is more consistent to prefer const char* or std::string in all cases. The
        # original bug report is here:
        #    https://bitbucket.org/wlav/cppyy/issues/127/string-argument-resolves-incorrectly

        import cppjit

        cppjit.cppdef("""
        namespace csoc1 {
            std::string call(char) { return "char"; }
        }

        namespace csoc2 {
            std::string call(char) { return "char"; }
            std::string call(const char*) { return "const char*"; }
        }

        namespace csoc3 {
            std::string call(char) { return "char"; }
            std::string call(const std::string&) { return "string"; }
        }
        """)

        assert cppjit.gbl.csoc1.call("0") == "char"
        raises(ValueError, cppjit.gbl.csoc1.call, "00")

        assert cppjit.gbl.csoc2.call("0") == "const char*"
        assert cppjit.gbl.csoc2.call("00") == "const char*"

        assert cppjit.gbl.csoc3.call("0") == "string"
        assert cppjit.gbl.csoc3.call("00") == "string"

    def test14_struct_direct_definition(self):
        """Struct defined directly in a scope miseed scope in renormalized name"""

        import cppjit

        cppjit.cppdef("""
        namespace struct_direct_definition {
        struct Bar {
            struct Baz {
                std::vector<double> data;
            } baz[2];

            Bar() {
                baz[0].data.push_back(3.14);
                baz[1].data.push_back(2.73);
            }
        };

        class Foo {
        public:
            class Bar {
            public:
                Bar(): x(5) {}
                int x;
            } bar;

        }; }""")

        from cppjit.gbl import struct_direct_definition as sds

        b = sds.Bar()

        assert len(b.baz) == 2
        assert len(b.baz[0].data) == 1
        assert b.baz[0].data[0] == 3.14
        assert len(b.baz[1].data) == 1
        assert b.baz[1].data[0] == 2.73

        f = sds.Foo()
        assert f.bar.x == 5

    def test15_vector_vs_initializer_list(self):
        """Prefer vector in template and initializer_list in formal arguments"""

        import cppjit

        cppjit.cppdef("""
        namespace vec_vs_init {
           template<class T>
           std::string nameit1(const T& t) {
               return typeid(T).name();
           }
           template<class T>
           std::string nameit2(T&& t) {
               return typeid(T).name();
           }
           template<class T>
           size_t sizeit(T&& t) {
               return t.size();
           }
        }""")

        nameit1 = cppjit.gbl.vec_vs_init.nameit1
        assert "vector" in nameit1(list(range(10)))
        assert "vector" in nameit1(cppjit.gbl.std.vector[int]())

        nameit2 = cppjit.gbl.vec_vs_init.nameit2
        assert "vector" in nameit2(list(range(10)))
        assert "vector" in nameit2(cppjit.gbl.std.vector[int]())

        sizeit = cppjit.gbl.vec_vs_init.sizeit
        assert sizeit(list(range(10))) == 10

    def test16_iterable_enum(self):
        """Use template to iterate over an enum"""
        # from: https://stackoverflow.com/questions/52459530/pybind11-emulate-python-enum-behaviour

        import cppjit

        cppjit.cppdef("""
        template <typename Enum>
        struct my_iter_enum {
            struct iterator {
                using value_type = Enum;
                using difference_type = ptrdiff_t;
                using reference = const Enum&;
                using pointer = const Enum*;
                using iterator_category = std::input_iterator_tag;

                iterator(Enum value) : cur(value) {}

                reference operator*() { return cur; }
                pointer operator->() { return &cur; }
                bool operator==(const iterator& other) { return cur == other.cur; }
                bool operator!=(const iterator& other) { return !(*this == other); }
                iterator& operator++() { if (cur != Enum::Unknown) cur = static_cast<Enum>(static_cast<std::underlying_type_t<Enum>>(cur) + 1); return *this; }
                iterator operator++(int) { iterator other = *this; ++(*this); return other; }

            private:
                Enum cur;
                int TODO_why_is_this_placeholder_needed; // JIT error? Too aggressive optimization?
            };

            iterator begin() {
                return iterator(Enum::Black);
            }

            iterator end() {
                return iterator(Enum::Unknown);
            }
        };

        enum class MyColorEnum : char {
            Black = 1,
            Blue,
            Red,
            Yellow,
            Unknown
        };""")

        Color = cppjit.gbl.my_iter_enum["MyColorEnum"]
        assert Color.iterator

        c_iterable = Color()
        assert c_iterable.begin().__deref__() == chr(1)

        all_enums = []
        for c in c_iterable:
            all_enums.append(ord(c))
        assert all_enums == list(range(1, 5))

    def test17_operator_eq_pickup(self):
        """Base class python-side operator== interered with derived one"""

        import cppjit

        cppjit.cppdef("""
        namespace SelectOpEq {
        class Base {};

        class Derived1 : public Base {
        public:
            bool operator==(Derived1&) { return true; }
        };

        class Derived2 : public Base {
        public:
            bool operator!=(Derived2&) { return true; }
        }; }""")

        soe = cppjit.gbl.SelectOpEq

        soe.Base.__eq__ = lambda first, second: False
        soe.Base.__ne__ = lambda first, second: False

        a = soe.Derived1()
        b = soe.Derived1()

        assert a == b  # derived class' C++ operator== called

        a = soe.Derived2()
        b = soe.Derived2()

        assert a != b  # derived class' C++ operator!= called

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test18_operator_plus_overloads(self):
        """operator+(string, string) should return a string"""

        import cppjit

        a = cppjit.gbl.std.string("a")
        b = cppjit.gbl.std.string("b")

        assert a == "a"
        assert b == "b"

        assert type(a + b) == cppjit.gbl.std.string
        assert a + b == "ab"

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test19_std_string_hash(self):
        """Hashing of std::string"""

        import cppjit

        s = cppjit.gbl.std.string("text")
        d = {}

        # hashes of std::string larger than 2**31 would fail; run a couple of
        # strings to check although it may still succeed by accident (and never
        # was an issue on p3 anyway)
        for s in ["abc", "text", "321", "stuff", "very long string"]:
            d[s] = 1

    def test20_signed_char_ref(self):
        """Signed char executor was self-referencing"""

        import cppjit

        cppjit.cppdef("""
        class SignedCharRefGetter {
        public:
            void setter(signed char sc) { m_c = sc; }
            signed char& getter() { return m_c; }
            signed char m_c;
        };""")

        obj = cppjit.gbl.SignedCharRefGetter()
        obj.setter("c")

        assert obj.getter() == "c"

    def test21_temporaries_and_vector(self):
        """Extend a life line to references into a vector if needed"""

        import cppjit

        cppjit.cppdef("""
            std::vector<std::string> get_some_temporary_vector() { return { "x", "y", "z" }; }
        """)

        l = [e for e in cppjit.gbl.get_some_temporary_vector()]
        assert l == ["x", "y", "z"]

    def test22_initializer_list_and_temporary(self):
        """Conversion rules when selecting intializer_list v.s. temporary"""

        import cppjit

        cppjit.cppdef("""\
        namespace regression_test21 {
        std::string what_called = "";
        class Foo {
        public:
            Foo() = default;
            Foo(int i) {
                what_called += "Foo(int)";
            }
            Foo(std::initializer_list<uint8_t> il) {
                std::ostringstream os;
                os << "Foo(il<size=" << il.size() << ">)";
                what_called += os.str();
            }
        };

        class Bar {
        public:
            Bar() = default;
            Bar(int i) {
                what_called = "Bar(int)";
            }
            Bar(std::initializer_list<uint8_t> il) {
                std::ostringstream os;
                os << "Bar(il<size=" << il.size() << ">)";
                what_called += os.str();
            }
            Bar(Foo x) {
                what_called += "Bar(Foo)";
            }
        }; }""")

        r21 = cppjit.gbl.regression_test21

        assert len(r21.what_called) == 0

        r21.Bar(1)
        assert r21.what_called == "Bar(int)"

        r21.what_called = ""
        r21.Bar([1, 2])  # used to call Bar(Foo x) through implicit conversion
        assert r21.what_called == "Bar(il<size=2>)"

    def test23_copy_constructor(self):
        """Copy construct an object into an empty (NULL) proxy"""

        import gc

        import cppjit

        cppjit.cppdef("""\
        namespace regression_test22 {
        struct Countable {
            static int s_count;
            Countable() { ++s_count; }
            Countable(const Countable&) { ++s_count; }
            Countable& operator=(const Countable&) { return *this; }
            ~Countable() { --s_count; }
        };
        int Countable::s_count = 0;

        Countable copy_creation() { return Countable{}; }
        Countable* c;
        void destroyit() { delete c; }
        }""")

        r22 = cppjit.gbl.regression_test22

        assert r22.Countable.s_count == 0
        c = r22.Countable()
        assert r22.Countable.s_count == 1

        raises(ReferenceError, c.__init__, r22.Countable())
        gc.collect()
        assert r22.Countable.s_count == 1

        c.__assign__(r22.Countable())
        gc.collect()
        assert r22.Countable.s_count == 1

        c.__destruct__()
        assert r22.Countable.s_count == 0
        c.__init__(r22.Countable())
        gc.collect()
        assert r22.Countable.s_count == 1

        del c
        gc.collect()
        assert r22.Countable.s_count == 0

        c = cppjit.bind_object(cppjit.nullptr, r22.Countable)
        assert r22.Countable.s_count == 0
        c.__init__(r22.Countable())
        gc.collect()
        assert r22.Countable.s_count == 1

        del c
        gc.collect()
        assert r22.Countable.s_count == 0

        c = r22.copy_creation()
        assert r22.Countable.s_count == 1
        del c
        gc.collect()
        assert r22.Countable.s_count == 0

        c = r22.copy_creation()
        r22.c = c
        c.__python_owns__ = False
        del c
        gc.collect()
        assert r22.Countable.s_count == 1

        r22.destroyit()
        r22.c = cppjit.nullptr
        assert r22.Countable.s_count == 0

    def test24_C_style_enum(self):
        """Support C-style enum variable declarations"""

        import cppjit

        cppjit.cppdef("""\
        namespace CStyleEnum {
           enum MyEnum { kOne, kTwo };
           MyEnum my_enum = kOne;

           enum YourEnum { kThree, kFour };
           enum YourEnum your_enum = kThree;
        }""")

        CSE = cppjit.gbl.CStyleEnum

        assert CSE.my_enum == CSE.MyEnum.kOne
        CSE.my_enum = CSE.MyEnum.kTwo
        assert CSE.my_enum == CSE.MyEnum.kTwo

        # the following would fail b/c the type was not properly resolved
        assert CSE.your_enum == CSE.YourEnum.kThree
        CSE.your_enum = CSE.YourEnum.kFour
        assert CSE.your_enum == CSE.YourEnum.kFour

    def test25_const_iterator(self):
        """const_iterator failed to resolve the proper return type"""

        import cppjit

        cppjit.cppdef("""\
        namespace RooStuff {
        struct RooArg {};

        struct RooCollection {
            using const_iterator = std::vector<RooArg*>::const_iterator;
            std::vector<RooArg*> _list;

            RooCollection() { _list.emplace_back(); }
            const_iterator begin() const { return _list.begin(); }
            const_iterator end() const { return _list.end(); }
        }; }""")

        l = cppjit.gbl.RooStuff.RooCollection()

        i = 0
        for e in l:
            assert type(e) == cppjit.gbl.RooStuff.RooArg
            i += 1
        assert i

    def test26_const_charptr_data(self):
        """const char* is not const; const char* const is"""

        import cppjit

        cppjit.cppdef("""
        namespace ConstCharStar {
        struct ImGuiIO1 {
            ImGuiIO1() : BackendPlatformName(nullptr) {}
            const char* BackendPlatformName;
        };
        struct ImGuiIO2 {
            ImGuiIO2() : BackendPlatformName(nullptr) {}
            const char* const BackendPlatformName;
        }; }""")

        io = cppjit.gbl.ConstCharStar.ImGuiIO1()

        io.BackendPlatformName = "aap"
        assert io.BackendPlatformName == "aap"

        io.BackendPlatformName = "aap\0noot"
        io.BackendPlatformName = "aap\0noot"

        io = cppjit.gbl.ConstCharStar.ImGuiIO2()
        with raises(TypeError):
            io.BackendPlatformName = "aap"

    def test27_exception_by_value(self):
        """Proper memory management of exception return by value"""

        import gc

        import cppjit

        cppjit.cppdef("""\
        namespace ExceptionByValue {
        class Countable : std::exception {
        public:
            static int s_count;
            Countable() : fMsg("error") { ++s_count; }
            Countable(const Countable&) { ++s_count; }
            Countable& operator=(const Countable&) { return *this; }
            ~Countable() { --s_count; }
            const char* what() const throw() override { return fMsg.c_str(); }
        private:
            std::string fMsg;
        };

        int Countable::s_count = 0;

        Countable create_one() { return Countable{}; }
        int count() { return Countable::s_count; }
        }""")

        ns = cppjit.gbl.ExceptionByValue

        assert ns.count() == 0
        c = ns.create_one()
        assert ns.count() == 1

        del c
        gc.collect()
        assert ns.count() == 0

    def test28_exception_as_shared_ptr(self):
        """shared_ptr of an exception object null-checking"""

        import cppjit

        cppjit.cppdef("""\
        namespace exception_as_shared_ptr {
            std::shared_ptr<std::exception> get_shared_null() {
                return std::shared_ptr<std::exception>();
            }
        }""")

        null = cppjit.gbl.exception_as_shared_ptr.get_shared_null()
        assert not null

    @mark.xfail(
        condition=(IS_CLING and IS_MAC) or IS_MAC_ARM,
        run=False,
        reason="Dispatcher fix #53 introduces canonical types with std:: namespace that introduces OS X exceptions similar to test_stltypes",
    )
    def test29_callback_pointer_values(self):
        """Make sure pointer comparisons in callbacks work as expected"""

        import cppjit

        cppjit.cppdef("""\
        namespace addressof_regression {
        class ChangeBroadcaster;

        class ChangeListener {
        public:
            virtual ~ChangeListener() = default;
            virtual void changeCallback(ChangeBroadcaster*) = 0;
        };

        class ChangeBroadcaster {
        public:
            virtual ~ChangeBroadcaster() = default;
            void triggerChange() {
                std::for_each(l.begin(), l.end(), [this](auto* p) { p->changeCallback(this); });
            }

            void addChangeListener(ChangeListener* x) {
                l.push_back(x);
            }

        private:
            std::vector<ChangeListener*> l;
        };

        class BaseClass {
        public:
            virtual ~BaseClass() = default;
        };

        class DerivedClass : public BaseClass, public ChangeBroadcaster {
            /* empty */
        };

        class Implementation {
        public:
            Implementation() { }
            virtual ~Implementation() = default;
            DerivedClass derived;
        }; }""")

        ns = cppjit.gbl.addressof_regression

        class Glue(cppjit.multi(ns.Implementation, ns.ChangeListener)):
            def __init__(self):
                super(ns.Implementation, self).__init__()
                self.derived.addChangeListener(self)
                self.success = False

            def triggerChange(self):
                self.derived.triggerChange()

            def changeCallback(self, b):
                assert type(b) == type(self.derived)
                assert b == self.derived
                cast = cppjit.gbl.std.addressof[type(b)]
                assert cast(b) == cast(self.derived)
                self.success = True

        g = Glue()
        assert not g.success

        g.triggerChange()
        assert g.success

    def test30_uint64_t(self):
        """Failure due to typo"""

        import cppjit

        cppjit.cppdef("""\
        #include <limits>
        namespace UInt64_Typo {
            uint64_t Test(uint64_t v) { return v; }
            template <typename T> struct Struct { Struct(T t) : fT(t) {}; T fT; };
            template <typename T> Struct<T> TTest(T t) { return Struct<T>{t}; }
        }""")

        from cppjit.gbl import UInt64_Typo as ns

        std = cppjit.gbl.std
        uint64_t = cppjit.gbl.uint64_t
        umax64 = std.numeric_limits[uint64_t].max()
        int64_t = cppjit.gbl.int64_t
        max64 = std.numeric_limits[int64_t].max()
        min64 = std.numeric_limits[int64_t].min()

        assert max64 < umax64
        assert min64 < max64
        assert umax64 == ns.Test(umax64)

        assert ns.TTest(umax64).fT == umax64
        assert ns.TTest(max64).fT == max64
        assert ns.TTest(min64).fT == min64
        assert ns.TTest(1.01).fT == 1.01
        assert ns.TTest(True).fT == True
        assert type(ns.TTest(True).fT) == bool

    def test31_enum_in_dir(self):
        """Failed to pick up enum data"""

        import cppjit

        cppjit.cppdef("""\
        namespace enum_in_dir {
            long prod (long a, long b) { return a * b; }
            long prod (long a, long b, long c) { return a * b * c; }

            int a = 10;
            int b = 40;

            enum smth { ONE, };
            enum smth my_enum = smth::ONE;
        }""")

        all_names = set(dir(cppjit.gbl.enum_in_dir))

        required = {"prod", "a", "b", "smth", "my_enum"}
        assert all_names.intersection(required) == required

    def test32_typedef_class_enum(self):
        """Use of class enum with typedef'd type"""

        import cppjit

        cppjit.cppdef("""\
        namespace typedef_typed_enum {
        enum class Foo1 : uint16_t       { BAZ = 1, BAR = 2 };
        enum class Foo2 : unsigned short { BAZ = 3, BAR = 4 };
        enum       Foo3                  { BAZ = 5, BAR = 6 };

        template<typename ENUMTYPE>
        struct Info {
            ENUMTYPE x;
            uint8_t y;
        }; } """)

        ns = cppjit.gbl.typedef_typed_enum

        Info = ns.Info

        for Foo in [ns.Foo1, ns.Foo2, ns.Foo3]:
            o = Info[Foo]()
            o.x = Foo.BAR
            o.y = 0

            assert o.x == Foo.BAR
            assert o.y == 0

            o.y = 1
            assert o.x == Foo.BAR
            assert o.y == 1

            o.x = Foo.BAZ
            assert o.x == Foo.BAZ
            assert o.y == 1

    def test33_explicit_template_in_namespace(self):
        """Lookup of explicit template in namespace"""

        import cppjit

        cppjit.cppdef("""\
        namespace libchemist {
        namespace type {
            template<typename T> class tensor {};
        } // namespace type

        template<typename element_type = double, typename tensor_type = type::tensor<element_type>>
        class CanonicalMO {};

        template class CanonicalMO<double, type::tensor<double>>;

        auto produce() {
            return std::make_tuple(10., type::tensor<double>{});
        }

        } // namespace libchemist

        namespace property_types {
        namespace type {
            template<typename T>
            using canonical_mos = libchemist::CanonicalMO<T>;
        }

        auto produce() {
            return std::make_tuple(5., type::canonical_mos<double>{});
        }

        template<typename element_type = double, typename orbital_type = type::canonical_mos<element_type>>
        class ReferenceWavefunction {};

        template class ReferenceWavefunction<double>;

        template<class T>
        auto run_as() {
            return std::make_tuple(20., T{});
        } } // namespace type, property_types
        """)

        assert cppjit.gbl.property_types.type.canonical_mos["double"]
        assert cppjit.gbl.std.get[0](cppjit.gbl.libchemist.produce()) == 10.0
        assert cppjit.gbl.std.get[0](cppjit.gbl.property_types.produce()) == 5.0

        pt_type = cppjit.gbl.property_types.ReferenceWavefunction["double"]
        assert (
            cppjit.gbl.std.get[0](cppjit.gbl.property_types.run_as[pt_type]()) == 20.0
        )

    @mark.xfail(
        run=False,
        reason="Crashes on ClangRepl with 'toString not implemented', and on Cling",
    )
    def test34_print_empty_collection(self):
        """Print empty collection through Cling"""

        import cppjit

        # printing an empty collection used to have a missing symbol on 64b Windows
        v = cppjit.gbl.std.vector[int]()
        str(v)

    @mark.skipif(
        not CAN_JIT_STD_FILESYSTEM,
        reason="std::filesystem is unresolvable in the JIT on libstdc++ < 9",
    )
    @mark.xfail(condition=IS_CLING, run=False, reason="Crashes on Cling")
    def test35_filesytem(self):
        """Static path object used to crash on destruction"""

        if IS_WINDOWS:
            # TODO: this is b/c of the mangling: it's looking for '_std', but name is '__'
            skip("fails due to missing _std_fs_convert_narrow_to_wide symbol")

        import cppjit

        cppjit.cppdef("""\
        #include <filesystem>
        std::string stack_std_path() {
            std::filesystem::path p = "/usr";
            std::ostringstream os;
            os << p;
            return os.str();
        }""")

        assert cppjit.gbl.stack_std_path() == '"/usr"'

    def test36_ctypes_sizeof(self):
        """cppjit.sizeof forwards to ctypes.sizeof where necessary"""

        import ctypes

        import cppjit

        cppjit.cppdef("""\
        namespace test36_ctypes_sizeof {
            void func(uint32_t* param) {
                *param = 42;
            }
        }""")

        ns = cppjit.gbl.test36_ctypes_sizeof

        holder = ctypes.c_uint32(17)
        param = ctypes.pointer(holder)

        ns.func(param)
        assert holder.value == 42

        holder = ctypes.c_uint32(17)
        ns.func(holder)
        assert holder.value == 42

        assert cppjit.sizeof(param) == ctypes.sizeof(param)

    def test37_array_of_pointers_argument(self):
        """Passing an array of pointers used to crash"""

        import cppjit
        import cppjit.ll

        cppjit.cppdef("""\
        namespace ArrayOfPointers {
           void* test(int *arr[8], bool is_int=true) { return is_int ? (void*)arr : nullptr; }
           void* test(uint8_t *arr[8], bool is_int=false) { return is_int ? nullptr : (void*)arr; }
        }""")

        ns = cppjit.gbl.ArrayOfPointers

        N = 9

        for t, b in (("int*", True), ("uint8_t*", False)):
            arr = cppjit.ll.array_new[t](N, managed=True)
            assert arr.shape[0] == N
            assert len(arr) == N

            res = ns.test(arr, b)

            assert cppjit.addressof(res) == cppjit.addressof(arr)

    @mark.xfail(
        condition=IS_MAC and IS_CLING, run=False, reason="Crashes on OS X Cling"
    )
    def test38_char16_arrays(self):
        """Access to fixed-size char16 arrays as data members"""

        import warnings

        import cppjit
        import cppjit.ll

        cppjit.cppdef(r"""\
        namespace Char16Fixed {
        struct AxisInformation {
            char16_t name[6];
        };

        void fillem(AxisInformation* a, int N) {
            char16_t fill[] = {u'h', u'e', u'l', u'l', u'o', u'\0'};
            for (int i = 0; i < N; ++i)
                memcpy(a[i].name, fill, sizeof(fill));
        }}""")

        N = 10

        ns = cppjit.gbl.Char16Fixed

        ai = ns.AxisInformation()
        for s in ["hello", "hellow"]:
            ai.name = s
            len(ai.name) == 6
            assert ai.name[: len(s)] == s

        # isolate the warning configuration
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn warnings into errors

            with raises(RuntimeWarning) as exc:
                ai.name = "hellowd"
            assert "too long" in str(exc.value)

        # vector of objects
        va = cppjit.gbl.std.vector[ns.AxisInformation](N)
        ns.fillem(va.data(), N)
        for ai in va:
            assert len(ai.name) == 6
            assert ai.name[:5] == "hello"

        # array of objects
        aa = cppjit.gbl.std.array[ns.AxisInformation, N]()
        ns.fillem(aa.data(), N)
        for ai in aa:
            assert len(ai.name) == 6
            assert ai.name[:5] == "hello"

        # low-level array of objects
        aa = cppjit.ll.array_new[ns.AxisInformation](N)
        ns.fillem(aa, N)
        for ai in aa:
            assert len(ai.name) == 6
            assert ai.name[:5] == "hello"
        cppjit.ll.array_delete(aa)

    def test39_vector_of_pointers_conversion(self):
        """vector<T*>'s const T*& used to be T**, now T*"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace VectorOfPointers {
        struct Base1 { std::string name; explicit Base1(const std::string& name) : name(name) { }};
        struct Derived1 : Base1 { explicit Derived1(const std::string& name) : Base1(name) { } };
        struct Owner {
            Derived1 d1 { "d1" };
            Derived1 d2 { "d2" };
            std::vector<const Base1*> GetVector() { return { &d1, &d2 }; }
        }; }""")

        cppjit.gbl.VectorOfPointers
        from cppjit.gbl.VectorOfPointers import Base1, Derived1, Owner  # noqa: F401

        o = Owner()

        assert len(o.GetVector()) == 2
        assert type(o.GetVector()[0]) == Base1
        assert type(o.GetVector()[1]) == Base1
        assert o.GetVector()[0].name == "d1"
        assert o.GetVector()[1].name == "d2"

        v = o.GetVector()
        assert len(list(v)) == 2
        assert list(v)[0].name == "d1"
        assert list(v)[1].name == "d2"

        assert len(list(o.GetVector())) == 2
        assert list(o.GetVector())[0].name == "d1"
        assert list(o.GetVector())[1].name == "d2"

        l = list(v)
        v.__destruct__()
        assert l[0].name == "d1"
        assert l[1].name == "d2"

        cppjit.cppdef(r"""\
        namespace VectorOfPointers {
        struct Base2 { };
        struct Derived2 : Base2 { };
        struct Base3 { virtual ~Base3() noexcept = default; };
        struct Derived3 : Base3 { };

        Derived2 d2;
        Derived3 d3;
        std::vector<const Base2*> vec2 { &d2 };
        std::vector<const Base3*> vec3 { &d3 };
        }""")

        from cppjit.gbl import std  # noqa: F401
        from cppjit.gbl.VectorOfPointers import (
            Base2,
            Base3,  # noqa: F401
            Derived2,
            Derived3,
            vec2,
            vec3,
        )

        assert len(vec2) == 1
        assert type(vec2[0]) == Base2
        assert len(list(vec2)) == 1
        assert type(list(vec2)[0]) == Base2
        assert len([d for d in vec2 if isinstance(d, Derived2)]) == 0

        assert len(vec3) == 1
        assert type(vec3[0]) == Derived3
        assert len(list(vec3)) == 1
        assert type(list(vec2)[0]) == Base2
        assert len([d for d in vec3 if isinstance(d, Derived3)]) == 1

    @mark.xfail(condition=not IS_CLANG_REPL, run=False, reason="Crashes with Cling")
    def test40_explicit_initializer_list(self):
        """Construct and pass an explicit initializer list"""

        import cppjit

        cppjit.cppdef(r"""\
        namespace ExplicitInitializer {
        enum class TestEnum { Foo, Bar };

        using TestDictClass = std::initializer_list<std::pair<TestEnum, int>>;

        class TestClass {
        public:
            TestClass(TestDictClass x) {}
        }; }""")

        ns = cppjit.gbl.ExplicitInitializer

        TestPair = cppjit.gbl.std.pair[ns.TestEnum, int]
        arg = ns.TestDictClass(
            [TestPair(ns.TestEnum.Bar, 4), TestPair(ns.TestEnum.Foo, 12)]
        )
        assert ns.TestClass(arg)

    def test41_typedefed_enums(self):
        """Typedef-ed enums do not have enum tag in declarations"""

        import cppjit

        cppjit.cppdef("""\
        namespace TypedefedEnum {
        typedef enum {
            MONDAY    = 0,
            TUESDAY   = 1,
            WEDNESDAY = 2
        } Day;

        int func(const Day day) { return (int)day; }
        }""")

        ns = cppjit.gbl.TypedefedEnum

        assert ns.func(ns.WEDNESDAY) == 2

    def test42_char_arrays_consistence(self):
        """Consistent usage of char[] arrays"""

        import cppjit

        cppjit.cppdef(r"""
        namespace CharArrays {
        struct Foo {
            char val[10], *ptr;
            char values[2][10], *pointers[2];
        };

        void set_pointers(struct Foo* foo) {
        // populate arrays
            strcpy(foo->val, "howdy!");
            strcpy(foo->values[0], "hello");
            strcpy(foo->values[1], "world!");

        // set pointers
            foo->ptr = foo->val;
            foo->pointers[0] = foo->values[0];
            foo->pointers[1] = foo->values[1];
        } }""")

        ns = cppjit.gbl.CharArrays

        foo = ns.Foo()
        ns.set_pointers(foo)

        howdy = "howdy!"
        hello = "hello"
        world = "world!"

        assert "".join(foo.val)[: len(howdy)] == howdy
        assert foo.ptr == howdy
        assert foo.values[0].as_string() == hello
        assert foo.values[1].as_string() == world
        assert foo.pointers[0] == "hello"
        assert foo.pointers[1] == "world!"

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test43_static_with_default(self):
        """Call a static method with default args on an instance"""

        import cppjit

        cppjit.cppdef("""\
        namespace StaticWithDefault {
        struct MyClass {
            void static smethod(const std::string& s1, const std::string& s2="") {}
        }; }""")

        ns = cppjit.gbl.StaticWithDefault
        obj = ns.MyClass()

        obj.smethod("one", "two")
        obj.smethod("one")  # used to fail with vectorcall

    def test44_heuristic_mem_policy(self):
        """Ownership of arguments with heuristic memory policy"""

        import cppjit

        cppjit.cppdef("""\
        namespace MemTester {
           void CallRef( std::string&) {}
           void CallConstRef( const std::string&) {}
           void CallPtr( std::string*) {}
           void CallConstPtr( const std::string*) {}
        };
        """)

        try:
            # The scope with the heuristic memory policy is in a try-except-finally block
            # to ensure the memory policy is always reset.
            old_memory_policy = cppjit._backend.SetMemoryPolicy(
                cppjit._backend.kMemoryHeuristics
            )

            # Validate the intended behavior for different argument types:
            #   const ref : caller keeps ownership
            #   const ptr : caller keeps ownership
            #   ref       : caller keeps ownership
            #   ptr       : caller passed ownership to callee

            # The actual type doesn't matter
            args = [cppjit.gbl.std.string() for i in range(4)]

            cppjit.gbl.MemTester.CallConstRef(args[0])
            assert args[0].__python_owns__

            cppjit.gbl.MemTester.CallConstPtr(args[1])
            assert args[1].__python_owns__

            cppjit.gbl.MemTester.CallRef(args[2])
            assert args[2].__python_owns__

            cppjit.gbl.MemTester.CallPtr(args[3])
            assert not args[3].__python_owns__
            # Let's give back the ownership to Python here so there is no leak
            cppjit._backend.SetOwnership(args[3], True)
        except:
            raise  # rethrow the exception
        finally:
            cppjit._backend.SetMemoryPolicy(old_memory_policy)

    @mark.xfail(condition=IS_MAC, reason="Fails on OS X")
    def test45_typedef_resolution(self):
        """Typedefs starting with 'c'"""

        import cppjit

        cppjit.cppdef("""\
        namespace cppjit::interop {
            std::string ResolveName(const std::string& name);
        }
        typedef const int my_custom_type_t;
        typedef const int cmy_custom_type_t;
        """)

        assert cppjit.gbl.cppjit.interop.ResolveName("my_custom_type_t") == "const int"
        assert cppjit.gbl.cppjit.interop.ResolveName("cmy_custom_type_t") == "const int"

    @mark.xfail(
        condition=IS_MAC_ARM,
        run=False,
        reason="Crashes with exception not being caught on Apple Silicon",
    )
    def test46_exception_narrowing(self):
        """Exception narrowing to C++ exception of all overloads"""

        import cppjit

        cppjit.cppdef("""\
        namespace OverloadThrows {
        class Foo {
        public:
            void bar() { throw std::logic_error("This is fine"); }
            void bar() const { throw std::logic_error("This is fine"); }
        }; }""")

        ns = cppjit.gbl.OverloadThrows

        foo = ns.Foo()
        with raises(cppjit.gbl.std.logic_error):
            foo.bar()

    def test47_initializer_list_fail(self, capfd):
        """Conversion to intializer_list requires default constructor"""

        import cppjit

        cppjit.cppdef("""\
        namespace regression_test47 {
        std::size_t size = 0;
        struct IntWrapper { IntWrapper(int i) : fInt(i) {} int fInt; };
        void f(std::initializer_list<IntWrapper> l) {}
        void f(std::vector<IntWrapper> l) { size = l.size(); }
        }""")

        r47 = cppjit.gbl.regression_test47

        assert r47.size == 0
        capfd.readouterr()  # discard any output produced so far

        r47.f([1])
        assert r47.size == 1
        (out, err) = capfd.readouterr()
        assert out == ""
        assert err == ""

    def test48_templated_using_with_const(self):
        """Test parsing of template arguments mixed with using clause and constants"""

        import cppjit
        from cppjit import gbl

        cppjit.cppdef("""
            template <typename T, int N>
            struct NN {
              bool F = true;
            };
        """)

        assert gbl.NN["std::vector<int>::value_type, 5"]().F is True

    def test50_using_decl_base_this_offset(self):
        """A using-declaration-imported method from a non-zero-offset base must
        adjust the `this` pointer to that base's subobject.

        When a derived class pulls a base method into its own overload set with
        `using Base::method;` (typically to merge it with a same-named local
        overload), the method is still declared in the base. If that base is not
        the first one, its subobject sits at a non-zero offset in the derived
        object. The `this` offset used to be computed against the class the
        method was bound on (the derived class) rather than its declaring base,
        yielding a zero offset; the call then wrote through an unadjusted
        pointer, corrupting memory and crashing on destruction.
        """

        import cppjit

        cppjit.cppdef(r"""
        namespace UsingDeclThisOffset {

        // Fat first base so that SecondBase lands at a non-zero offset in Derived.
        struct FirstBase {
            long long a, b, c, d, e, f, g, h;
            FirstBase() : a(11), b(22), c(33), d(44), e(55), f(66), g(77), h(88) {}
            long long get_a() const { return a; }
            long long get_h() const { return h; }
        };

        struct SecondBase {
            int value;
            SecondBase() : value(-1) {}
            void set_value(int v) { value = v; }   // 1-arg setter in a non-zero-offset base
            int  get_value() const { return value; }
        };

        struct Derived : public FirstBase, public SecondBase {
            int extra;
            Derived() : extra(0) {}
            using SecondBase::set_value;                // import the 1-arg overload
            void set_value(int v, int w) { value = v + w; extra = w; }  // local 2-arg overload
        };

        std::ptrdiff_t secondbase_offset() {
            Derived* d = new Derived();
            std::ptrdiff_t off = (char*)static_cast<SecondBase*>(d) - (char*)d;
            delete d;
            return off;
        }

        } """)

        ns = cppjit.gbl.UsingDeclThisOffset

        # the bug only manifests when SecondBase is at a non-zero offset
        assert ns.secondbase_offset() != 0

        d = ns.Derived()

        # the 1-arg call resolves to the using-imported SecondBase::set_value(int)
        d.set_value(42)

        # the value lands in the correct SecondBase subobject ...
        assert d.get_value() == 42
        # ... and the FirstBase subobject (at offset 0) is left untouched; with a
        # wrong (zero) `this` offset the write would have clobbered FirstBase::a.
        assert d.get_a() == 11
        assert d.get_h() == 88

        # the local 2-arg overload still works and likewise leaves FirstBase intact
        d.set_value(10, 5)
        assert d.get_value() == 15
        assert d.extra == 5
        assert d.get_a() == 11

        # destruction must not crash (heap integrity preserved)
        del d

    def test51_nontype_enum_template_arg(self):
        """Regression test for a class template with a non-type enum parameter


        clang prints such an argument as a C-style cast, e.g.
        ``NonTypeEnumTmpl::Impl<float, (NonTypeEnumTmpl::EOp)0>``. Resolving
        that name (as happens when a base pointer is auto-downcast to its
        actual derived type) used to either fail to instantiate the template
        ("non-type template parameter must be an expression") or leave the
        interpreter in an error state that broke the next, unrelated JIT call
        wrapper ("failed to resolve function").
        """

        import cppjit

        cppjit.cppdef(r"""
        namespace NonTypeEnumTmpl {
            enum class EOp { Add = 0, Sub = 1, Mul = 2 };

            struct Base {
                virtual ~Base() {}
                virtual int code() const = 0;
            };

            template <typename T, EOp Op>
            struct Impl : public Base {
                int code() const override { return (int)Op; }
            };

            // Factory returning a *raw* base pointer whose dynamic type carries
            // a non-type enum template argument. Auto-downcasting it back to
            // Python makes cppjit resolve the actual type by name, i.e. the
            // cast-form "Impl<float, (EOp)0>".
            Base* get(int which) {
                static Impl<float, EOp::Add> a;
                static Impl<float, EOp::Sub> s;
                static Impl<float, EOp::Mul> m;
                if (which == 0) return &a;
                if (which == 1) return &s;
                return &m;
            }

            // Called after the downcast to detect interpreter poisoning: its
            // call wrapper is JIT-compiled only on first use.
            int probe(int x) { return x + 1; }
        }
        """)

        ns = cppjit.gbl.NonTypeEnumTmpl

        # resolving the derived template type during the auto-downcast must not
        # crash and must yield the actual (derived) class...
        op = ns.get(0)
        assert op.code() == 0
        assert "Impl<float" in type(op).__cpp_name__
        assert ns.get(1).code() == 1
        assert ns.get(2).code() == 2

        # ...nor leave the interpreter unable to compile a later call wrapper
        assert ns.probe(41) == 42

    def test52_no_cpp_name_for_template_arg(self):
        """Template arguments with no C++ equivalent raise TypeError"""

        import cppjit

        # a lambda has a __name__ ("<lambda>") that resolves to no C++ type
        with raises(TypeError):
            cppjit.gbl.std.vector[lambda: None]

        with raises(TypeError):
            cppjit.gbl.std.vector[object()]
