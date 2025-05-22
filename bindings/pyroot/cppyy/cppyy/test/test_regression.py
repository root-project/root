import os, sys, pytest
from pytest import mark, raises, skip
from support import setup_make, IS_WINDOWS, ispypy, IS_MAC_X86, IS_MAC_ARM, IS_MAC


class TestREGRESSION:
    helpout = []

    def setup_class(cls):
        import cppyy

        if sys.hexversion < 0x30d0000:
            def stringpager(text, cls=cls):
                cls.helpout.append(text)
        else:
            def stringpager(text, title='', cls=cls):
                cls.helpout.append(text)

        import pydoc
        pydoc.pager = stringpager

    def test01_kdcraw(self):
        """Doc strings for KDcrawIface (used to crash)."""

        import cppyy, pydoc

        # TODO: run a find for these paths
        qtpath = "/usr/include/qt5"
        kdcraw_h = "/usr/include/KF5/KDCRAW/kdcraw/kdcraw.h"
        if not os.path.isdir(qtpath) or not os.path.exists(kdcraw_h):
            skip("no KDE/Qt found, skipping test01_kdcraw")

        # need to resolve qt_version_tag for the incremental compiler; since
        # it's not otherwise used, just make something up
        cppyy.cppdef("int qt_version_tag = 42;")
        cppyy.add_include_path(qtpath)
        cppyy.include(kdcraw_h)

        # bring in some symbols to resolve the class
        cppyy.load_library("libQt5Core.so")
        cppyy.load_library("libKF5KDcraw.so")

        from cppyy.gbl import KDcrawIface

        self.__class__.helpout = []
        pydoc.doc(KDcrawIface.KDcraw)
        helptext  = ''.join(self.__class__.helpout)
        assert 'KDcraw' in helptext
        assert 'CPPInstance' in helptext

    def test02_dir(self):
        """For the same reasons as test01_kdcraw, this used to crash."""

        if ispypy:
            skip('hangs (??) in pypy')

        import cppyy, pydoc

        assert not '__abstractmethods__' in dir(cppyy.gbl.gInterpreter)
        assert '__class__' in dir(cppyy.gbl.gInterpreter)

        self.__class__.helpout = []
        pydoc.doc(cppyy.gbl.gInterpreter)
        helptext = ''.join(self.__class__.helpout)
        assert 'TInterpreter' in helptext
        assert 'CPPInstance' in helptext
        assert 'AddIncludePath' in helptext

        cppyy.cppdef("namespace cppyy_regression_test { void iii() {}; }")

        assert not 'iii' in cppyy.gbl.cppyy_regression_test.__dict__
        assert not '__abstractmethods__' in dir(cppyy.gbl.cppyy_regression_test)
        assert '__class__' in dir(cppyy.gbl.cppyy_regression_test)
        assert 'iii' in dir(cppyy.gbl.cppyy_regression_test)

        assert not 'iii' in cppyy.gbl.cppyy_regression_test.__dict__
        assert cppyy.gbl.cppyy_regression_test.iii
        assert 'iii' in cppyy.gbl.cppyy_regression_test.__dict__

        self.__class__.helpout = []
        pydoc.doc(cppyy.gbl.cppyy_regression_test)
        helptext = ''.join(self.__class__.helpout)
        # TODO: it's deeply silly that namespaces inherit from CPPInstance (in CPyCppyy)
        assert ('CPPInstance' in helptext or 'CPPNamespace' in helptext)

    @mark.xfail(condition=IS_MAC, reason="Fails on OSX")
    def test03_pyfunc_doc(self):
        """Help on a generated pyfunc used to crash."""

        import cppyy, pydoc, sys
        import sysconfig as sc

        cppyy.add_include_path(sc.get_config_var("INCLUDEPY"))
        if sys.hexversion < 0x3000000:
            cppyy.cppdef("#undef _POSIX_C_SOURCE")
            cppyy.cppdef("#undef _XOPEN_SOURCE")
        else:
            cppyy.cppdef("#undef slots")     # potentially pulled in by Qt/xapian.h

        cppyy.cppdef("""#include "Python.h"
           long py2long(PyObject* obj) { return PyLong_AsLong(obj); }""")

        pydoc.doc(cppyy.gbl.py2long)

        assert 1 == cppyy.gbl.py2long(1)

    @mark.xfail(reason="Fails on \"alma9 modules_off runtime_cxxmodules=Off\"")
    def test04_avx(self):
        """Test usability of AVX by default."""

        import cppyy, subprocess

        has_avx = False
        try:
            f = open('/proc/cpuinfo', 'r')
            for line in f.readlines():
                if 'avx' in line:
                    has_avx = True
                    break
            f.close()
        except Exception:
            try:
                cli_arg = subprocess.check_output(['sysctl', 'machdep.cpu.features'])
                has_avx = 'avx' in cli_arg.decode("utf-8").strip().lower()
            except Exception:
                pass

        if has_avx:
            assert cppyy.cppdef('int check_avx() { return (int) __AVX__; }')
            assert cppyy.gbl.check_avx()   # attribute error if compilation failed

    def test05_default_template_arguments(self):
        """Calling a templated method on a templated class with all defaults used to crash."""

        import cppyy

        cppyy.cppdef("""\
        template<typename T>
        class AllDefault {
        public:
            AllDefault(int val) : m_t(val) {}
            template<int aap=1, int noot=2>
               int do_stuff() { return m_t+aap+noot; }

        public:
            T m_t;
        };""")

        a = cppyy.gbl.AllDefault[int](24)
        a.m_t = 21;
        assert a.do_stuff() == 24

    def test06_default_float_or_unsigned_argument(self):
        """Calling with default argument for float or unsigned, which not parse"""

        import cppyy

        cppyy.cppdef("""\
        namespace Defaulters {
            float take_float(float a=5.f, int b=2) { return a*b; }
            double take_double(double a=5.f, int b=2) { return a*b; }
            long take_long(long a=5l, int b=2) { return a*b; }
            unsigned long take_ulong(unsigned long a=5ul, int b=2) { return a*b; }
        }""")

        ns = cppyy.gbl.Defaulters

      # the following default argument used to fail to parse
        assert ns.take_float()     == 10.
        assert ns.take_float(b=2)  == 10.
        assert ns.take_double()    == 10.
        assert ns.take_double(b=2) == 10.
        assert ns.take_long()      == 10
        assert ns.take_long(b=2)   == 10
        assert ns.take_ulong()     == 10
        assert ns.take_ulong(b=2)  == 10

    def test07_class_refcounting(self):
        """The memory regulator would leave an additional refcount on classes"""

        import cppyy, gc, sys

        x = cppyy.gbl.std.vector['float']
        old_refcnt = sys.getrefcount(x)

        y = x()
        del y
        gc.collect()

        assert sys.getrefcount(x) == old_refcnt

    def test08_typedef_identity(self):
        """Nested typedefs should retain identity"""

        import cppyy

        cppyy.cppdef("""
        namespace PyABC {
        struct S1 {};
        struct S2 {
            typedef std::vector<const PyABC::S1*> S1_coll;
        }; }""")

        from cppyy.gbl import PyABC

        assert PyABC.S2.S1_coll
        assert 'S1_coll' in dir(PyABC.S2)
        assert not 'vector<const PyABC::S1*>' in dir(PyABC.S2)
        assert PyABC.S2.S1_coll is cppyy.gbl.std.vector('const PyABC::S1*')

    def test09_gil_not_released(self):
        """GIL was released by accident for by-value returns"""

        import cppyy

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

        cppyy.cppdef(code)
        cppyy.gbl.some_foo_calling_python()

    def test10_enum_in_global_space(self):
        """Enum declared in search.h did not appear in global space"""

        if IS_WINDOWS:
            return           # no such enum in MSVC's search.h

        import cppyy

        cppyy.include('search.h')

        assert cppyy.gbl.ACTION
        assert hasattr(cppyy.gbl, 'ENTER')
        assert hasattr(cppyy.gbl, 'FIND')

    def test11_cobject_addressing(self):
        """AsCObject (now as_cobject) had a deref too many"""

        import cppyy
        import cppyy.ll

        cppyy.cppdef('struct CObjA { CObjA() : m_int(42) {} int m_int; };')
        a = cppyy.gbl.CObjA()
        co = cppyy.ll.as_cobject(a)

        assert a == cppyy.bind_object(co, 'CObjA')
        assert a.m_int == 42
        assert cppyy.bind_object(co, 'CObjA').m_int == 42

    def test12_exception_while_exception(self):
        """Exception from SetDetailedException during exception handling used to crash"""

        import cppyy

        cppyy.cppdef("namespace AnExceptionNamespace { }")

        try:
            cppyy.gbl.blabla
        except AttributeError:
            try:
                cppyy.gbl.AnExceptionNamespace.blabla
            except AttributeError:
                pass

    def test13_char_star_over_char(self):
        """Map str to const char* over char"""

      # This is debatable, but although a single character string passes through char,
      # it is more consistent to prefer const char* or std::string in all cases. The
      # original bug report is here:
      #    https://bitbucket.org/wlav/cppyy/issues/127/string-argument-resolves-incorrectly

        import cppyy

        cppyy.cppdef("""
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

        assert cppyy.gbl.csoc1.call('0') == 'char'
        raises(ValueError, cppyy.gbl.csoc1.call, '00')

        assert cppyy.gbl.csoc2.call('0')  == 'const char*'
        assert cppyy.gbl.csoc2.call('00') == 'const char*'

        assert cppyy.gbl.csoc3.call('0')  == 'string'
        assert cppyy.gbl.csoc3.call('00') == 'string'

    def test14_struct_direct_definition(self):
        """Struct defined directly in a scope miseed scope in renormalized name"""

        import cppyy

        cppyy.cppdef("""
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

        from cppyy.gbl import struct_direct_definition as sds

        b = sds.Bar()

        assert len(b.baz) == 2
        assert len(b.baz[0].data) == 1
        assert b.baz[0].data[0]   == 3.14
        assert len(b.baz[1].data) == 1
        assert b.baz[1].data[0]   == 2.73

        f = sds.Foo()
        assert f.bar.x == 5

    def test15_vector_vs_initializer_list(self):
        """Prefer vector in template and initializer_list in formal arguments"""

        import cppyy

        cppyy.cppdef("""
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

        nameit1 = cppyy.gbl.vec_vs_init.nameit1
        assert 'vector' in nameit1(list(range(10)))
        assert 'vector' in nameit1(cppyy.gbl.std.vector[int]())

        nameit2 = cppyy.gbl.vec_vs_init.nameit2
        assert 'vector' in nameit2(list(range(10)))
        assert 'vector' in nameit2(cppyy.gbl.std.vector[int]())

        sizeit = cppyy.gbl.vec_vs_init.sizeit
        assert sizeit(list(range(10))) == 10

    @mark.xfail()
    def test16_iterable_enum(self):
        """Use template to iterate over an enum"""
      # from: https://stackoverflow.com/questions/52459530/pybind11-emulate-python-enum-behaviour

        import cppyy

        cppyy.cppdef("""
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

        Color = cppyy.gbl.my_iter_enum['MyColorEnum']
        assert Color.iterator

        c_iterable = Color()
        assert c_iterable.begin().__deref__() == chr(1)

        all_enums = []
        for c in c_iterable:
            all_enums.append(ord(c))
        assert all_enums == list(range(1, 5))

    def test17_operator_eq_pickup(self):
        """Base class python-side operator== interered with derived one"""

        import cppyy

        cppyy.cppdef("""
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

        soe = cppyy.gbl.SelectOpEq

        soe.Base.__eq__ = lambda first, second: False
        soe.Base.__ne__ = lambda first, second: False

        a = soe.Derived1()
        b = soe.Derived1()

        assert a == b             # derived class' C++ operator== called

        a = soe.Derived2()
        b = soe.Derived2()

        assert a != b             # derived class' C++ operator!= called

    @mark.xfail()
    def test18_operator_plus_overloads(self):
        """operator+(string, string) should return a string"""

        import cppyy

        a = cppyy.gbl.std.string("a")
        b = cppyy.gbl.std.string("b")

        assert a == 'a'
        assert b == 'b'

        assert type(a+b) == cppyy.gbl.std.string
        assert a+b == 'ab'

    def test19_std_string_hash(self):
        """Hashing of std::string"""

        import cppyy

        import cppyy

        s = cppyy.gbl.std.string("text")
        d = {}

      # hashes of std::string larger than 2**31 would fail; run a couple of
      # strings to check although it may still succeed by accident (and never
      # was an issue on p3 anyway)
        for s in ['abc', 'text', '321', 'stuff', 'very long string']:
            d[s] = 1

    def test20_signed_char_ref(self):
        """Signed char executor was self-referencing"""

        import cppyy

        cppyy.cppdef("""
        class SignedCharRefGetter {
        public:
            void setter(signed char sc) { m_c = sc; }
            signed char& getter() { return m_c; }
            signed char m_c;
        };""")

        obj = cppyy.gbl.SignedCharRefGetter()
        obj.setter('c')

        assert obj.getter() == 'c'

    def test21_temporaries_and_vector(self):
        """Extend a life line to references into a vector if needed"""

        import cppyy

        cppyy.cppdef("""
            std::vector<std::string> get_some_temporary_vector() { return { "x", "y", "z" }; }
        """)

        l = [e for e in cppyy.gbl.get_some_temporary_vector()]
        assert l == ['x', 'y', 'z']

    def test22_initializer_list_and_temporary(self):
        """Conversion rules when selecting intializer_list v.s. temporary"""

        import cppyy

        cppyy.cppdef("""\
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

        r21 = cppyy.gbl.regression_test21

        assert len(r21.what_called) == 0

        r21.Bar(1)
        assert r21.what_called == 'Bar(int)'

        r21.what_called = ''
        r21.Bar([1,2])  # used to call Bar(Foo x) through implicit conversion
        assert r21.what_called == 'Bar(il<size=2>)'

    def test23_copy_constructor(self):
        """Copy construct an object into an empty (NULL) proxy"""

        import cppyy, gc

        cppyy.cppdef("""\
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

        r22 = cppyy.gbl.regression_test22

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

        c = cppyy.bind_object(cppyy.nullptr, r22.Countable)
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
        r22.c = cppyy.nullptr
        assert r22.Countable.s_count == 0

    def test24_C_style_enum(self):
        """Support C-style enum variable declarations"""

        import cppyy

        cppyy.cppdef("""\
        namespace CStyleEnum {
           enum MyEnum { kOne, kTwo };
           MyEnum my_enum = kOne;

           enum YourEnum { kThree, kFour };
           enum YourEnum your_enum = kThree;
        }""")

        CSE = cppyy.gbl.CStyleEnum

        assert CSE.my_enum == CSE.MyEnum.kOne
        CSE.my_enum = CSE.MyEnum.kTwo
        assert CSE.my_enum == CSE.MyEnum.kTwo

      # the following would fail b/c the type was not properly resolved
        assert CSE.your_enum == CSE.YourEnum.kThree
        CSE.your_enum = CSE.YourEnum.kFour
        assert CSE.your_enum == CSE.YourEnum.kFour

    def test25_const_iterator(self):
        """const_iterator failed to resolve the proper return type"""

        import cppyy

        cppyy.cppdef("""\
        namespace RooStuff {
        struct RooArg {};

        struct RooCollection {
            using const_iterator = std::vector<RooArg*>::const_iterator;
            std::vector<RooArg*> _list;

            RooCollection() { _list.emplace_back(); }
            const_iterator begin() const { return _list.begin(); }
            const_iterator end() const { return _list.end(); }
        }; }""")

        l = cppyy.gbl.RooStuff.RooCollection()

        i = 0
        for e in l:
            assert type(e) == cppyy.gbl.RooStuff.RooArg
            i += 1
        assert i

    @mark.xfail()
    def test26_const_charptr_data(self):
        """const char* is not const; const char* const is"""

        import cppyy

        cppyy.cppdef("""
        namespace ConstCharStar {
        struct ImGuiIO1 {
            ImGuiIO1() : BackendPlatformName(nullptr) {}
            const char* BackendPlatformName;
        };
        struct ImGuiIO2 {
            ImGuiIO2() : BackendPlatformName(nullptr) {}
            const char* const BackendPlatformName;
        }; }""")

        io = cppyy.gbl.ConstCharStar.ImGuiIO1()

        io.BackendPlatformName = "aap"
        assert io.BackendPlatformName == "aap"

        io.BackendPlatformName = "aap\0noot"
        io.BackendPlatformName = "aap\0noot"

        io = cppyy.gbl.ConstCharStar.ImGuiIO2()
        with raises(TypeError):
            io.BackendPlatformName = "aap"

    def test27_exception_by_value(self):
        """Proper memory management of exception return by value"""

        import cppyy, gc

        cppyy.cppdef("""\
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

        ns = cppyy.gbl.ExceptionByValue

        assert ns.count() == 0
        c = ns.create_one()
        assert ns.count() == 1

        del c
        gc.collect()
        assert ns.count() == 0

    def test28_exception_as_shared_ptr(self):
        """shared_ptr of an exception object null-checking"""

        import cppyy

        cppyy.cppdef("""\
        namespace exception_as_shared_ptr {
            std::shared_ptr<std::exception> get_shared_null() {
                return std::shared_ptr<std::exception>();
            }
        }""")

        null = cppyy.gbl.exception_as_shared_ptr.get_shared_null()
        assert not null

    @mark.skip()
    def test29_callback_pointer_values(self):
        """Make sure pointer comparisons in callbacks work as expected"""

        import cppyy

        cppyy.cppdef("""\
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

        ns = cppyy.gbl.addressof_regression

        class Glue(cppyy.multi(ns.Implementation, ns.ChangeListener)):
            def __init__(self):
                super(ns.Implementation, self).__init__()
                self.derived.addChangeListener(self)
                self.success = False

            def triggerChange(self):
                self.derived.triggerChange()

            def changeCallback(self, b):
                assert type(b) == type(self.derived)
                assert b == self.derived
                cast = cppyy.gbl.std.addressof[type(b)]
                assert cast(b) == cast(self.derived)
                self.success = True

        g = Glue()
        assert not g.success

        g.triggerChange()
        assert g.success

    @mark.xfail()
    def test30_uint64_t(self):
        """Failure due to typo"""

        import cppyy

        cppyy.cppdef("""\
        #include <limits>
        namespace UInt64_Typo {
            uint64_t Test(uint64_t v) { return v; }
            template <typename T> struct Struct { Struct(T t) : fT(t) {}; T fT; };
            template <typename T> Struct<T> TTest(T t) { return Struct<T>{t}; }
        }""")

        from cppyy.gbl import UInt64_Typo as ns

        std = cppyy.gbl.std
        uint64_t = cppyy.gbl.uint64_t
        umax64   = std.numeric_limits[uint64_t].max()
        int64_t  = cppyy.gbl.int64_t
        max64    = std.numeric_limits[int64_t].max()
        min64    = std.numeric_limits[int64_t].min()

        assert max64 < umax64
        assert min64 < max64
        assert umax64 == ns.Test(umax64)

        assert ns.TTest(umax64).fT == umax64
        assert ns.TTest(max64).fT  ==  max64
        assert ns.TTest(min64).fT  ==  min64
        assert ns.TTest(1.01).fT == 1.01
        assert ns.TTest(True).fT == True
        assert type(ns.TTest(True).fT) == bool

    @mark.xfail()
    def test31_enum_in_dir(self):
        """Failed to pick up enum data"""

        import cppyy

        cppyy.cppdef("""\
        namespace enum_in_dir {
            long prod (long a, long b) { return a * b; }
            long prod (long a, long b, long c) { return a * b * c; }

            int a = 10;
            int b = 40;

            enum smth { ONE, };
            enum smth my_enum = smth::ONE;
        }""")

        all_names = set(dir(cppyy.gbl.enum_in_dir))

        required = {'prod', 'a', 'b', 'smth', 'my_enum'}
        assert all_names.intersection(required) == required

    @mark.xfail()
    def test32_typedef_class_enum(self):
        """Use of class enum with typedef'd type"""

        import cppyy

        cppyy.cppdef("""\
        namespace typedef_typed_enum {
        enum class Foo1 : uint16_t       { BAZ = 1, BAR = 2 };
        enum class Foo2 : unsigned short { BAZ = 3, BAR = 4 };
        enum       Foo3                  { BAZ = 5, BAR = 6 };

        template<typename ENUMTYPE>
        struct Info {
            ENUMTYPE x;
            uint8_t y;
        }; } """)

        ns = cppyy.gbl.typedef_typed_enum

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

    @mark.xfail()
    def test33_explicit_template_in_namespace(self):
        """Lookup of explicit template in namespace"""

        import cppyy

        cppyy.cppdef("""\
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

        assert cppyy.gbl.property_types.type.canonical_mos['double']
        assert cppyy.gbl.std.get[0](cppyy.gbl.libchemist.produce())     == 10.
        assert cppyy.gbl.std.get[0](cppyy.gbl.property_types.produce()) ==  5.

        pt_type = cppyy.gbl.property_types.ReferenceWavefunction['double']
        assert cppyy.gbl.std.get[0](cppyy.gbl.property_types.run_as[pt_type]()) ==  20.

    def test34_print_empty_collection(self):
        """Print empty collection through Cling"""

        import cppyy

      # printing an empty collection used to have a missing symbol on 64b Windows
        v = cppyy.gbl.std.vector[int]()
        str(v)

    def test35_filesytem(self):
        """Static path object used to crash on destruction"""

        if IS_WINDOWS:
            # TODO: this is b/c of the mangling: it's looking for '_std', but name is '__'
            skip('fails due to missing _std_fs_convert_narrow_to_wide symbol')

        import cppyy

        if cppyy.gbl.gInterpreter.ProcessLine("__cplusplus;") > 201402:
            cppyy.cppdef("""\
            #include <filesystem>
            std::string stack_std_path() {
                std::filesystem::path p = "/usr";
                std::ostringstream os;
                os << p;
                return os.str();
            }""")

            assert cppyy.gbl.stack_std_path() == '"/usr"'

    def test36_ctypes_sizeof(self):
        """cppyy.sizeof forwards to ctypes.sizeof where necessary"""

        import cppyy, ctypes

        cppyy.cppdef("""\
        namespace test36_ctypes_sizeof {
            void func(uint32_t* param) {
                *param = 42;
            }
        }""")

        ns = cppyy.gbl.test36_ctypes_sizeof

        holder = ctypes.c_uint32(17)
        param = ctypes.pointer(holder)

        ns.func(param)
        assert holder.value == 42

        holder = ctypes.c_uint32(17)
        ns.func(holder)
        assert holder.value == 42

        assert cppyy.sizeof(param) == ctypes.sizeof(param)

    def test37_array_of_pointers_argument(self):
        """Passing an array of pointers used to crash"""

        import cppyy
        import cppyy.ll

        cppyy.cppdef("""\
        namespace ArrayOfPointers {
           void* test(int *arr[8], bool is_int=true) { return is_int ? (void*)arr : nullptr; }
           void* test(uint8_t *arr[8], bool is_int=false) { return is_int ? nullptr : (void*)arr; }
        }""")

        ns = cppyy.gbl.ArrayOfPointers

        N = 9

        for t, b in (('int*', True), ('uint8_t*', False)):
            arr = cppyy.ll.array_new[t](N, managed=True)
            assert arr.shape[0] == N
            assert len(arr) == N

            res = ns.test(arr, b)

            assert cppyy.addressof(res) == cppyy.addressof(arr)

    def test38_char16_arrays(self):
        """Access to fixed-size char16 arrays as data members"""

        import cppyy
        import cppyy.ll
        import warnings

        cppyy.cppdef(r"""\
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

        ns = cppyy.gbl.Char16Fixed

        ai = ns.AxisInformation()
        for s in [u'hello', u'hellow']:
            ai.name = s
            len(ai.name) == 6
            assert ai.name[:len(s)] == s

        with warnings.catch_warnings(record=True) as w:
            ai.name = u'hellowd'
            assert 'too long' in str(w[-1].message)

        # vector of objects
        va = cppyy.gbl.std.vector[ns.AxisInformation](N)
        ns.fillem(va.data(), N)
        for ai in va:
            assert len(ai.name) == 6
            assert ai.name[:5] == u'hello'

        # array of objects
        aa = cppyy.gbl.std.array[ns.AxisInformation, N]()
        ns.fillem(aa.data(), N)
        for ai in aa:
            assert len(ai.name) == 6
            assert ai.name[:5] == u'hello'

        # low-level array of objects
        aa = cppyy.ll.array_new[ns.AxisInformation](N)
        ns.fillem(aa, N)
        for ai in aa:
            assert len(ai.name) == 6
            assert ai.name[:5] == u'hello'
        cppyy.ll.array_delete(aa)

    def test39_vector_of_pointers_conversion(self):
        """vector<T*>'s const T*& used to be T**, now T*"""

        import cppyy

        cppyy.cppdef(r"""\
        namespace VectorOfPointers {
        struct Base1 { std::string name; explicit Base1(const std::string& name) : name(name) { }};
        struct Derived1 : Base1 { explicit Derived1(const std::string& name) : Base1(name) { } };
        struct Owner {
            Derived1 d1 { "d1" };
            Derived1 d2 { "d2" };
            std::vector<const Base1*> GetVector() { return { &d1, &d2 }; }
        }; }""")

        cppyy.gbl.VectorOfPointers
        from cppyy.gbl.VectorOfPointers import Base1, Derived1, Owner

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

        cppyy.cppdef(r"""\
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

        from cppyy.gbl import std
        from cppyy.gbl.VectorOfPointers import Base2, Derived2, Base3, Derived3, vec2, vec3

        assert len(vec2)     == 1
        assert type(vec2[0]) == Base2
        assert len(list(vec2))     == 1
        assert type(list(vec2)[0]) == Base2
        assert len([d for d in vec2 if isinstance(d, Derived2)]) == 0

        assert len(vec3)     == 1
        assert type(vec3[0]) == Derived3
        assert len(list(vec3))     == 1
        assert type(list(vec2)[0]) == Base2
        assert len([d for d in vec3 if isinstance(d, Derived3)]) == 1

    def test40_explicit_initializer_list(self):
        """Construct and pass an explicit initializer list"""

        import cppyy

        cppyy.cppdef(r"""\
        namespace ExplicitInitializer {
        enum class TestEnum { Foo, Bar };

        using TestDictClass = std::initializer_list<std::pair<TestEnum, int>>;

        class TestClass {
        public:
            TestClass(TestDictClass x) {}
        }; }""")

        ns = cppyy.gbl.ExplicitInitializer

        TestPair = cppyy.gbl.std.pair[ns.TestEnum, int]
        arg = ns.TestDictClass([TestPair(ns.TestEnum.Bar, 4), TestPair(ns.TestEnum.Foo, 12)])
        assert ns.TestClass(arg)

    def test41_typedefed_enums(self):
        """Typedef-ed enums do not have enum tag in declarations"""

        import cppyy
        cppyy.cppdef("""\
        namespace TypedefedEnum {
        typedef enum {
            MONDAY    = 0,
            TUESDAY   = 1,
            WEDNESDAY = 2
        } Day;

        int func(const Day day) { return (int)day; }
        }""")

        ns = cppyy.gbl.TypedefedEnum

        assert ns.func(ns.WEDNESDAY) == 2

    def test42_char_arrays_consistence(self):
        """Consistent usage of char[] arrays"""

        import cppyy

        cppyy.cppdef(r"""
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

        ns = cppyy.gbl.CharArrays

        foo = ns.Foo()
        ns.set_pointers(foo)

        howdy = 'howdy!'
        hello = 'hello'
        world = 'world!'

        assert ''.join(foo.val)[:len(howdy)] == howdy
        assert foo.ptr == howdy
        assert foo.values[0].as_string() == hello
        assert foo.values[1].as_string() == world
        assert foo.pointers[0] == 'hello'
        assert foo.pointers[1] == 'world!'

    def test43_static_with_default(self):
        """Call a static method with default args on an instance"""

        import cppyy

        cppyy.cppdef("""\
        namespace StaticWithDefault {
        struct MyClass {
            void static smethod(const std::string& s1, const std::string& s2="") {}
        }; }""")

        ns = cppyy.gbl.StaticWithDefault
        obj = ns.MyClass()

        obj.smethod("one", "two")
        obj.smethod("one")        # used to fail with vectorcall

    def test44_heuristic_mem_policy(self):
        """Ownership of arguments with heuristic memory policy"""

        import cppyy

        cppyy.cppdef("""\
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
            old_memory_policy = cppyy._backend.SetMemoryPolicy(cppyy._backend.kMemoryHeuristics)

            # Validate the intended behavior for different argument types:
            #   const ref : caller keeps ownership
            #   const ptr : caller keeps ownership
            #   ref       : caller keeps ownership
            #   ptr       : caller passed ownership to callee

            # The actual type doesn't matter
            args = [cppyy.gbl.std.string() for i in range(4)]

            cppyy.gbl.MemTester.CallConstRef(args[0])
            assert args[0].__python_owns__

            cppyy.gbl.MemTester.CallConstPtr(args[1])
            assert args[1].__python_owns__

            cppyy.gbl.MemTester.CallRef(args[2])
            assert args[2].__python_owns__

            cppyy.gbl.MemTester.CallPtr(args[3])
            assert not args[3].__python_owns__
            # Let's give back the ownership to Python here so there is no leak
            cppyy._backend.SetOwnership(args[3], True)
        except:
            raise # rethrow the exception
        finally:
            cppyy._backend.SetMemoryPolicy(old_memory_policy)

    @mark.xfail()
    def test45_typedef_resolution(self):
        """Typedefs starting with 'c'"""

        import cppyy

        cppyy.cppdef("""\
        typedef const int my_custom_type_t;
        typedef const int cmy_custom_type_t;
        """)

        assert cppyy.gbl.CppyyLegacy.TClassEdit.ResolveTypedef("my_custom_type_t") == "const int"
        assert cppyy.gbl.CppyyLegacy.TClassEdit.ResolveTypedef("cmy_custom_type_t") == "const int"

    @mark.xfail(run=False, condition=IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test46_exception_narrowing(self):
        """Exception narrowing to C++ exception of all overloads"""

        import cppyy

        cppyy.cppdef("""\
        namespace OverloadThrows {
        class Foo {
        public:
            void bar() { throw std::logic_error("This is fine"); }
            void bar() const { throw std::logic_error("This is fine"); }
        }; }""")

        ns = cppyy.gbl.OverloadThrows

        foo = ns.Foo()
        with raises(cppyy.gbl.std.logic_error):
            foo.bar()


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
