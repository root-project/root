import py, os, pytest
from pytest import raises, skip, mark
from support import setup_make, pylong, IS_MAC_ARM


currpath = os.getcwd()
test_dct = currpath + "/libcrossinheritanceDict"


class TestCROSSINHERITANCE:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.example01 = cppyy.load_reflection_info(cls.test_dct)

    @mark.xfail()
    def test01_override_function(self):
        """Test ability to override a simple function"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1().get_value() == 42

        class Derived(Base1):
            def get_value(self):
                return 13

        assert Derived().get_value() == 13

        assert 'Derived' in str(Derived())
        assert 'Derived' in repr(Derived())

        assert Base1.call_get_value(Base1())   == 42
        assert Base1.call_get_value(Derived()) == 13

    def test02_constructor(self):
        """Test constructor usage for derived classes"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1(27).get_value() == 27

        class Derived1(Base1):
            def __init__(self, pyval):
                Base1.__init__(self)
                self.m_pyint = pyval

            def get_value(self):
                return self.m_pyint+self.m_int

        d = Derived1(2)
        assert d.m_int   == 42
        assert d.m_pyint ==  2
        assert d.get_value()           == 44
        assert Base1.call_get_value(d) == 44

        class Derived2(Base1):
            def __init__(self, pyval, cppval):
                Base1.__init__(self, cppval)
                self.m_pyint = pyval

            def get_value(self):
                return self.m_pyint+self.m_int

        d = Derived2(2, 27)
        assert d.m_int   == 27
        assert d.m_pyint ==  2
        assert d.get_value()           == 29
        assert Base1.call_get_value(d) == 29

    def test03_override_function_abstract_base(self):
        """Test ability to override a simple function with an abstract base"""

        import cppyy
        CX = cppyy.gbl.CrossInheritance

        class C1PyBase2(CX.IBase2):
            def __init__(self):
                super(C1PyBase2, self).__init__()

            def get_value(self):
                return 99

        class C2PyBase2(CX.IBase2):
            def __init__(self):
                CX.IBase2.__init__(self)

            def get_value(self):
                return 91

        class C3PyBase2(CX.CBase2):
            def __init__(self):
                super(C3PyBase2, self).__init__()

        class C4PyBase2(CX.CBase2):
            def __init__(self):
                super(C4PyBase2, self).__init__()

            def get_value(self):
                return 13

        try:
            c2 = C2PyBase2()           # direct call to init can not work
            assert not "should have raised TypeError"
        except TypeError as e:
            assert "super" in str(e)   # clarifying message
            assert "abstract" in str(e)

        c1, c3, c4 = C1PyBase2(), C3PyBase2(), C4PyBase2()

        assert CX.IBase2.call_get_value(c1) == 99
        assert CX.IBase2.call_get_value(c3) == 42
        assert CX.IBase2.call_get_value(c4) == 13

        # now with abstract constructor that takes an argument
        class C4PyBase2(CX.IBase3):
            def __init__(self, intval):
                super(C4PyBase2, self).__init__(intval)

            def get_value(self):
                return 77

        c4 = C4PyBase2(88)
        assert c4.m_int == 88
        assert CX.IBase2.call_get_value(c4) == 77

    def test04_arguments(self):
        """Test ability to override functions that take arguments"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1(27).sum_value(-7) == 20

        class Derived1(Base1):
            def sum_value(self, val):
                return val + 13

        d1 = Derived1()
        assert d1.m_int   == 42
        assert d1.sum_value(-7)             == 6
        assert Base1.call_sum_value(d1, -7) == 6

        b1 = Base1()
        assert Base1.sum_pass_value(b1) == 6+2*b1.m_int

        class Derived2(Base1):
            def pass_value1(self, a):
                return a*2
            def pass_value2(self, a):
                return a.value*2
            def pass_value3(self, a):
                return a.value*2
            def pass_value4(self, b):
                return b.m_int*2
            def pass_value5(self, b):
                return b.m_int*2

        d2 = Derived2()
        assert Base1.sum_pass_value(d2) == 12+4*d2.m_int

    def test05_override_overloads(self):
        """Test ability to override overloaded functions"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1(27).sum_all(-7)     == 20
        assert Base1(27).sum_all(-3, -4) == 20

        class Derived(Base1):
            def sum_all(self, *args):
                return sum(args) + 13

        d = Derived()
        assert d.m_int   == 42
        assert d.sum_all(-7)             == 6
        assert Base1.call_sum_all(d, -7) == 6
        assert d.sum_all(-7, -5)             == 1
        assert Base1.call_sum_all(d, -7, -5) == 1

    def test06_const_methods(self):
        """Declared const methods should keep that qualifier"""

        import cppyy
        CX = cppyy.gbl.CrossInheritance

        class C1PyBase4(CX.IBase4):
            def __init__(self):
                super(C1PyBase4, self).__init__()

            def get_value(self):
                return 17

        class C2PyBase4(CX.CBase4):
            def __init__(self):
                super(C2PyBase4, self).__init__()

        c1, c2 = C1PyBase4(), C2PyBase4()

        assert CX.IBase4.call_get_value(c1) == 17
        assert CX.IBase4.call_get_value(c2) == 27

    def test07_templated_base(self):
        """Derive from a base class that is instantiated from a template"""

        import cppyy

        from cppyy.gbl.CrossInheritance import TBase1, TDerived1, TBase1_I

        class TPyDerived1(TBase1_I):
            def __init__(self):
                super(TBase1_I, self).__init__()

            def get_value(self):
                return 13

        b1, b2 = TBase1[int](), TBase1_I()
        assert b1.get_value() == 42
        assert b2.get_value() == 42

        d1 = TDerived1()
        assert d1.get_value() == 27

        p1 = TPyDerived1()
        assert p1.get_value() == 13

    @mark.xfail(run=False, condition=IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test08_error_handling(self):
        """Python errors should propagate through wrapper"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1(27).sum_value(-7) == 20

        errmsg = "I do not like the given value"
        class Derived(Base1):
            def sum_value(self, val):
                raise ValueError(errmsg)

        d = Derived()
        raises(ValueError, Base1.call_sum_value, d, -7)

        if IS_MAC_ARM:
            skip("JIT exceptions from signals not supported on Mac ARM")

        try:
            Base1.call_sum_value(d, -7)
            assert not "should not get here"
        except ValueError as e:
            assert errmsg in str(e)

        cppyy.cppdef("""\
        namespace CrossInheritance {
        std::string call_base1(Base1* b) {
            try {
                b->sum_value(-7);
            } catch (CPyCppyy::PyException& e) {
                e.clear();
                return e.what();
            }
            return "";
        } }""")

        res = cppyy.gbl.CrossInheritance.call_base1(d)

        assert 'ValueError' in res
        assert os.path.basename(__file__) in res

    @mark.xfail(run=False, condition=IS_MAC_ARM, reason = "Crashes on OS X ARM with" \
    "libc++abi: terminating due to uncaught exception")
    def test09_interface_checking(self):
        """Conversion errors should be Python exceptions"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1(27).sum_value(-7) == 20

        errmsg = "I do not like the given value"
        class Derived(Base1):
            def get_value(self):
                self.m_int*2       # missing return

        d = Derived(4)

        assert raises(TypeError, Base1.call_get_value, d)

    def test10_python_in_templates(self):
        """Usage of Python derived objects in std::vector"""

        import cppyy, gc

        CB = cppyy.gbl.CrossInheritance.CountableBase

        class PyCountable(CB):
            def call(self):
                try:
                    return self.extra + 42
                except AttributeError:
                    return 42

        start_count = CB.s_count

        v = cppyy.gbl.std.vector[PyCountable]()
        v.emplace_back(PyCountable())     # uses copy ctor
        assert len(v) == 1
        gc.collect()
        assert CB.s_count == 1 + start_count

        p = PyCountable()
        assert p.call() == 42
        p.extra = 42
        assert p.call() == 84
        v.emplace_back(p)
        assert len(v) == 2
        assert v[1].call() == 84          # copy ctor copies python state
        p.extra = 13
        assert p.call() == 55
        assert v[1].call() == 84
        del p
        gc.collect()
        assert CB.s_count == 2 + start_count

        v.push_back(PyCountable())        # uses copy ctor
        assert len(v) == 3
        gc.collect()
        assert CB.s_count == 3 + start_count

        del v
        gc.collect()
        assert CB.s_count == 0 + start_count

    def test11_python_in_make_shared(self):
        """Usage of Python derived objects with std::make_shared"""

        import cppyy

        cppyy.cppdef("""namespace MakeSharedTest {
        class Abstract {
        public:
          virtual ~Abstract() = 0;
          virtual int some_imp() = 0;
        };

        Abstract::~Abstract() {}

        int call_shared(std::shared_ptr<Abstract>& ptr) {
          return ptr->some_imp();
        } }""")

        from cppyy.gbl import std, MakeSharedTest
        from cppyy.gbl.MakeSharedTest import Abstract, call_shared

        class PyDerived(Abstract):
            def __init__(self, val):
                super(PyDerived, self).__init__()
                self.val = val
            def some_imp(self):
                return self.val

        v = std.make_shared[PyDerived](42)

        assert call_shared(v) == 42
        assert v.some_imp() == 42

        p = PyDerived(13)
        v = std.make_shared[PyDerived](p)
        assert call_shared(v) == 13
        assert v.some_imp() == 13

    def test12a_counter_test(self):
        """Test countable base counting"""

        import cppyy, gc

        std = cppyy.gbl.std
        CB  = cppyy.gbl.CrossInheritance.CountableBase

        class PyCountable(CB):
            def call(self):
                try:
                    return self.extra + 42
                except AttributeError:
                    return 42

        start_count = CB.s_count

      # test counter
        pyc = PyCountable()
        assert CB.s_count == 1 + start_count

        del pyc
        gc.collect()
        assert CB.s_count == 0 + start_count

    def test12_python_shared_ptr_memory(self):
        """Usage of Python derived objects with std::shared_ptr"""

        import cppyy, gc

        std = cppyy.gbl.std
        CB  = cppyy.gbl.CrossInheritance.CountableBase

        class PyCountable(CB):
            def call(self):
                try:
                    return self.extra + 42
                except AttributeError:
                    return 42

        start_count = CB.s_count

        v = std.vector[std.shared_ptr[PyCountable]]()
        v.push_back(std.make_shared[PyCountable]())

        gc.collect()
        assert CB.s_count == 1 + start_count

        del v
        gc.collect()
        assert CB.s_count == 0 + start_count

    def test13_virtual_dtors_and_del(self):
        """Usage of virtual destructors and Python-side del."""

        import cppyy, warnings

        cppyy.cppdef("""namespace VirtualDtor {
        class MyClass1 {};    // no virtual dtor ...

        class MyClass2 {
        public:
          virtual ~MyClass2() {}
        };

        class MyClass3 : public MyClass2 {};

        template<class T>
        class MyClass4 {
        public:
          virtual ~MyClass4() {}
        }; }""")

        VD = cppyy.gbl.VirtualDtor

      # rethought this: just issue a warning if there is no virtual destructor
      # as the C++ side now carries the type of the dispatcher, not the type of
      # the direct base class
        with warnings.catch_warnings(record=True) as w:
            class MyPyDerived1(VD.MyClass1):
                pass        # TODO: verify warning is given
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "has no virtual destructor" in str(w[-1].message)

            d = MyPyDerived1()
            del d             # used to crash

        class MyPyDerived2(VD.MyClass2):
            pass

        d = MyPyDerived2()
        del d                 # used to crash

      # check a few more derivations that should not fail
        class MyPyDerived3(VD.MyClass3):
            pass

        class MyPyDerived4(VD.MyClass4[int]):
            pass

    @mark.xfail()
    def test14_protected_access(self):
        """Derived classes should have access to protected members"""

        import cppyy

        ns = cppyy.gbl.AccessProtected

        assert not 'my_data' in ns.MyBase.__dict__
        assert not hasattr(ns.MyBase(), 'my_data')

        class MyPyDerived(ns.MyBase):
            pass

        assert 'my_data' in MyPyDerived.__dict__
        assert MyPyDerived().my_data == 101

        class MyPyDerived(ns.MyBase):
            def __init__(self):
                super(MyPyDerived, self).__init__()
                assert self.my_data == 101
                self.py_data = 13
                self.my_data = 42

        m = MyPyDerived()
        assert m.py_data      == 13
        assert m.my_data      == 42
        assert m.get_data()   == 42
        assert m.get_data_v() == 42

    def test15_object_returns(self):
        """Return of C++ objects from overridden functions"""

        import cppyy

      # Part 1: return of a new C++ object
        cppyy.cppdef("""namespace object_returns {
        class Base {
        public:
          virtual Base* foo() { return new Base(); }
          virtual ~Base() {}
          virtual std::string whoami() { return "Base"; }
        };

        class CppDerived : public Base {
          CppDerived* foo() { return new CppDerived(); }
          ~CppDerived() {}
          virtual std::string whoami() { return "CppDerived"; }
        };

        Base* call_foo(Base& obj) { return obj.foo(); } }""")

        ns = cppyy.gbl.object_returns

        class PyDerived1(ns.Base):
            def foo(self):
                return ns.CppDerived()

        obj = PyDerived1()
        assert not ns.call_foo(obj)

        class PyDerived2(ns.Base):
            def foo(self):
                x = ns.CppDerived()
                x.__python_owns__ = False
                return x

        obj = PyDerived2()
        assert not not ns.call_foo(obj)

      # Part 2: return of a new Python derived object
        class PyDerived3(ns.Base):
            def foo(self):
                return PyDerived3()

            def whoami(self):
                return "PyDerived3"

        obj = PyDerived3()
        newobj = ns.call_foo(obj)
        assert not newobj

        class PyDerived4(ns.Base):
            def foo(self):
                d = PyDerived4()
                d.__python_owns__ = False
                d.alpha = 2
                return d

            def whoami(self):
                return "PyDerived4"

        obj = PyDerived4()
        new_obj = ns.call_foo(obj)
        assert not not new_obj
        assert new_obj.whoami() == "PyDerived4"

    def test16_cctor_access_controlled(self):
        """Python derived class of C++ class with access controlled cctor"""

        import cppyy

        cppyy.cppdef("""namespace cctor_access_controlled {
        class CommonBase {
        public:
          virtual ~CommonBase() {}
          virtual std::string whoami() = 0;
        };

        class Base1 : public CommonBase {
          Base1(const Base1&) {}
        public:
          Base1() {}
          virtual ~Base1() {}
          virtual std::string whoami() { return "Base1"; }
        };

        class Base2 : public CommonBase {
        protected:
          Base2(const Base2&) {}
        public:
          Base2() {}
          virtual ~Base2() {}
          virtual std::string whoami() { return "Base2"; }
        };

        std::string callit(CommonBase& obj) { return obj.whoami(); } }""")

        ns = cppyy.gbl.cctor_access_controlled

        for base in (ns.Base1, ns.Base2):
            class PyDerived(base):
                def whoami(self):
                    return "PyDerived"

            obj = PyDerived()
            assert ns.callit(obj) == "PyDerived"

    def test17_deep_hierarchy(self):
        """Test a deep Python hierarchy with pure virtual functions"""

        import cppyy

        cppyy.cppdef("""namespace deep_hierarchy {
        class Base {
        public:
          virtual ~Base() {}
          virtual std::string whoami() = 0;
        };

        std::string callit(Base& obj) { return obj.whoami(); } }""")

        ns = cppyy.gbl.deep_hierarchy

        class PyDerived1(ns.Base):
            def whoami(self):
                return "PyDerived1"

        obj = PyDerived1()
        assert ns.callit(obj) == "PyDerived1"

        class PyDerived2(PyDerived1):
            pass

        obj = PyDerived2()
        assert obj.whoami()   == "PyDerived1"
        assert ns.callit(obj) == "PyDerived1"

        class PyDerived3(PyDerived1):
            def whoami(self):
                return "PyDerived3"

        obj = PyDerived3()
        assert obj.whoami()   == "PyDerived3"
        assert ns.callit(obj) == "PyDerived3"

        class PyDerived4(PyDerived2):
            def whoami(self):
                return "PyDerived4"

        obj = PyDerived4()
        assert obj.whoami()   == "PyDerived4"
        assert ns.callit(obj) == "PyDerived4"

    def test18_abstract_hierarchy(self):
        """Hierarchy with abstract classes"""


        import cppyy

        cppyy.cppdef("""namespace abstract_classes {
        class Base {
        public:
          virtual ~Base() {}
          virtual std::string whoami()  = 0;
          virtual std::string message() = 0;
        };

        std::string whois(Base& obj) { return obj.whoami(); }
        std::string saywot(Base& obj) { return obj.message(); } }""")

        ns = cppyy.gbl.abstract_classes

        class PyDerived1(ns.Base):
            def __init__(self):
                super(PyDerived1, self).__init__()
                self._name = "PyDerived1"

            def whoami(self):
                return self._name

        class PyDerived2(PyDerived1):
            def __init__(self):
                super(PyDerived2, self).__init__()
                self._message = "Hello, World!"

            def message(self):
                return self._message

        obj = PyDerived2()
        assert obj.whoami()  == "PyDerived1"
        assert ns.whois(obj) == "PyDerived1"

        assert obj.message()  == "Hello, World!"
        assert ns.saywot(obj) == "Hello, World!"

    def test19_cpp_side_multiple_inheritance(self):
        """Hierarchy with multiple inheritance on the C++ side"""

        import cppyy

        cppyy.cppdef(""" namespace cpp_side_multiple_inheritance {
        struct Result {
          Result() : result(1337) {}
          Result(int r) : result(r) {}
          int result;
        };

        class Base1 {
        public:
          virtual ~Base1() {}
          virtual Result abstract1() = 0;
        };

        class Base2 {
        public:
          virtual ~Base2() {}
          virtual Result abstract2() = 0;
        };

        class Base : public Base1, public Base2 {
        public:
          Result abstract2() override { return Result(999); }
        }; } """)

        ns = cppyy.gbl.cpp_side_multiple_inheritance

        class Derived(ns.Base):
            def abstract1(self):
                return ns.Result(1)

    @mark.skip
    def test20_basic_multiple_inheritance(self):
        """Basic multiple inheritance"""

        import cppyy

        cppyy.cppdef("""namespace basic_multiple_inheritance {
        class MyClass1 {
        public:
          MyClass1() : m_1(13) {}
          virtual ~MyClass1() {}
          virtual int x() = 0;

        public:
          int m_1;
        };
        int callx(MyClass1& m) { return m.x(); }

        class MyClass2 {
        public:
          MyClass2() : m_2(42) {}
          virtual ~MyClass2() {}
          virtual int y() = 0;

        public:
          int m_2;
        };
        int cally(MyClass2& m) { return m.y(); }

        class MyClass3 {
        public:
          MyClass3() : m_3(67) {}
         virtual ~MyClass3() {}
          virtual int z() = 0;

        public:
          int m_3;
        };
        int callz(MyClass3& m) { return m.z(); } }""")

        ns = cppyy.gbl.basic_multiple_inheritance

        class MyPyDerived(cppyy.multi(ns.MyClass1, ns.MyClass2)):
            def x(self):
                return 16

            def y(self):
                return 32

        assert len(MyPyDerived.__bases__) == 2

        a = MyPyDerived()
        assert a.x() == ns.callx(a)
        assert a.y() == ns.cally(a)

        assert a.m_1 == 13
        assert a.m_2 == 42

        class MyPyDerived2(cppyy.multi(ns.MyClass1, ns.MyClass2, ns.MyClass3)):
            def x(self):
                return 16

            def y(self):
                return 32

            def z(self):
                return 64

        assert len(MyPyDerived2.__bases__) == 3

        a = MyPyDerived2()
        assert a.x() == ns.callx(a)
        assert a.y() == ns.cally(a)
        assert a.z() == ns.callz(a)

        assert a.m_1 == 13
        assert a.m_2 == 42
        assert a.m_3 == 67

    @mark.skip()
    def test21_multiple_inheritance_with_constructors(self):
        """Multiple inheritance with constructors"""

        import cppyy

        cppyy.cppdef("""namespace multiple_inheritance_with_constructors {
        class MyClass1 {
        public:
          MyClass1() : m_1(13) {}
          MyClass1(int i) : m_1(i) {}
          virtual ~MyClass1() {}
          virtual int x() = 0;

        public:
          int m_1;
        };
        int callx(MyClass1& m) { return m.x(); }

        class MyClass2 {
        public:
          MyClass2() : m_2(42) {}
          MyClass2(int i) : m_2(i) {}
          virtual ~MyClass2() {}
          virtual int y() = 0;

        public:
          int m_2;
        };
        int cally(MyClass2& m) { return m.y(); }

        class MyClass3 {
        public:
          MyClass3() : m_3(67) {}
          MyClass3(int i) : m_3(i) {}
          virtual ~MyClass3() {}
          virtual int z() = 0;

        public:
          int m_3;
        };
        int callz(MyClass3& m) { return m.z(); } }""")

        ns = cppyy.gbl.multiple_inheritance_with_constructors

        class MyPyDerived(cppyy.multi(ns.MyClass1, ns.MyClass2)):
            def __init__(self, val1, val2):
                super(MyPyDerived, self).__init__((val1,), (val2,))

            def x(self):
                return 16

            def y(self):
                return 32

        assert len(MyPyDerived.__bases__) == 2

        a = MyPyDerived(27, 88)
        assert a.x() == ns.callx(a)
        assert a.y() == ns.cally(a)

        assert a.m_1 == 27
        assert a.m_2 == 88

        class MyPyDerived2(cppyy.multi(ns.MyClass1, ns.MyClass2, ns.MyClass3)):
            def __init__(self, val1, val2, val3):
                super(MyPyDerived2, self).__init__((val1,), (val2,), (val3,))

            def x(self):
                return 16

            def y(self):
                return 32

            def z(self):
                return 64

        assert len(MyPyDerived2.__bases__) == 3

        a = MyPyDerived2(27, 88, -11)
        assert a.x() == ns.callx(a)
        assert a.y() == ns.cally(a)
        assert a.z() == ns.callz(a)

        assert a.m_1 ==  27
        assert a.m_2 ==  88
        assert a.m_3 == -11

    @mark.skip()
    def test22_multiple_inheritance_with_defaults(self):
        """Multiple inheritance with defaults"""

        import cppyy

        cppyy.cppdef("""namespace multiple_inheritance_with_defaults {
        class MyClass1 {
        public:
          MyClass1(int i=13) : m_1(i) {}
          virtual ~MyClass1() {}
          virtual int x() = 0;

        public:
          int m_1;
        };
        int callx(MyClass1& m) { return m.x(); }

        class MyClass2 {
        public:
          MyClass2(int i=42) : m_2(i) {}
          virtual ~MyClass2() {}
          virtual int y() = 0;

        public:
            int m_2;
        };
        int cally(MyClass2& m) { return m.y(); }

        class MyClass3 {
        public:
            MyClass3(int i=67) : m_3(i) {}
            virtual ~MyClass3() {}
            virtual int z() = 0;

        public:
            int m_3;
        };
        int callz(MyClass3& m) { return m.z(); } }""")

        ns = cppyy.gbl.multiple_inheritance_with_defaults

        class MyPyDerived(cppyy.multi(ns.MyClass1, ns.MyClass2, ns.MyClass3)):
            def __init__(self, val1=None, val2=None, val3=None, nArgs=3):
                a1 = val1 is not None and (val1,) or ()
                a2 = val2 is not None and (val2,) or ()
                a3 = val3 is not None and (val3,) or ()
                if nArgs == 3:
                    super(MyPyDerived, self).__init__(a1, a2, a3)
                elif nArgs == 0:
                    super(MyPyDerived, self).__init__()
                elif nArgs == 1:
                    super(MyPyDerived, self).__init__(a1)
                elif nArgs == 2:
                    super(MyPyDerived, self).__init__(a1, a2)

            def x(self):
                return 16

            def y(self):
                return 32

            def z(self):
                return 64

        assert len(MyPyDerived.__bases__) == 3

        def verify(a, n1, n2, n3):
            assert a.m_1 == n1
            assert a.m_2 == n2
            assert a.m_3 == n3

        a = MyPyDerived(27, 88, -11)
        assert a.x() == ns.callx(a)
        assert a.y() == ns.cally(a)
        assert a.z() == ns.callz(a)

        verify(a, 27, 88, -11)

        a = MyPyDerived(val2=27)
        verify(a, 13, 27, 67)

        a = MyPyDerived(nArgs=0)
        verify(a, 13, 42, 67)

        a = MyPyDerived(27, nArgs=1)
        verify(a, 27, 42, 67)

        a = MyPyDerived(27, 55, nArgs=2)
        verify(a, 27, 55, 67)

    def test23_const_byvalue_return(self):
        """Const by-value return in overridden method"""

        import cppyy

        cppyy.cppdef("""namespace const_byvalue_return {
        struct Const {
            Const() = default;
            explicit Const(const std::string& s) { m_value = s; }
            std::string m_value;
        };

        struct Abstract {
          virtual ~Abstract() {}
          virtual const Const return_const() = 0;
        };

        const Const callit(Abstract* a) { return a->return_const(); } }""")

        ns = cppyy.gbl.const_byvalue_return

        class ReturnConstByValue(ns.Abstract):
             def return_const(self):
                 return ns.Const("abcdef")

        a = ReturnConstByValue()
        assert a.return_const().m_value == "abcdef"
        assert ns.callit(a).m_value     == "abcdef"

    @mark.skip()
    def test24_non_copyable(self):
        """Inheriting from a non-copyable base class"""

        import cppyy

        cppyy.cppdef("""\
        namespace non_copyable {
        struct Copyable {
            Copyable() = default;
            virtual ~Copyable() {}

            Copyable(const Copyable&) = default;
            Copyable& operator=(const Copyable&) = default;
        };

        struct Movable {
            Movable() = default;
            virtual ~Movable() {}

            Movable(const Movable&) = delete;
            Movable& operator=(const Movable&) = delete;
            Movable(Movable&&) = default;
            Movable& operator=(Movable&&) = default;
        };

        class NoCopyNoMove {
        public:
            NoCopyNoMove() = delete;
            NoCopyNoMove(const NoCopyNoMove&) = delete;
            NoCopyNoMove(NoCopyNoMove&&) = delete;
            NoCopyNoMove& operator=(const NoCopyNoMove&) = delete;
            NoCopyNoMove& operator=(NoCopyNoMove&&) = delete;
            virtual ~NoCopyNoMove() = default;

            template<typename DerivedType>
            explicit NoCopyNoMove(DerivedType* ptr) : fActual(ptr) {}

            std::string callme() {
                if (!fActual) return "failed!";
                return fActual->callme_imp();
            }

        private:
            virtual std::string callme_imp() = 0;

        protected:
            NoCopyNoMove* fActual;
        }; }""")

        ns = cppyy.gbl.non_copyable

        Copyable = ns.Copyable
        Movable  = ns.Movable
        NoCopyNoMove = ns.NoCopyNoMove

        class DerivedCopyable(Copyable):
            pass

      # used to fail with compilation error
        class DerivedMovable(Movable):
            pass

      # used to fail with compilation error
        class DerivedMulti(cppyy.multi(Movable, Copyable)):
            pass

     # used to fail with compilation error
        class DerivedNoCopyNoMove(NoCopyNoMove):
            def __init__(self):
                super(DerivedNoCopyNoMove, self).__init__(self)
              # TODO: chicken-and-egg situation here, 'this' from 'self' is
              # nullptr until the constructor has been called, so it can't
              # be passed as an argument to the same constructor
                self.fActual = self

            def callme_imp(self):
                return "Hello, World!"

        assert DerivedNoCopyNoMove().callme() == "Hello, World!"

    @mark.skip()
    def test25_default_ctor_and_multiple_inheritance(self):
        """Regression test: default ctor did not get added"""

        import cppyy

        cppyy.cppdef("""namespace default_ctor_and_multiple {
        struct Copyable {
          Copyable() = default;
          virtual ~Copyable() {}

          Copyable(const Copyable&) = default;
          Copyable& operator=(const Copyable&) = default;
        };

        struct Movable {
          Movable() = default;
          virtual ~Movable() {}

          Movable(const Movable&) = delete;
          Movable& operator=(const Movable&) = delete;
          Movable(Movable&&) = default;
          Movable& operator=(Movable&&) = default;
        };

        struct SomeClass {
          virtual ~SomeClass() {}
        }; }""")

        ns = cppyy.gbl.default_ctor_and_multiple
        Copyable  = ns.Copyable
        Movable   = ns.Movable
        SomeClass = ns.SomeClass

        class DerivedMulti(cppyy.multi(Movable, Copyable, SomeClass)):
            def __init__(self):
                super(DerivedMulti, self).__init__()

        d = DerivedMulti()
        assert d

    @mark.skip()
    def test26_no_default_ctor(self):
        """Make sure no default ctor is created if not viable"""

        import cppyy, warnings

        cppyy.cppdef("""namespace no_default_ctor {
        struct NoDefCtor1 {
          NoDefCtor1(int) {}
          virtual ~NoDefCtor1() {}
        };

        struct NoDefCtor2 {
          NoDefCtor2() = delete;
          virtual ~NoDefCtor2() {}
        };

        class NoDefCtor3 {
          NoDefCtor3() = default;
        public:
          virtual ~NoDefCtor3() {}
        };

        class Simple {}; }""")

        ns = cppyy.gbl.no_default_ctor

        for kls in (ns.NoDefCtor1, ns.NoDefCtor2, ns.NoDefCtor3):
            class PyDerived(kls):
                def __init__(self):
                    super(PyDerived, self).__init__()

            with raises(TypeError):
                PyDerived()

            with warnings.catch_warnings(record=True) as w:
                class PyDerived(cppyy.multi(kls, ns.Simple)):
                    def __init__(self):
                        super(PyDerived, self).__init__()

            with raises(TypeError):
                PyDerived()

            with warnings.catch_warnings(record=True) as w:
                class PyDerived(cppyy.multi(ns.Simple, kls)):
                    def __init__(self):
                        super(PyDerived, self).__init__()

            with raises(TypeError):
                PyDerived()

    def test27_interfaces(self):
        """Inherit from base with non-standard offset"""

        import cppyy

        cppyy.gbl.gInterpreter.Declare("""\
        namespace NonStandardOffset {
        struct Calc1 {
          virtual int calc1() = 0;
          virtual ~Calc1() = default;
        };

        struct Calc2 {
          virtual int calc2() = 0;
          virtual ~Calc2() = default;
        };

        struct Base : virtual public Calc1, virtual public Calc2 {
          Base() {}
        };

        struct Derived : Base, virtual public Calc2 {
          int calc1() override { return 1; }
          int calc2() override { return 2; }
        };

        int callback1(Calc1* c1) { return c1->calc1(); }
        int callback2(Calc2* c2) { return c2->calc2(); }
        }""")

        ns = cppyy.gbl.NonStandardOffset

        class MyPyDerived(ns.Derived):
            pass

        obj = MyPyDerived()

        assert obj.calc1()       == 1
        assert ns.callback1(obj) == 1

        assert obj.calc2()       == 2
        assert ns.callback2(obj) == 2

    def test28_cross_deep(self):
        """Deep inheritance hierarchy"""

        import cppyy

        cppyy.cppdef("""\
        namespace CrossDeep {

        class A {
        public:
            A(const std::string& /* name */) {}
            virtual ~A() {}
            virtual int fun1() const { return 0; }
            virtual int fun2() const { return fun1(); }
        }; }""")

        A = cppyy.gbl.CrossDeep.A

        class B(A):
            def __init__ (self, name = 'b'):
                super(B, self).__init__(name)

            def fun1(self):
                return  1

        class C(B):
            def fun1(self):
                return -1

        class D(B):
            pass

        for inst, val1 in [(A('a'), 0), (B('b'), 1), (C('c'), -1), (D('d'), 1)]:
            assert inst.fun1() == val1
            assert inst.fun2() == inst.fun1()

    @mark.skip()
    def test29_cross_deep_multi(self):
        """Deep multi-inheritance hierarchy"""

        import cppyy

        cppyy.cppdef("""\
        namespace CrossMultiDeep {

        class A {
        public:
            virtual ~A() {}
            virtual int calc_a() { return 17; }
        };

        int calc_a(A* a) { return a->calc_a(); }

        class B {
        public:
            virtual ~B() {}
            virtual int calc_b() { return 42; }
        };

        int calc_b(B* b) { return b->calc_b(); } }""")

        ns = cppyy.gbl.CrossMultiDeep

        class C(cppyy.multi(ns.A, ns.B)):
            def calc_a(self):
                return 18

            def calc_b(self):
                return 43

        c = C()
        assert ns.calc_a(c) == 18
        assert ns.calc_b(c) == 43

        class D(ns.B):
            def calc_b(self):
                return 44

        d = D()
        assert ns.calc_b(d) == 44

        class E(cppyy.multi(ns.A, D)):
            def calc_a(self):
                return 19

        e = E()
        assert ns.calc_a(e) == 19
        assert ns.calc_b(e) == 44

        class F(ns.A):
            def calc_a(self):
                return 20

        f = F()
        assert ns.calc_a(f) == 20

        class G(cppyy.multi(F, ns.B)):
            def calc_b(self):
                return 45

        g = G()
        assert ns.calc_a(g) == 20
        assert ns.calc_b(g) == 45

        class H(object):
            def calc_a(self):
                return 66

        class I(cppyy.multi(ns.A, H)):
            def calc_a(self):
                return 77

        i = I()
        assert ns.calc_a(i) == 77

        class J(cppyy.multi(H, ns.A)):
            def calc_a(self):
                return 88

        j = J()
        assert ns.calc_a(j) == 88


    def test30_access_and_overload(self):
        """Inheritance with access and overload complications"""

        import cppyy

        cppyy.cppdef("""\
        namespace AccessAndOverload {
        class Base {
        public:
            virtual ~Base() {}

        protected:
            virtual int  call1(int i) { return i; }
            virtual int  call1(int i, int j) { return i+j; }

            virtual void call2(int) { return; }
            virtual void call2(int, int) { return; }

            int call3(int i) { return i; }

        private:
            int call3(int i, int j) { return i+j; }
        }; }""")

        ns = cppyy.gbl.AccessAndOverload

      # used to produce uncompilable code
        class PyDerived(ns.Base):
            pass

    @mark.xfail()
    def test31_object_rebind(self):
        """Usage of bind_object to cast with Python derived objects"""

        import cppyy, gc

        ns = cppyy.gbl.CrossInheritance
        ns.build_component.__creates__ = True

        assert ns.Component.get_count() == 0

        cmp1 = ns.build_component(42)
        assert cmp1.__python_owns__
        assert type(cmp1) == ns.Component
        with raises(AttributeError):
            cmp1.getValue()

        assert ns.Component.get_count() == 1

      # introduce the actual component type; would have been a header,
      # but this simply has to match what is in crossinheritance.cxx
        cppyy.cppdef("""namespace CrossInheritance {
        class ComponentWithValue : public Component {
        public:
            ComponentWithValue(int value) : m_value(value) {}
            int getValue() { return m_value; }

        protected:
            int m_value;
        }; }""")

      # rebind cmp1 to its actual C++ class
        act_cmp1 = cppyy.bind_object(cmp1, ns.ComponentWithValue)
        assert not cmp1.__python_owns__          # b/c transferred
        assert act_cmp1.__python_owns__
        assert act_cmp1.getValue() == 42

        del act_cmp1, cmp1
        gc.collect()
        assert ns.Component.get_count() == 0

      # introduce a Python derived class
        ns.ComponentWithValue.__init__.__creates__ = False
        class PyComponentWithValue(ns.ComponentWithValue):
            def getValue(self):
                return self.m_value + 12

      # wipe the python-side connection
        pycmp2a = PyComponentWithValue(27)
        assert not pycmp2a.__python_owns__
        pycmp2a.__python_owns__ = True
        assert ns.Component.get_count() == 1

        pycmp2b = ns.cycle_component(pycmp2a)
        assert ns.Component.get_count() == 1
        assert pycmp2b is pycmp2a

        del pycmp2b, pycmp2a
        gc.collect()
        assert ns.Component.get_count() == 0

        cmp2 = cppyy.bind_object(cppyy.addressof(PyComponentWithValue(13)), ns.Component)
        assert ns.Component.get_count() == 1

        cmp2 = ns.cycle_component(cmp2)     # causes auto down-cast
        assert ns.Component.get_count() == 1
        #assert type(cmp2) != PyComponentWithValue

      # rebind cmp2 to the python type
        act_cmp2 = cppyy.bind_object(cmp2, PyComponentWithValue)
        act_cmp2.__python_owns__ = True
        assert act_cmp2.getValue() == 13+12

        del cmp2, act_cmp2
        gc.collect()
        assert ns.Component.get_count() == 0

      # introduce a Python derived class with initialization
        ns.ComponentWithValue.__init__.__creates__ = True
        class PyComponentWithInit(ns.ComponentWithValue):
            def __init__(self, cppvalue):
                super(PyComponentWithInit, self).__init__(cppvalue)
                self.m_pyvalue = 11

            def getValue(self):
                return self.m_value + self.m_pyvalue

        cmp3 = cppyy.bind_object(PyComponentWithInit(77), PyComponentWithInit)
        assert type(cmp3) == PyComponentWithInit
        assert ns.Component.get_count() == 1

        assert cmp3.getValue() == 77+11

        del cmp3
        gc.collect()
        assert ns.Component.get_count() == 0

        pyc = PyComponentWithInit(88)
        cmp4 = cppyy.bind_object(cppyy.addressof(pyc), ns.Component)
        assert type(cmp4) == ns.Component
        assert ns.Component.get_count() == 1

      # rebind cmp4 to the python type
        act_cmp4 = cppyy.bind_object(cmp4, PyComponentWithInit)
        assert act_cmp4.getValue() == 88+11

        del cmp4, act_cmp4, pyc
        gc.collect()
        assert ns.Component.get_count() == 0

        ns.ComponentWithValue.__init__.__creates__ = False
        cmp5 = cppyy.bind_object(cppyy.addressof(PyComponentWithInit(22)), ns.Component)
        cmp5.__python_owns__ = True
        assert type(cmp5) == ns.Component
        assert ns.Component.get_count() == 1

      # rebind cmp5 to the python type
        act_cmp5 = cppyy.bind_object(cmp5, PyComponentWithInit)
        assert not cmp5.__python_owns__
        assert act_cmp5.__python_owns__
        assert act_cmp5.getValue() == 22+11

        del cmp5, act_cmp5
        gc.collect()
        assert ns.Component.get_count() == 0

    def test32_by_value_arguments(self):
        """Override base function taking by-value arguments"""

        import cppyy

        cppyy.cppdef("""\
        namespace CrossCallWithValue {
        struct Data {
            int value;
        };

        struct CppBase {
            virtual ~CppBase() {}

            int func(Data d) {
                return d.value + extra_func(d);
            }

            virtual int extra_func(Data d) = 0;
        }; }""")

        ns = cppyy.gbl.CrossCallWithValue

        class PyDerived(ns.CppBase):
            def extra_func(self, d):
                return 42 + d.value

        d = ns.Data(13)
        p = PyDerived()

        assert p.func(d) == 42 + 2 * d.value

    @mark.xfail()
    def test33_direct_base_methods(self):
        """Call base class methods directly"""

        import cppyy

        cppyy.cppdef("""\
        namespace DirectCalls {
        struct A {
            virtual ~A() {}
            virtual int func() {
                return 1;
            }
        };

        struct B : public A {
            virtual int func() {
                return 2;
            }
        }; }""")

        ns = cppyy.gbl.DirectCalls

        a = ns.A()
        assert a.func()     == 1
        assert ns.A.func(a) == 1

        b = ns.B()
        assert b.func()     == 2
        assert ns.B.func(b) == 2
        assert ns.A.func(b) == 1

        with raises(TypeError):
            ns.B.func(a)

        class C(ns.A):
            def func(self):
                from_a = ns.A.func(self)
                return from_a + 2

        c = C()
        assert c.func() == 3

    def test34_no_ctors_in_base(self):
        """Base classes with no constructors"""

        import cppyy

        cppyy.cppdef("""\
        namespace BlockedCtors {
        class Z {
        public:
            virtual ~Z() {}
        };

        class X : Z {
            X();
            X(const X&&) = delete;
        };

        class Y: Z {
        protected:
            Y() {}
            Y(const Y&&) = delete;
        }; }""")

        ns = cppyy.gbl.BlockedCtors

        with raises(TypeError):
            ns.X()

        with raises(TypeError):
            ns.Y()

        class PyY1(ns.Y):
            pass

        with raises(TypeError):
            PyY1()

        class PyY2(ns.Y):
            def __init__(self):
                super(ns.Y, self).__init__()

        assert PyY2()

    def test35_deletion(self):
        """C++ side deletion should propagate to the Python side"""

        import cppyy

        cppyy.cppdef("""\
        namespace DeletionTest1 {
        class Base {
        public:
            virtual ~Base() {}
            void do_work() {}
        };

        void delete_it(Base *p) { delete p; }
        }""")

        ns = cppyy.gbl.DeletionTest1

        class Derived(ns.Base):
            was_deleted = False
            def __del__(self):
                Derived.was_deleted = True

        o1 = Derived()
        o1.do_work()
        ns.delete_it(o1)

        with raises(ReferenceError):
            o1.do_work()

        assert Derived.was_deleted == False
        del o1
        assert Derived.was_deleted == True

    def test36_custom_destruct(self):
        """C++ side deletion calls __destruct__"""

        import cppyy

        cppyy.cppdef("""\
        namespace DeletionTest2 {
        class Base {
        public:
            virtual ~Base() {}
            void do_work() {}
        };

        void delete_it(Base *p) { delete p; }
        }""")

        ns = cppyy.gbl.DeletionTest2

        class Derived(ns.Base):
            was_cpp_deleted = False
            was_py_deleted  = False

            def __destruct__(self):
                Derived.was_cpp_deleted = True

            def __del__(self):
                Derived.was_py_deleted  = True

        o1 = Derived()
        o1.do_work()
        ns.delete_it(o1)

        with raises(ReferenceError):
            o1.do_work()

        assert Derived.was_cpp_deleted == True
        assert Derived.was_py_deleted  == False
        del o1
        assert Derived.was_py_deleted  == True

    def test37_deep_tree(self):
        """Find overridable methods deep in the tree"""

        import cppyy

        cppyy.cppdef("""\
        namespace DeepTree {

        class Base {
        public:
            virtual ~Base() {}

            virtual std::string f1() { return "C++: Base::f1()"; }
            virtual std::string f2() { return "C++: Base::f2()"; }
            virtual std::string f3() { return "C++: Base::f3()"; }
        };

        class Intermediate: public Base {
        public:
            virtual ~Intermediate() {}

            using Base::f2;
        };

        class Sub: public Intermediate {
        public:
            virtual ~Sub() {}

            using Intermediate::f3; // `using Base::f3;` would also work
        };

        class CppSub: public Sub {
        public:
            virtual ~CppSub() {}

            std::string f1() { return "C++: CppSub::f1()"; }
            std::string f2() { return "C++: CppSub::f2()"; }
            std::string f3() { return "C++: CppSub::f3()"; }
        };

        std::string call_fs(Base *b) {
            std::string res;
            res += b->f1();
            res += b->f2();
            res += b->f3();
            return res;
        } }""")

        ns = cppyy.gbl.DeepTree

        cppsub = ns.CppSub()
        assert cppsub.f1() == "C++: CppSub::f1()"
        assert cppsub.f2() == "C++: CppSub::f2()"
        assert cppsub.f3() == "C++: CppSub::f3()"
        assert ns.call_fs(cppsub) == cppsub.f1() + cppsub.f2() + cppsub.f3()

        class PySub(ns.Sub):
            def f1(self):
                return "Python: PySub::f1()"

            def f2(self):
                return "Python: PySub::f2()"

            def f3(self):
                return "Python: PySub::f3()"

        pysub = PySub()
        assert pysub.f1() == "Python: PySub::f1()"
        assert pysub.f2() == "Python: PySub::f2()"
        assert pysub.f3() == "Python: PySub::f3()"
        assert ns.call_fs(pysub) == pysub.f1() + pysub.f2() + pysub.f3()

    @mark.xfail()
    def test38_protected_data(self):
        """Multiple cross inheritance with protected data"""

        import cppyy

        cppyy.cppdef("""
        namespace multiple_inheritance_with_protected_data {
        class MyBaseClass {
        public:
            virtual ~MyBaseClass() {}

        protected:
            int x = 0;
            std::string s = "Hello";
            int y = 0;
            std::string t = "World";
            int z = 0;
        public:
            MyBaseClass(int x, int y, int z) : x(x), y(y), z(z) {}
            int get_x() { return x; }
            int get_y() { return y; }
            int get_z() { return z; }
        }; }""")

        ns = cppyy.gbl.multiple_inheritance_with_protected_data

        class MyDerivedClass(ns.MyBaseClass):
            def __init__(self, x, y, z):
                super(MyDerivedClass, self).__init__(x, y, z)

        derived = MyDerivedClass(5, 7, 9)
        assert derived.get_x() == derived.x
        assert derived.get_y() == derived.y
        assert derived.get_z() == derived.z
        assert derived.s == "Hello"
        assert derived.t == "World"


if __name__ == "__main__":
    exit(pytest.main(args=['-ra', __file__]))
