import py, os, sys
from pytest import raises
from .support import setup_make, pylong

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("crossinheritanceDict"))

def setup_module(mod):
    setup_make("crossinheritance")


class TestCROSSINHERITANCE:
    def setup_class(cls):
        cls.test_dct = test_dct
        import cppyy
        cls.example01 = cppyy.load_reflection_info(cls.test_dct)

    def test01_override_function(self):
        """Test ability to override a simple function"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1().get_value() == 42

        class Derived(Base1):
            def get_value(self):
                return 13

        assert Derived().get_value() == 13

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

    def test07_const_methods(self):
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

    def test08_templated_base(self):
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

    def test09_error_handling(self):
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

        try:
            Base1.call_sum_value(d, -7)
            assert not "should not get here"
        except ValueError as e:
            assert errmsg in str(e)

    def test10_interface_checking(self):
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
