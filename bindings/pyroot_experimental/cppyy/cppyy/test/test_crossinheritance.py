import py, os, sys
from pytest import raises
from .support import setup_make, pylong

currpath = py.path.local(__file__).dirpath()
test_dct = str(currpath.join("crossinheritanceDict.so"))

def setup_module(mod):
    setup_make("crossinheritanceDict.so")


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

    def test03_arguments(self):
        """Test ability to override functions that take arguments"""

        import cppyy
        Base1 = cppyy.gbl.CrossInheritance.Base1

        assert Base1(27).sum_value(-7) == 20

        class Derived(Base1):
            def sum_value(self, val):
                return val + 13

        d = Derived()
        assert d.m_int   == 42
        assert d.sum_value(-7)             == 6
        assert Base1.call_sum_value(d, -7) == 6

    def test04_override_overloads(self):
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

    def test05_error_handling(self):
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
