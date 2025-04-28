# File: roottest/python/basic/PyROOT_overloadtests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 04/15/05
# Last: 05/05/15

"""Overload unit tests for PyROOT package."""

import os, sys
sys.path.append(os.path.dirname( os.path.dirname(__file__)))

from common import *
from pytest import raises

# Compatibility notes: __dispatch__ still requires formal arguments.

PYTEST_MIGRATION = True

def setup_module(mod):
    import sys, os
    if not os.path.exists('Overloads_C.so'):
        os.chdir(os.path.dirname(__file__))
        sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
        err = os.system("make Overloads_C")
        if err:
            raise OSError("'make' failed (see stderr)")

    check_cppyy_backend()


class TestClassOVERLOADS:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "Overloads_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)

    def test01_class_based_overloads(self):
        """Functions overloaded on different C++ class arguments"""

        import cppyy
        OverloadA = cppyy.gbl.OverloadA
        OverloadB = cppyy.gbl.OverloadB
        OverloadC = cppyy.gbl.OverloadC
        OverloadD = cppyy.gbl.OverloadD

        NamespaceA = cppyy.gbl.NamespaceA
        NamespaceB = cppyy.gbl.NamespaceB

        assert OverloadC().get_int(OverloadA()) == 42
        assert OverloadC().get_int(OverloadB()) == 13
        assert OverloadD().get_int(OverloadA()) == 42
        assert OverloadD().get_int(OverloadB()) == 13

        assert OverloadC().get_int(NamespaceA.OverloadA()) ==  88
        assert OverloadC().get_int(NamespaceB.OverloadA()) == -33

        assert OverloadD().get_int(NamespaceA.OverloadA()) ==  88
        assert OverloadD().get_int(NamespaceB.OverloadA()) == -33

    def test02_class_based_overloads_explicit_resolution(self):
        """Explicitly resolved function overloads"""

        import cppyy
        OverloadA = cppyy.gbl.OverloadA
        OverloadB = cppyy.gbl.OverloadB
        OverloadC = cppyy.gbl.OverloadC
        OverloadD = cppyy.gbl.OverloadD

        NamespaceA = cppyy.gbl.NamespaceA

        c = OverloadC()
        raises(TypeError, c.__dispatch__, 'get_int', 12)
        raises(LookupError, c.__dispatch__, 'get_int', 'does_not_exist')
        assert c.__dispatch__('get_int', 'OverloadA* a')(OverloadA()) == 42
        assert c.__dispatch__('get_int', 'OverloadB* b')(OverloadB()) == 13

        assert OverloadC().__dispatch__('get_int', 'OverloadA* a')(OverloadA())  == 42
        # TODO: #assert c_overload.__dispatch__('get_int', 'OverloadB* b')(c, OverloadB()) == 13

        d = OverloadD()
        assert d.__dispatch__('get_int', 'OverloadA* a')(OverloadA()) == 42
        assert d.__dispatch__('get_int', 'OverloadB* b')(OverloadB()) == 13

        nb = NamespaceA.OverloadB()
        raises(TypeError, nb.f, OverloadC())

    def test03_fragile_class_based_overloads(self):
        """Test functions overloaded on void* and non-existing classes"""

        import cppyy
        MoreOverloads = cppyy.gbl.MoreOverloads

        # ROOT-specific ignores --
        import ROOT
        oldval = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kError

        OlAA = cppyy.gbl.OlAA
        OlBB = cppyy.gbl.OlBB
        OlCC = cppyy.gbl.OlCC
        OlDD = cppyy.gbl.OlDD

        from cppyy.gbl import get_OlBB, get_OlDD

        # first verify that BB and DD are indeed unknown
        raises(TypeError, OlBB)
        raises(TypeError, OlDD)

        # then try overloads based on them
        assert MoreOverloads().call(OlAA())     == "OlAA"
        # New Cppyy calls the same overload as C++: (const OlBB&, void*)
        get_olbb_res = "OlBB"
        assert MoreOverloads().call(get_OlBB()) == get_olbb_res
        assert MoreOverloads().call(OlCC())     == "OlCC"
        assert MoreOverloads().call(get_OlDD()) == "OlDD"   # <- has an unknown

        # -- ROOT-specific ignores
        ROOT.gErrorIgnoreLevel = oldval

    def test04_fully_fragile_overloads(self):
        """An unknown* is preferred over unknown&"""

        import cppyy
        MoreOverloads2 = cppyy.gbl.MoreOverloads2

        # ROOT-specific ignores --
        import ROOT
        oldval = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kError

        from cppyy.gbl import get_OlBB, get_OlDD

        assert MoreOverloads2().call(get_OlBB())    == "OlBB*"
        assert MoreOverloads2().call(get_OlDD(), 1) == "OlDD*"

        # -- ROOT-specific ignores
        ROOT.gErrorIgnoreLevel = oldval

    def test05_array_overloads(self):
        """Functions overloaded on different arrays"""

        import cppyy
        OverloadC = cppyy.gbl.OverloadC
        OverloadD = cppyy.gbl.OverloadD

        from array import array

        ai = array('i', [525252])
        assert OverloadC().get_int(ai) == 525252
        assert OverloadD().get_int(ai) == 525252

        ah = array('h', [25])
        assert OverloadC().get_int(ah) == 25
        assert OverloadD().get_int(ah) == 25

    def test06_double_int_overloads(self):
        """Overloads on int/doubles"""

        import cppyy
        MoreOverloads = cppyy.gbl.MoreOverloads

        assert MoreOverloads().call(1)   == "int"
        assert MoreOverloads().call(1.)  == "double"
        assert MoreOverloads().call1(1)  == "int"
        assert MoreOverloads().call1(1.) == "double"

    def test07_mean_overloads(self):
        """Adapted test for array overloading"""

        import cppyy, array
        cmean = cppyy.gbl.calc_mean

        numbers = [8, 2, 4, 2, 4, 2, 4, 4, 1, 5, 6, 3, 7]
        mean, median = 4.0, 4.0

        for l in ['f', 'd', 'i', 'h', 'l']:
            a = array.array(l, numbers)
            assert round(cmean(len(a), a) - mean, 8) == 0

    def test08_templated_mean_overloads(self):
        """Adapted test for array overloading with templates"""

        import cppyy, array
        cmean = cppyy.gbl.calc_mean_templ

        numbers = [8, 2, 4, 2, 4, 2, 4, 4, 1, 5, 6, 3, 7]
        mean, median = 4.0, 4.0

        for l in ['f', 'd', 'i', 'h', 'l']:
            a = array.array(l, numbers)
            assert round(cmean(len(a), a) - mean, 8) == 0

    def test09_more_builtin_overloads(self):
        """Overloads on bool/int/double"""

        import cppyy
        d = cppyy.gbl.MoreBuiltinOverloads()

        assert d.method(0.0)    == "double"
        assert d.method(0.1234) == "double"
        assert d.method(1234)   == "int"
        assert d.method(-1234)  == "int"
        assert d.method(True)   == "bool"
        assert d.method(False)  == "bool"

        # allow bool -> int
        assert d.method2(True)  == "int"

        # allow int(0) and int(1) -> bool (even over int)
        assert d.method3(0)     == "bool"
        assert d.method3(1)     == "bool"

        # do not allow 0.0 -> bool
        raises(ValueError, d.method3, 0.0)

        # do not allow 0.0 -> char
        assert d.method4(1, 1.) == "double"
        assert d.method4(1, 1 ) == "char"

        assert cppyy.gbl.global_builtin_overload(1, 1 ) == "char"
        assert cppyy.gbl.global_builtin_overload(1, 1.) == "double"

    def test10_tmath_overloads(self):
        """Overloads on small integers with TMath"""

        import cppyy

      # fails if short int selected for overload
        assert cppyy.gbl.TMath.Abs(104125) == 104125

    def test11_free_function_namespace(self):
        """Free functions on a namespace"""
        import cppyy
        cppyy.gbl.gInterpreter.Declare("""
          namespace Gaudi { 
            namespace Utils { 
              namespace Histos { 
                int histoDump() { return 42; }
              }
            }
          }""")
        assert cppyy.gbl.Gaudi.Utils.Histos.histoDump() == 42

## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
