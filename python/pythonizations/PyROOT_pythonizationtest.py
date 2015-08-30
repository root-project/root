# File: roottest/python/pythonizations/PyROOT_pythonizationstests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 07/03/15
# Last: 07/03/15

"""Pythonization tests for PyROOT package."""

import os, sys
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from common import *
from pytest import raises


def setup_module(mod):
    import sys, os
    sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
    err = os.system("make Pythonizables_C")
    if err:
        raise OSError("'make' failed (see stderr)")


class TestClassPYTHONIZATIONS:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "Pythonizables_C"
        cls.pythonizables = cppyy.load_reflection_info(cls.test_dct)

    def test01_size_mapping(self):
        """Use composites to map GetSize() onto buffer returns"""

        import cppyy

        def set_size(self, buf):
            buf.SetSize(self.GetN())
            return buf

        cppyy.add_pythonization(
            cppyy.compose_method('MyBufferReturner$', 'Get[XY]$', set_size))

        m = cppyy.gbl.MyBufferReturner


class TestClassPYTHONIZATIONS_FRAGILITY:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "Pythonizables_C"
        cls.pythonizables = cppyy.load_reflection_info(cls.test_dct)



class TestClassROOT_PYTHONIZATIONS:
    def test01_tgraph(self):
        """TGraph has GetN() mapped as size to its various buffer returns"""

        import ROOT


## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
