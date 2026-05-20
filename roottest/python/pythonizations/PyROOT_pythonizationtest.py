# File: roottest/python/pythonizations/PyROOT_pythonizationstests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 07/03/15
# Last: 07/03/15

"""Pythonization tests for PyROOT package."""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import *
from pytest import raises


def setup_module(mod):
    if not os.path.exists('Pythonizables.C'):
        os.chdir(os.path.dirname(__file__))
        err = os.system("make Pythonizables_C")
        if err:
            raise OSError("'make' failed (see stderr)")


class TestClassPYTHONIZATIONS:
    def setup_class(cls):
        # we need to import ROOT because in macOS if a cppyy env
        # variable is not set libcppyy_backend cannot be found
        import ROOT
        import cppyy
        cls.test_dct = "Pythonizables_C"
        cls.pythonizables = cppyy.load_reflection_info(cls.test_dct)

    def test01_size_mapping(self):
        """Use composites to map GetSize() onto buffer returns"""

        import cppyy

        def set_size(self, buf):
            buf.reshape((self.GetN(),))
            return buf

        cppyy.py.add_pythonization(
            cppyy.py.compose_method("pythonizables::MyBufferReturner$", "Get[XY]$", set_size)
            )

        bsize, xval, yval = 3, 2, 5
        m = cppyy.gbl.pythonizables.MyBufferReturner(bsize, xval, yval)

        x = m.GetX()
        assert len(x) == bsize
        assert list(x) == list(map(lambda x: x*xval, range(bsize)))

        y = m.GetY()
        assert len(y) == bsize
        assert list(y) == list(map(lambda x: x*yval, range(bsize)))

    def test02_type_pinning(self):
        """Verify pinnability of returns"""

        import cppyy

        cppyy.gbl.pythonizables.GimeDerived.__creates__ = True

        result = cppyy.gbl.pythonizables.GimeDerived()
        assert type(result) == cppyy.gbl.pythonizables.MyDerived

        cppyy.py.pin_type(cppyy.gbl.pythonizables.MyBase)
        assert type(result) == cppyy.gbl.pythonizables.MyDerived


class TestClassPYTHONIZATIONS_FRAGILITY:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "Pythonizables_C"
        cls.pythonizables = cppyy.load_reflection_info(cls.test_dct)



class TestClassROOT_PYTHONIZATIONS:

    def test01_tgraph(self):
        """TGraph has GetN() mapped as size to its various buffer returns"""

        import ROOT, array

        N = 5; xval, yval = 3, 7
        ax = array.array('d', map(lambda x: x*xval, range(N)))
        ay = array.array('d', map(lambda x: x*yval, range(N)))

        g = ROOT.TGraph(N, ax, ay)

        x = g.GetX()
        assert len(x) == N
        assert list(x) == list(ax)

        y = g.GetY()
        assert len(y) == N
        assert list(y) == list(ay)

    def test02_tfile(self):
        """TFile life time control of objects read from file: ensure cache"""

        import ROOT

        ROOT.gEnv.SetValue("RooFit.Banner", "0")
        try:
            ws = ROOT.RooWorkspace('w')
        except AttributeError as e:
            if 'RooWorkspace' in str(e):
                return            # no roofit enabled, just declare success
            raise
        ws.factory("x[0, 10]")
        ws.writeToFile("foo.root")
        del ws

        f = ROOT.TFile.Open("foo.root")
        ws = f.Get("w")
        assert ws.var("x").getVal() == 5
        ws.var("x").setVal(6)
        assert ws.var("x").getVal() == 6   # uncached would give 5

    def test03_th2(self):
        """GetBinErrorUp and GetBinErrorLow overloads obtained with using decls"""

        # ROOT-10109
        import ROOT
        h = ROOT.TH2D()
        assert h.GetBinErrorUp(1)  == 0.
        assert h.GetBinErrorLow(1) == 0.


## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
