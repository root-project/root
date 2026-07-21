# File: roottest/python/pythonizations/PyROOT_pythonizationstests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 07/03/15
# Last: 07/03/15

"""Pythonization tests for PyROOT package.

NOTE: the former TestClassPYTHONIZATIONS (size mapping, type pinning) was
removed: it exists verbatim upstream in the cppyy test suite
(bindings/pyroot/cppyy/cppyy/test/test_pythonization.py, with identical helper
classes in pythonizables.h). The former TGraph and TH2 tests were removed as
well: they are covered by bindings/pyroot/pythonizations/test/tgraph_getters.py
and th2.py.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import *
from pytest import raises


class TestClassROOT_PYTHONIZATIONS:

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

## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
