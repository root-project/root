# -*- coding: utf8 -*-`
# File: roottest/python/basic/PyROOT_datatypetests.py
# Author: Wim Lavrijsen (LBNL, WLavrijsen@lbl.gov)
# Created: 05/11/05
# Last: 04/20/15

"""Data type conversion unit tests for PyROOT package."""

import os, sys, unittest
sys.path.append(os.path.dirname( os.path.dirname(__file__)))

from common import *
from pytest import raises

# Compatibility notes: set_char() and set_uchar() raise a TypeError in PyROOT
# when handed a string argument, but a ValueError in cppyy. Further, in cppyy
# all object participate in memory regulation, but in PyROOT only TObject
# deriveds (which receive a recursive call).

PYTEST_MIGRATION = True

is_64bit = sys.maxsize > 2**32

def setup_module(mod):
    import sys, os
    if not os.path.exists('DataTypes.C'):
        os.chdir(os.path.dirname(__file__))
        err = os.system("make DataTypes_C")
        if err:
            raise OSError("'make' failed (see stderr)")


class TestClassDATATYPES:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "DataTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N
        # In new Cppyy, nullptr can't be found in gbl.
        cls.nullptr = cppyy.nullptr

    def test01_buffer_to_numpy(self):
        """Wrap buffer with NumPy array"""

        import numpy as np
        import cppyy
        c = cppyy.gbl.CppyyTestData()
        N = cppyy.gbl.N

        arr = c.get_double_array()
        np_arr = np.frombuffer(arr, 'f8', N)
        assert len(np_arr) == N

        val = 1.0
        arr[N-1] = val
        assert arr[N-1] == np_arr[N-1] == val

    # NOTE: former test02_boolarray2cpp (ROOT-10731) and
    # test03_arraydatamember_lifeline (#7501) were removed: they are covered
    # upstream by cppyy's test_lowlevel.py::test09_numpy_bool_array and
    # test_datatypes.py::test44_buffer_memory_handling respectively.


## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
