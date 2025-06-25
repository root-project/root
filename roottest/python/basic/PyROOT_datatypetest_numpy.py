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

    check_cppyy_backend()


class TestClassDATATYPES:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "DataTypes_C"
        cls.datatypes = cppyy.load_reflection_info(cls.test_dct)
        cls.N = cppyy.gbl.N
        # In new Cppyy, nullptr can't be found in gbl.
        # Take it from libcppyy (we could also use ROOT.nullptr)
        import libcppyy
        cls.nullptr = libcppyy.nullptr

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

    def test02_boolarray2cpp(self):
        '''
        Pass an bool array to a C++ function taking a bool*
        Fixes ROOT-10731
        '''
        import numpy as np
        import cppyy
        cppyy.cppdef('int convert(bool* x) { return x[0]; }')
        x1 = np.array([True], '?') # bool
        x2 = np.array([True], 'b') # signed char, treated as bool before
        y1 = cppyy.gbl.convert(x1)
        y2 = cppyy.gbl.convert(x2)
        assert y1 == 1
        assert y2 == 1

    def test03_arraydatamember_lifeline(self):
        """Test setting of lifeline for array data members"""
        # 7501

        import numpy as np
        import cppyy
        cppyy.cppdef("""
        class array_ll {
        public:
            float *v1 = nullptr;
            float *v2 = nullptr;
        };
        """)
        a = cppyy.gbl.array_ll()
        a.v1 = np.array([1, 2], dtype=np.float32)
        a.v2 = np.array([3, 4], dtype=np.float32)

        assert a.v1[0] == 1
        assert a.v1[1] == 2


## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
