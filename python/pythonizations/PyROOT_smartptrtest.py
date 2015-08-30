# File: roottest/python/pythonizations/PyROOT_smartptrtest.py
# Author: Toby St. Clere-Smithe (mail@tsmithe.net)
# Modified: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 08/29/15
# Last: 08/29/15

"""Smart pointer tests for cppyy."""

import os, sys
sys.path.append( os.path.join( os.getcwd(), os.pardir ) )

from common import *
from pytest import raises


def setup_module(mod):
    import sys, os
    sys.path.append( os.path.join( os.getcwd(), os.pardir ) )
    err = os.system("make SmartPtr_C")
    if err:
        raise OSError("'make' failed (see stderr)")


class TestClassSMARTPTRS:
    def setup_class(cls):
        import cppyy
        cls.test_dct = "SmartPtr_C"
        cls.smartptr = cppyy.load_reflection_info(cls.test_dct)

    def test01_transparency(self):
        import cppyy

        MyShareable = cppyy.gbl.MyShareable
        mine = cppyy.gbl.mine

        assert type(mine) == MyShareable
        assert type(mine._get_smart_ptr()) == cppyy.gbl.std.auto_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"

    def test02_converters(self):
        import cppyy

        mine = cppyy.gbl.mine

        cppyy.gbl.pass_mine_rp_ptr(mine)
        cppyy.gbl.pass_mine_rp_ref(mine)
        cppyy.gbl.pass_mine_rp(mine)

        cppyy.gbl.pass_mine_sp_ptr(mine)
        cppyy.gbl.pass_mine_sp_ref(mine)

        cppyy.gbl.pass_mine_sp_ptr(mine._get_smart_ptr())
        cppyy.gbl.pass_mine_sp_ref(mine._get_smart_ptr())

        cppyy.gbl.pass_mine_sp(mine)
        cppyy.gbl.pass_mine_sp(mine._get_smart_ptr())

        # TODO:
        # cppyy.gbl.mine = mine
        cppyy.gbl.renew_mine()

    def test03_executors(self):
        import cppyy
        
        MyShareable = cppyy.gbl.MyShareable

        mine = cppyy.gbl.gime_mine_ptr()
        assert type(mine) == MyShareable
        assert type(mine._get_smart_ptr()) == cppyy.gbl.std.auto_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"

        mine = cppyy.gbl.gime_mine_ref()
        assert type(mine) == MyShareable
        assert type(mine._get_smart_ptr()) == cppyy.gbl.std.auto_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"

        mine = cppyy.gbl.gime_mine()
        assert type(mine) == MyShareable
        assert type(mine._get_smart_ptr()) == cppyy.gbl.std.auto_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"


## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
