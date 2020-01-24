# File: roottest/python/pythonizations/PyROOT_smartptrtest.py
# Author: Toby St. Clere-Smithe (mail@tsmithe.net)
# Modified: Wim Lavrijsen (WLavrijsen@lbl.gov)
# Created: 08/29/15
# Last: 08/29/15

"""Smart pointer tests for cppyy."""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common import *
from pytest import raises


def setup_module(mod):
    if not os.path.exists('SmartPtr.C'):
        os.chdir(os.path.dirname(__file__))
        err = os.system("make SmartPtr_C")
        if err:
            raise OSError("'make' failed (see stderr)")


class TestClassSMARTPTRS:
    def setup_class(cls):
        # we need to import ROOT because in macOS if a cppyy env
        # variable is not set libcppyy_backend cannot be found
        import ROOT
        import cppyy
        cls.test_dct = "SmartPtr_C"
        cls.smartptr = cppyy.load_reflection_info(cls.test_dct)

        # We need to introduce it in order to distinguish between
        # _get_smart_ptr in old Cppyy and __smartptr__ in new Cppyy
        cls.exp_pyroot = os.environ.get('EXP_PYROOT') == 'True'

    def test01_transparency(self):
        import cppyy

        MyShareable = cppyy.gbl.MyShareable
        mine = cppyy.gbl.mine

        assert type(mine) == MyShareable
        if self.exp_pyroot:
            assert type(mine.__smartptr__()) == cppyy.gbl.std.shared_ptr(MyShareable)
        else:
            assert type(mine._get_smart_ptr()) == cppyy.gbl.std.shared_ptr(MyShareable)

        assert mine.say_hi() == "Hi!"

    def test02_converters(self):
        import cppyy

        mine = cppyy.gbl.mine

        cppyy.gbl.pass_mine_rp_ptr(mine)
        cppyy.gbl.pass_mine_rp_ref(mine)
        cppyy.gbl.pass_mine_rp(mine)

        cppyy.gbl.pass_mine_sp_ptr(mine)
        cppyy.gbl.pass_mine_sp_ref(mine)

        if self.exp_pyroot:
            cppyy.gbl.pass_mine_sp_ptr(mine.__smartptr__())
            cppyy.gbl.pass_mine_sp_ref(mine.__smartptr__())
        else:
            cppyy.gbl.pass_mine_sp_ptr(mine._get_smart_ptr())
            cppyy.gbl.pass_mine_sp_ref(mine._get_smart_ptr())

        cppyy.gbl.pass_mine_sp(mine)
        if self.exp_pyroot:
            cppyy.gbl.pass_mine_sp(mine.__smartptr__())
        else:
            cppyy.gbl.pass_mine_sp(mine._get_smart_ptr())

        # TODO:
        # cppyy.gbl.mine = mine
        cppyy.gbl.renew_mine()

    def test03_executors(self):
        import cppyy

        MyShareable = cppyy.gbl.MyShareable

        mine = cppyy.gbl.gime_mine_ptr()
        assert type(mine) == MyShareable
        if self.exp_pyroot:
            assert type(mine.__smartptr__()) == cppyy.gbl.std.shared_ptr(MyShareable)
        else:
            assert type(mine._get_smart_ptr()) == cppyy.gbl.std.shared_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"

        mine = cppyy.gbl.gime_mine_ref()
        assert type(mine) == MyShareable
        if self.exp_pyroot:
            assert type(mine.__smartptr__()) == cppyy.gbl.std.shared_ptr(MyShareable)
        else:
            assert type(mine._get_smart_ptr()) == cppyy.gbl.std.shared_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"

        mine = cppyy.gbl.gime_mine()
        assert type(mine) == MyShareable
        if self.exp_pyroot:
            assert type(mine.__smartptr__()) == cppyy.gbl.std.shared_ptr(MyShareable)
        else:
            assert type(mine._get_smart_ptr()) == cppyy.gbl.std.shared_ptr(MyShareable)
        assert mine.say_hi() == "Hi!"

    def test04_reset(self):
        # ROOT-10245
        import ROOT

        ROOT.gROOT.ProcessLine('std::shared_ptr<TObject> optr(new TObject());')
        o2 = ROOT.TObject()
        ROOT.optr.__smartptr__().reset(o2)
        assert ROOT.optr == o2


## actual test run
if __name__ == '__main__':
    result = run_pytest(__file__)
    sys.exit(result)
