import unittest

import ROOT

# Necessary inside the "eval" call
RooArgSet = ROOT.RooArgSet
RooCmdArg = ROOT.RooCmdArg

x = ROOT.RooRealVar("x", "x", 1.0)
y = ROOT.RooRealVar("y", "y", 2.0)
z = ROOT.RooRealVar("z", "z", 3.0)


def args_equal(arg_1, arg_2):
    same = True

    same &= str(arg_1.GetName()) == str(arg_2.GetName())
    same &= str(arg_1.GetTitle()) == str(arg_2.GetTitle())

    for i in range(2):
        same &= arg_1.getInt(i) == arg_2.getInt(i)

    for i in range(2):
        same &= arg_1.getDouble(i) == arg_2.getDouble(i)

    for i in range(3):
        same &= str(arg_1.getString(i)) == str(arg_2.getString(i))

    same &= arg_1.procSubArgs() == arg_2.procSubArgs()
    same &= arg_1.prefixSubArgs() == arg_2.prefixSubArgs()

    for i in range(2):
        same &= arg_1.getObject(i) == arg_2.getObject(i)

    def set_equal(set_1, set_2):
        if set_1 == ROOT.nullptr and set_2 == ROOT.nullptr:
            return True
        if set_1 == ROOT.nullptr and set_2 != ROOT.nullptr:
            return False
        if set_1 != ROOT.nullptr and set_2 == ROOT.nullptr:
            return False

        if set_1.size() != set_2.size():
            return False

        return set_2.hasSameLayout(set_1)

    for i in range(2):
        same &= set_equal(arg_1.getSet(i), arg_2.getSet(i))

    return same


class TestRooArgList(unittest.TestCase):
    """
    Test for RooCmdArg pythonizations.
    """

    def test_constructor_eval(self):

        set_1 = ROOT.RooArgSet(x, y)
        set_2 = ROOT.RooArgSet(y, z)

        def do_test(*args):
            arg_1 = ROOT.RooCmdArg(*args)

            # The arg should be able to recreate itself by emitting the right
            # constructor code:
            arg_2 = eval(arg_1.constructorCode())

            self.assertTrue(args_equal(arg_1, arg_2))

        nullp = ROOT.nullptr

        # only fill the non-object fields:
        do_test("Test", -1, 3, 4.2, 4.7, "hello", "world", nullp, nullp, nullp, "s3", nullp, nullp)

        # RooArgSet tests:
        do_test("Test", -1, 3, 4.2, 4.7, "hello", "world", nullp, nullp, nullp, "s3", set_1, set_2)


if __name__ == "__main__":
    unittest.main()
