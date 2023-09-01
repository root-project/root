import unittest

import ROOT


class TestRooArgList(unittest.TestCase):
    """
    Test for RooArgList pythonizations.
    """

    def test_conversion_from_python_collection(self):

        # General check that the conversion from string or tuple works, using
        # constants to get compact test code.
        l1 = ROOT.RooArgList([1.0, 2.0, 3.0])
        self.assertEqual(len(l1), 3)

        l2 = ROOT.RooArgList((1.0, 2.0, 3.0))
        self.assertEqual(len(l2), 3)

        # Let's make sure that we can add two arguments with the same name to
        # the RooArgList. Here, we try to add the same RooConst two times. The
        # motivation for this test if the RooArgList is created via an
        # intermediate RooArgSet, which should not happen.
        l3 = ROOT.RooArgList([1.0, 2.0, 2.0])
        self.assertEqual(len(l3), 3)


if __name__ == "__main__":
    unittest.main()
