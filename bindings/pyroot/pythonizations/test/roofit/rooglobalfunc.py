import unittest

import ROOT


class TestRooGlobalFunc(unittest.TestCase):
    """
    Test for RooGlobalFunc pythonizations.
    """

    def test_color_codes(self):
        """Test that the color code pythonizations in the functions like
        RooFit.LineColor are working as they should.
        """

        def code(color):
            """Get the color code that will be obtained by a given argument
            passed to RooFit.LineColor.
            """
            return ROOT.RooFit.LineColor(color).getInt(0)

        # Check that general string to enum pythonization works
        self.assertEqual(code(ROOT.kRed), code("kRed"))

        # Check that matplotlib-style color strings work
        self.assertEqual(code(ROOT.kRed), code("r"))

        # Check that postfix operations applied to ROOT color codes work
        self.assertEqual(code(ROOT.kRed+1), code("kRed+1"))


if __name__ == "__main__":
    unittest.main()
