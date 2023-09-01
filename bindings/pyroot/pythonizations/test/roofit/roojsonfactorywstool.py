import unittest

import ROOT


class TestRooJSONFactoryWSTool(unittest.TestCase):
    """
    Test for RooJSONFactoryWSTool pythonizations.
    """

    def test_writedoc(self):

        ROOT.RooJSONFactoryWSTool.writedoc("roojsonfactorywstool_test_writedoc.tex")


if __name__ == "__main__":
    unittest.main()
