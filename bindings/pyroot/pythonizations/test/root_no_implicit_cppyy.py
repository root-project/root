import unittest


class ROOTNoImplicitCppyy(unittest.TestCase):
    """
    Test that import ROOT does not implicitly import cppyy.

    This test lives standalone in a separate file so that it does not get interference from other tests that may import
    parts of ROOT which involve initialization of the C++ runtime.
    """

    def test_no_implicit_cppyy_import(self):
        import sys

        import ROOT

        self.assertFalse("cppyy" in sys.modules)

        # Call some attribute that would initialize the C++ runtime
        ROOT.gInterpreter

        self.assertTrue("cppyy" in sys.modules)


if __name__ == "__main__":
    unittest.main()
