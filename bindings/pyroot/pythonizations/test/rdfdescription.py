import unittest
import ROOT

class RDFDescriptionTest(unittest.TestCase):
    """
    Testing of RDFDescription pythonization
    """
    def test_repr(self):
        """
        Test supported __repr__
        """
        df1 = ROOT.RDataFrame(1);
        self.assertEqual(df1.Describe().__repr__(), df1.Describe().AsString());


if __name__ == '__main__':
    unittest.main()
