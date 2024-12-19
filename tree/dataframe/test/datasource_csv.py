import unittest

import ROOT

class DataSourceCSV(unittest.TestCase):
    """Test for the FromCSV factory method pythonization"""

    def test_simple(self):
        """Test the single argument construction"""
        df = ROOT.RDF.FromCSV('RCsvDS_test_headers.csv')
        self.assertEqual(60, df.Max('Age').GetValue())

    def test_positional(self):
       """Test the construction with positional arguments"""
       df = ROOT.RDF.FromCSV('RCsvDS_test_noheaders.csv', False)
       self.assertEqual(60, df.Max('Col1').GetValue())

    def test_keyword(self):
        """Test the construction with keyword arguments"""
        df = ROOT.RDF.FromCSV('RCsvDS_test_parsing.csv', delimiter = ' ', leftTrim = True, rightTrim = True,
                              skipFirstNLines = 1, skipLastNLines = 2, comment = '#',
                              columnNames = ['FirstName', 'LastName', '', ''])
        self.assertEqual(1, df.Count().GetValue())
        self.assertEqual('Harry', df.Take['string']('FirstName').at(0))

if __name__ == "__main__":
    unittest.main()
