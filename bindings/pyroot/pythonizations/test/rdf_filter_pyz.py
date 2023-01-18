import unittest
import ROOT
import numpy as np

import os

class PyFilter(unittest.TestCase):
    """
    Testing Pythonized Filters of RDF
    """

    def test_cpp_functor(self):
        """
        Test that a C++ functor can be passed as a callable argument of a
        Filter operation.
        """

        ROOT.gInterpreter.Declare("""
        struct MyFunctor
        {
            bool operator()(ULong64_t l) { return l == 0; };
        };
        """)
        f = ROOT.MyFunctor()

        rdf = ROOT.RDataFrame(5)
        c = rdf.Filter(f, ["rdfentry_"]).Count().GetValue()

        self.assertEqual(c, 1)

    def test_std_function(self):
        """
        Test that an std::function can be passed as a callable argument of a
        Filter operation.
        """

        ROOT.gInterpreter.Declare("""
        std::function<bool(ULong64_t)> myfun = [](ULong64_t l) { return l == 0; };
        """)

        rdf = ROOT.RDataFrame(5)
        c = rdf.Filter(ROOT.myfun, ["rdfentry_"]).Count().GetValue()

        self.assertEqual(c, 1)

    
if __name__ == '__main__':
    unittest.main()
