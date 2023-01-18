import unittest
import ROOT
import numpy as np

class PyDefine(unittest.TestCase):
    """
    Testing Pythonized Define of RDF
    """

    def test_cpp_functor(self):
        """
        Test that a C++ functor can be passed as a callable argument of a
        Define operation.
        """

        ROOT.gInterpreter.Declare("""
        struct MyFunctor
        {
            ULong64_t operator()(ULong64_t l) { return l*l; };
        };
        """)
        f = ROOT.MyFunctor()

        rdf = ROOT.RDataFrame(5)
        rdf2 = rdf.Define("x", f, ["rdfentry_"])

        for x,y in zip(rdf2.Take['ULong64_t']("rdfentry_"), rdf2.Take['ULong64_t']("x")):
           self.assertEqual(x*x, y)

    def test_std_function(self):
        """
        Test that an std::function can be passed as a callable argument of a
        Define operation.
        """

        ROOT.gInterpreter.Declare("""
        std::function<ULong64_t(ULong64_t)> myfun = [](ULong64_t l) { return l*l; };
        """)

        rdf = ROOT.RDataFrame(5)
        rdf2 = rdf.Define("x", ROOT.myfun, ["rdfentry_"])

        for x,y in zip(rdf2.Take['ULong64_t']("rdfentry_"), rdf2.Take['ULong64_t']("x")):
           self.assertEqual(x*x, y)

    
if __name__ == '__main__':
    unittest.main()
