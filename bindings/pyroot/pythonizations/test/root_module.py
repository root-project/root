import unittest

# Helper functions for the tests


def hasattr_recursive(obj, keys):
    """
    Check that a Python object has a nested attribute that can be retrieved by
    recursively using a given list of keys.
    """
    has_first = hasattr(obj, keys[0])
    if not has_first:
        return False
    if len(keys) == 1:
        return has_first
    return hasattr_recursive(getattr(obj, keys[0]), keys[1:])


def root_module_has(path):
    """
    Check if the ROOT module has a certain attribute, that can also be nested.
    For example:

        root_module_has("RooFit.Experimental")
    """
    import ROOT

    return hasattr_recursive(ROOT, path.split("."))


class ROOTModule(unittest.TestCase):
    """
    Testing features of the ROOT module implemented in the ROOT module facade
    """

    def test_import(self):
        """
        Test import
        """
        import ROOT

        self.assertEqual(ROOT.__name__, "ROOT")

    def test_relative_import(self):
        """
        Test relative import
        """
        from ROOT import TH1F

        self.assertEqual(TH1F.__name__, "TH1F")
        from ROOT import TH1F as Foo

        self.assertEqual(Foo.__name__, "TH1F")
        self.assertEqual(TH1F, Foo)

    def test_version(self):
        """
        Test __version__ property
        """
        import ROOT

        v = ROOT.__version__
        self.assertTrue(type(v) == str)
        # Fix for #14068: we take into account the different way of expressing the version
        # number before and starting with 6.32.00
        self.assertIn("/", v) if ROOT.gROOT.GetVersionInt() < 63200 else self.assertNotIn("/", v)
        self.assertIn(".", v)
        self.assertEqual(v, ROOT.gROOT.GetVersion())

    def test_ignore_cmdline_options(self):
        """
        Test module flag to ignore command line options
        """
        import ROOT

        self.assertEqual(ROOT.PyConfig.IgnoreCommandLineOptions, True)
        ROOT.PyConfig.IgnoreCommandLineOptions = False

    def test_implicit_root_namespace(self):
        """
        Test importing implicitly from the ROOT namespace
        """
        import ROOT

        ROOT.RVec
        ROOT.ROOT.RVec
        ROOT.VecOps.RVec
        ROOT.ROOT.VecOps.RVec
        ROOT.EnableImplicitMT()
        ROOT.ROOT.EnableImplicitMT()

    def test_import_nested_submodules(self):
        """
        Test that we can correctly import C++ namespaces and other things that
        should behave like Python modules, including nested cases.
        """
        #

        if root_module_has("RDF.Experimental.Distributed"):
            import ROOT.RDF.Distributed

        if root_module_has("RNTuple"):
            from ROOT import RNTuple

        if root_module_has("RooFit.Evaluator"):
            from ROOT.RooFit import Evaluator


if __name__ == "__main__":
    unittest.main()
