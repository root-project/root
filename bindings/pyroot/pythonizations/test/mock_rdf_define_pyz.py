import unittest

import ROOT


class MockRDFTest(unittest.TestCase):

    def test_mock_define(self):

        ROOT.gInterpreter.Declare('#include "mock_rdf_define_header.hxx"')

        ROOT.MockRDF._original_define = ROOT.MockRDF.mock_define
        ROOT.MockRDF.mock_define = None

        # On Linux and MacOS this should work, on Windows it triggers an error like
        # Traceback (most recent call last):
        #   File "C:\root-dev\build\x64\relwithdebinfo\runtutorials\test_header.py", line 4, in <module>
        #     val = df._OriginalDefine("x", ROOT.foo).Sum("x").GetValue()
        #                                   ^^^^^^^^
        #   File "C:\root-dev\build\x64\relwithdebinfo\bin\ROOT\_facade.py", line 120, in _fallback_getattr
        #     raise AttributeError("Failed to get attribute {} from ROOT".format(name))
        # AttributeError: Failed to get attribute foo from ROOT
        signature_of_foo = "float()"
        std_fn = ROOT.std.function(signature_of_foo)
        ROOT.MockRDF()._original_define[type(std_fn)]("myname", ROOT.foo)


if __name__ == '__main__':
    unittest.main()
