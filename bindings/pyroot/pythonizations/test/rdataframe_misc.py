import cppyy
import platform
import unittest

import ROOT


class RDataFrameMisc(unittest.TestCase):
    """Miscellaneous RDataFrame tests"""

    def test_empty_filenames(self):
        """
        An empty list of filenames should be detected and the user should be informed
        """
        # See https://github.com/root-project/root/issues/7541 and
        # https://bugs.llvm.org/show_bug.cgi?id=49692 :
        # llvm JIT fails to catch exceptions on MacOS ARM, so we disable their testing
        # Also fails on Windows for the same reason
        if (
            (platform.processor() != "arm" or platform.mac_ver()[0] == '') and not
            platform.system() == "Windows"
        ):
            # With implicit conversions, cppyy also needs to try dispatching to the various
            # constructor overloads. The C++ exception will be thrown, but will be incapsulated
            # in a more generic TypeError telling the user that none of the overloads worked
            with self.assertRaisesRegex(TypeError, "RDataFrame: empty list of input files."):
                ROOT.RDataFrame("events", [])

            # When passing explicitly the vector of strings, type dispatching will not be necessary
            # and the real C++ exception will immediately surface
            with self.assertRaisesRegex(cppyy.gbl.std.invalid_argument, "RDataFrame: empty list of input files."):
                ROOT.RDataFrame("events", ROOT.std.vector[ROOT.std.string]())

            with self.assertRaisesRegex(TypeError, "RDataFrame: empty list of input files."):
                ROOT.RDataFrame("events", ())


if __name__ == '__main__':
    unittest.main()
