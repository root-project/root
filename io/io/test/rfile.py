import os
import platform
import unittest

import ROOT

RFile = ROOT.Experimental.RFile


class RFileTests(unittest.TestCase):
    def test_open_for_reading(self):
        """A RFile can read a ROOT file created by TFile"""

        fileName = "test_rfile_read_py.root"

        # Create a root file to open
        with ROOT.TFile.Open(fileName, "RECREATE") as tfile:
            hist = ROOT.TH1D("hist", "", 100, -10, 10)
            hist.FillRandom("gaus", 100)
            tfile.WriteObject(hist, "hist")

        with RFile.Open(fileName) as rfile:
            hist = rfile.Get("hist")
            self.assertNotEqual(hist, None)
            self.assertEqual(rfile.Get[ROOT.TH1D]("inexistent"), None)
            self.assertEqual(rfile.Get[ROOT.TH1F]("hist"), None)
            self.assertNotEqual(rfile.Get[ROOT.TH1]("hist"), None)

            if not platform.system() == "Windows":
                # TODO: re-enable it on Windows once the exception handling is fixed
                with self.assertRaises(ROOT.RException):
                    # This should fail because the file was opened as read-only
                    rfile.Put("foo", hist)

        os.remove(fileName)

    def test_writing_reading(self):
        """A RFile can be written into and read from"""

        fileName = "test_rfile_writeread_py.root"

        with RFile.Recreate(fileName) as rfile:
            hist = ROOT.TH1D("hist", "", 100, -10, 10)
            hist.FillRandom("gaus", 10)
            rfile.Put("hist", hist)
            if not platform.system() == "Windows":
                # TODO: re-enable it on Windows once the exception handling is fixed
                with self.assertRaises(ROOT.RException):
                    rfile.Put("hist/2", hist)

        with RFile.Open(fileName) as rfile:
            hist = rfile.Get("hist")
            self.assertNotEqual(hist, None)

        os.remove(fileName)

    def test_getkeyinfo(self):
        """A RFile can query individual keys of its objects"""

        fileName = "test_rfile_getkeyinfo_py.root"

        with RFile.Recreate(fileName) as rfile:
            hist = ROOT.TH1D("hist", "", 100, -10, 10)
            hist.FillRandom("gaus", 10)
            rfile.Put("hist", hist)
            rfile.Put("foo/hist", hist)
            rfile.Put("foo/bar/hist", hist)
            rfile.Put("foo/bar/hist2", hist)
            rfile.Put("foo/hist2", hist)

        with RFile.Open(fileName) as rfile:
            key = rfile.GetKeyInfo("hist")
            self.assertEqual(key.GetPath(), "hist")
            self.assertEqual(key.GetClassName(), "TH1D")

            key = rfile.GetKeyInfo("does_not_exist")
            self.assertEqual(key, None)

    def test_listkeys(self):
        """A RFile can query the keys of its objects and directories"""

        fileName = "test_rfile_listkeys_py.root"

        with RFile.Recreate(fileName) as rfile:
            hist = ROOT.TH1D("hist", "", 100, -10, 10)
            hist.FillRandom("gaus", 10)
            rfile.Put("hist", hist)
            rfile.Put("foo/hist", hist)
            rfile.Put("foo/bar/hist", hist)
            rfile.Put("foo/bar/hist2", hist)
            rfile.Put("foo/hist2", hist)

        with RFile.Open(fileName) as rfile:
            keys = [key.GetPath() for key in rfile.ListKeys()]
            self.assertEqual(keys, ["hist", "foo/hist", "foo/bar/hist", "foo/bar/hist2", "foo/hist2"])

            keys = [key.GetClassName() for key in rfile.ListKeys()]
            self.assertEqual(keys, ["TH1D"] * len(keys))

            self.assertEqual(
                [key.GetPath() for key in rfile.ListKeys("foo")],
                ["foo/hist", "foo/bar/hist", "foo/bar/hist2", "foo/hist2"],
            )

            self.assertEqual([key.GetPath() for key in rfile.ListKeys("foo/bar")], ["foo/bar/hist", "foo/bar/hist2"])

            self.assertEqual(
                [key.GetPath() for key in rfile.ListKeys("", listDirs=True, listObjects=False)], ["foo", "foo/bar"]
            )
            self.assertEqual(
                [key.GetPath() for key in rfile.ListKeys("", listDirs=True, listObjects=False, listRecursive=False)],
                ["foo"],
            )
            self.assertEqual(
                [key.GetPath() for key in rfile.ListKeys("", listDirs=True, listRecursive=False)], ["hist", "foo"]
            )

        os.remove(fileName)

    def test_putUnsupportedType(self):
        fileName = "test_rfile_putunsupported_py.root"

        with RFile.Recreate(fileName) as rfile:
            # Storing integers is unsupported
            with self.assertRaises(TypeError):
                rfile.Put("foo", 2)

            # Storing lists without an explicit template is unsupported
            with self.assertRaises(TypeError):
                rfile.Put("bar", [2, 3])

            # Storing lists with an explicit template is supported
            rfile.Put["std::vector<int>"]("bar", [2, 3])

            # Storing strings is supported
            rfile.Put("str", "foobar")

        with RFile.Open(fileName) as rfile:
            self.assertEqual(rfile.Get("str"), b"foobar")

        os.remove(fileName)

if __name__ == "__main__":
    unittest.main()
