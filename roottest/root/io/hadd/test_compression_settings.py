import os
import shlex
import subprocess
import sys
from array import array

import pytest

import ROOT

FILENAME = "hadd_test_compression_settings.root"
TREENAME = "mytree"


def create_dataset():
    with ROOT.TFile(
            FILENAME,
            "recreate",
            "myfile",
            ROOT.RCompressionSetting.EDefaults.kUseAnalysis) as f:
        t = ROOT.TTree(TREENAME, TREENAME)
        x = array("i", [0])
        x[0] = 42
        t.Branch("x", x, "x/I")
        t.Fill()
        f.WriteObject(t, TREENAME)


@pytest.fixture(scope="class")
def setup_testcompressionalgorithm(request):
    """
    Setup the test environment. Create the input dataset then clean up at the
    end of the test execution.
    """

    create_dataset()
    yield
    os.remove(FILENAME)


@pytest.mark.usefixtures("setup_testcompressionalgorithm")
class TestCompressionAlgorithm:
    """Test compression algorithm info before/after hadd"""

    def test_hadd_default_compression_settings(self):
        """
        The default behaviour of hadd is to use kUseCompiledDefault compression
        """
        with ROOT.TFile(FILENAME) as f:
            initial_settings = f.GetCompressionSettings()

        outputfile = "test_hadd_default_compression_settings.root"
        try:
            subprocess.run(shlex.split(
                f"hadd {outputfile} {FILENAME}"), check=True)

            with ROOT.TFile(outputfile) as f:
                output_settings = f.GetCompressionSettings()

            assert output_settings == ROOT.RCompressionSetting.EDefaults.kUseCompiledDefault, f"{output_settings=}"
        finally:
            os.remove(outputfile)

    def test_hadd_preserve_compression_settings(self):
        """
        The '-ff' option instructs hadd to use the same compression settings as
        the first input file
        """
        with ROOT.TFile(FILENAME) as f:
            initial_settings = f.GetCompressionSettings()

        outputfile = "test_hadd_preserve_compression_settings.root"
        try:
            subprocess.run(shlex.split(
                f"hadd -ff {outputfile} {FILENAME}"), check=True)

            with ROOT.TFile(outputfile) as f:
                output_settings = f.GetCompressionSettings()

            assert output_settings == initial_settings, f"{output_settings=}"
        finally:
            os.remove(outputfile)


if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
