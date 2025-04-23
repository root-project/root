import os
import shlex
import subprocess
import sys

import pytest

import ROOT

FILENAME = "hadd_input_validation"


def create_empty_files():
    with ROOT.TFile(FILENAME, "recreate") as _:
        pass

    with open(FILENAME+"_txt", "w") as _:
        pass

    with open("indirect.txt", "w") as text_file:
        text_file.write(FILENAME+"\n")
        text_file.write(FILENAME+"_txt\n")


@pytest.fixture(scope="class")
def setup_inputvalidation(request):
    """
    Setup the test environment. Create the input files then clean up at the
    end of the test execution.
    """

    create_empty_files()
    yield
    os.remove(FILENAME)
    os.remove(FILENAME+"_txt")
    os.remove("indirect.txt")


@pytest.mark.usefixtures("setup_inputvalidation")
class TestInputValidation:
    """Test input file format validation in hadd"""

    def test_hadd_input_validation(self):
        """
        Pass directly the filenames as argument
        """
        outputfile = "test_hadd_input_validation.root"
        try:
            out = subprocess.run(shlex.split(
                f"hadd {outputfile} {FILENAME} {FILENAME+'_txt'}"), capture_output=True)
        finally:
            os.remove(outputfile)

        assert f"exiting due to error in {FILENAME+'_txt'}" in out.stderr.decode()

    def test_hadd_input_validation_indirect(self):
        """
        Use an indirect file that contains all ROOT file names
        """
        outputfile = "test_hadd_input_validation_indirect.root"
        try:
            out = subprocess.run(shlex.split(
                f"hadd {outputfile} @indirect.txt"), capture_output=True)

        finally:
            os.remove(outputfile)

        assert f"exiting due to error in {FILENAME+'_txt'}" in out.stderr.decode()


if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
