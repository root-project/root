import subprocess
import sys
import pathlib
import ROOT
import os
import pytest

ROOT.gROOT.SetBatch(True)

tutorial_dir = pathlib.Path(str(ROOT.gROOT.GetTutorialDir()))

# ----------------------
# Python tutorials tests
# ----------------------
py_tutorials = list(tutorial_dir.rglob("*.py"))

def test_tutorials_are_detected():
    assert len(py_tutorials) > 0

@pytest.mark.parametrize("tutorial", py_tutorials, ids=lambda p: p.name)
def test_tutorial(tutorial):
    env = dict(**os.environ)
    # force matplotlib to use a non-GUI backend
    env["MPLBACKEND"] = "Agg"
    try:
        subprocess.run(
            [sys.executable, str(tutorial)],
            check=True,
            env=env,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        # read stderr to see if EOFError occurred
        if "EOFError" in e.stderr:
            pytest.skip("Skipping tutorial that requires user input")
        raise

# ----------------------
# C++ tutorials tests
# ----------------------
cpp_tutorials = list(tutorial_dir.rglob("*.C"))

def test_cpp_tutorials_are_detected():
    assert len(cpp_tutorials) > 0

@pytest.mark.parametrize("tutorial", cpp_tutorials, ids=lambda p: p.name)
def test_cpp_tutorial(tutorial):
    ROOT.gROOT.ProcessLine(f".x {tutorial}")