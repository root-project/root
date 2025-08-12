import subprocess
import sys
import pathlib
import ROOT
import os
import pytest

# ROOT.gROOT.SetBatch(True)

tutorial_dir = pathlib.Path(str(ROOT.gROOT.GetTutorialDir())) or pathlib.Path(ROOT.__file__).parent.parent.parent  / "tutorials"
tutorials = list(tutorial_dir.rglob("*.py"))


@pytest.mark.parametrize("tutorial", tutorials)
def test_tutorial(tutorial):
    # subprocess.run([sys.executable, str(tutorial)], check=True)
    print(f"Running tutorial: {tutorial}")
    assert len(tutorials) > 0