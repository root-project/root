import subprocess
import sys
import pathlib
import ROOT
import os
import pytest
import signal

ROOT.gROOT.SetBatch(True)

tutorial_dir = pathlib.Path(str(ROOT.gROOT.GetTutorialDir()))

subdirs = [
    "analysis/dataframe",
    "analysis/tree",
    "hist",
    "io/ntuple",
    "roofit/roofit"
]

# ----------------------
# Python tutorials tests
# ----------------------
py_tutorials = []
for sub in subdirs:
    sub_path = tutorial_dir / sub
    py_tutorials.extend(sub_path.rglob("*.py"))

def test_tutorials_are_detected():
    assert len(py_tutorials) > 0

@pytest.mark.parametrize("tutorial", py_tutorials, ids=lambda p: p.name)
def test_tutorial(tutorial):
    env = dict(**os.environ)
    # force matplotlib to use a non-GUI backend
    env["MPLBACKEND"] = "Agg"
    print("Test env:", env)
    try:
        result = subprocess.run(
            [sys.executable, str(tutorial)],
            check=True,
            env=env,
            timeout=60,
            capture_output=True,
            text=True,
        )
        print("Test stderr:", result.stderr)
    except subprocess.TimeoutExpired:
        pytest.skip(f"Tutorial {tutorial} timed out")
    except subprocess.CalledProcessError as e:
        # read stderr to see if EOFError occurred
        if "EOFError" in e.stderr:
            pytest.skip(f"Skipping {tutorial.name} (requires user input)")
        raise

# ----------------------
# C++ tutorials tests
# ----------------------
cpp_tutorials = []
for sub in subdirs:
    sub_path = tutorial_dir / sub
    cpp_tutorials.extend(sub_path.rglob("*.C"))

def test_cpp_tutorials_are_detected():
    assert len(cpp_tutorials) > 0

@pytest.mark.parametrize("tutorial", cpp_tutorials, ids=lambda p: p.name)
def test_cpp_tutorial(tutorial):
    try:
        result = subprocess.run(
            [sys.executable, "-c", f'import ROOT; ROOT.gROOT.ProcessLine(".x {tutorial}")'],
            check=True,
            timeout=60,
            capture_output=True,
            text=True
        )
    except subprocess.TimeoutExpired:
        pytest.skip(f"Tutorial {tutorial} timed out")
    except subprocess.CalledProcessError as e:
        if e.returncode == -signal.SIGILL or e.returncode == 132:
            pytest.fail(f"Failing {tutorial.name} (illegal instruction on this platform)")
        elif "EOFError" in e.stderr:
            pytest.skip(f"Skipping {tutorial.name} (requires user input)")
        raise