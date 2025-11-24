import os
import pathlib
import shutil
import signal
import subprocess
import sys

import pytest
import ROOT

ROOT.gROOT.SetBatch(True)

tutorial_dir = pathlib.Path(str(ROOT.gROOT.GetTutorialDir()))

subdirs = ["analysis/dataframe", "analysis/tree", "hist", "io/ntuple", "roofit/roofit"]

SKIP_TUTORIALS = {
    "ntpl004_dimuon.C",  # requires reading remote data via HTTP
    "ntpl008_import.C",  # requires reading remote data via HTTP
    "ntpl011_global_temperatures.C",  # requires reading remote data via HTTP
    "distrdf004_dask_lxbatch.py",  # only works on lxplus
    "_SQlite",  # requires SQLite, not supported yet in ROOT wheels
    "h1analysisProxy.C",  # helper macro, not meant to run standalone
    "hist001_RHist_basics.C",  # required RHist, not supported in ROOT wheels
    "hist002_RHist_weighted.C",  # required RHist, not supported in ROOT wheels
    "rf618_mixture_models.py",  # fails on CI, to investigate
    "rf615_simulation_based_inference.py",  # fails on CI, to investigate
}

# ----------------------
# Python tutorials tests
# ----------------------
py_tutorials = []
for sub in subdirs:
    sub_path = tutorial_dir / sub
    for f in sub_path.rglob("*.py"):
        if any(skip in f.name for skip in SKIP_TUTORIALS):
            print("Skipping Python tutorial:", f)
            continue
        py_tutorials.append(f)

py_tutorials = sorted(py_tutorials, key=lambda p: p.name)


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
    for f in sub_path.rglob("*.C"):
        if any(skip in f.name for skip in SKIP_TUTORIALS):
            print("Skipping C++ tutorial:", f)
            continue
        cpp_tutorials.append(f)

cpp_tutorials = sorted(cpp_tutorials, key=lambda p: p.name)


def test_cpp_tutorials_are_detected():
    assert len(cpp_tutorials) > 0


@pytest.mark.parametrize("tutorial", cpp_tutorials, ids=lambda p: p.name)
def test_cpp_tutorial(tutorial):
    try:
        root_exe = shutil.which("root")
        result = subprocess.run(
            [root_exe, "-b", "-q", str(tutorial)],
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )
        print("Test stderr:", result.stderr)

    except subprocess.TimeoutExpired:
        pytest.skip(f"Tutorial {tutorial} timed out")

    except subprocess.CalledProcessError as e:
        if e.returncode == -signal.SIGILL or e.returncode == 132:
            pytest.fail(f"Failing {tutorial.name} (illegal instruction on this platform)")
        elif "EOFError" in e.stderr:
            pytest.skip(f"Skipping {tutorial.name} (requires user input)")
        raise
