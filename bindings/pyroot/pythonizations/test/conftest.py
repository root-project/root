from itertools import count

import pytest
import ROOT
from ROOT._pythonization._th1 import _th1_derived_classes_to_pythonize
from ROOT._pythonization._th2 import _th2_derived_classes_to_pythonize
from ROOT._pythonization._th3 import _th3_derived_classes_to_pythonize
from ROOT._pythonization._uhi.indexing import _temporarily_disable_add_directory

# UHI tests
th1_classes = {
    ("test_hist", "Test Histogram", 10, 1, 4): {
        "classes": [getattr(ROOT, klass) for klass in _th1_derived_classes_to_pythonize],
        "fill": lambda hist: hist.Fill(ROOT.gRandom.Uniform(0, 5), 1.0),
    },
    ("test_hist", "Test Histogram", 10, 1, 4, 10, 1, 4): {
        "classes": [getattr(ROOT, klass) for klass in _th2_derived_classes_to_pythonize],
        "fill": lambda hist: hist.Fill(ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5), 1.0),
    },
    ("test_hist", "Test Histogram", 10, 1, 4, 10, 1, 4, 10, 1, 4): {
        "classes": [getattr(ROOT, klass) for klass in _th3_derived_classes_to_pythonize],
        "fill": lambda hist: hist.Fill(
            ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5), ROOT.gRandom.Uniform(0, 5), 1.0
        ),
    },
}


@pytest.fixture(
    params=[
        (constructor_args, hist_class, hist_config["fill"])
        for constructor_args, hist_config in th1_classes.items()
        for hist_class in hist_config["classes"]
    ],
    scope="function",
    ids=lambda param: param[1].__name__,
)
def hist_setup(request):
    constructor_args, hist_class, fill = request.param
    with _temporarily_disable_add_directory():
        hist = hist_class(*constructor_args)
        for _ in range(10000):
            fill(hist)
        yield hist


# Helpers
def _iterate_bins(hist, flow=False):
    ranges = [
        range(0 if flow else 1, hist.GetNbinsX() + (2 if flow else 1)),
        range(0 if flow else 1, hist.GetNbinsY() + (2 if flow else 1)) if hist.GetDimension() > 1 else [0],
        range(0 if flow else 1, hist.GetNbinsZ() + (2 if flow else 1)) if hist.GetDimension() > 2 else [0],
    ]
    yield from ((i, j, k)[: hist.GetDimension()] for i in ranges[0] for j in ranges[1] for k in ranges[2])


# RDataLoader tests
TREE_NAME = "mytree"

_file_ids = count()


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own temporary directory managed by pytest."""
    monkeypatch.chdir(tmp_path)


def from_files(*dfs):
    """Write each RDataFrame to its own file and return an RDataFrame reading these files."""
    file_names = [f"input_{next(_file_ids)}.root" for _ in dfs]
    for df, file_name in zip(dfs, file_names):
        df.Snapshot(TREE_NAME, file_name)
    return ROOT.RDataFrame(TREE_NAME, file_names if len(file_names) > 1 else file_names[0])


def squares(num_of_entries, b1="(int) rdfentry_"):
    """RDataFrame with the column b1, defined by the given expression, and b2 = b1 * b1."""
    return ROOT.RDataFrame(num_of_entries).Define("b1", b1).Define("b2", "(double) b1 * b1")


def weighted(num_of_entries, b1="(int) rdfentry_"):
    """RDataFrame with the column b1, defined by the given expression, b2 = b1 * b1, b3 = b1 * 10 and b4 = b3 * 10."""
    return (
        ROOT
        .RDataFrame(num_of_entries)
        .Define("b1", b1)
        .Define("b2", "(int) b1 * b1")
        .Define("b3", "(double) b1 * 10")
        .Define("b4", "(double) b3 * 10")
    )


def vectors(num_of_entries, b1="(int) rdfentry_"):
    """RDataFrame with the column b1, defined by the given expression, and the vector columns v1 = {b1, b1 * 10} and
    v2 = {b1 * 100, b1 * 1000}."""
    return (
        ROOT
        .RDataFrame(num_of_entries)
        .Define("b1", b1)
        .Define("v1", "ROOT::VecOps::RVec<int>{ b1,  b1 * 10}")
        .Define("v2", "ROOT::VecOps::RVec<int>{ b1 * 100,  b1 * 1000}")
    )


def big_data(num_of_entries, b1="(int) rdfentry_"):
    """RDataFrame with the column b1, defined by the given expression, and columns computed from the entry number:
    b2 = entry * 2, b3 = entry + 10192, b4 = -entry and b5 = -entry - 10192."""
    return (
        ROOT
        .RDataFrame(num_of_entries)
        .Define("b1", b1)
        .Define("b2", "(double) rdfentry_ * 2")
        .Define("b3", "(int) rdfentry_ + 10192")
        .Define("b4", "(int) -rdfentry_")
        .Define("b5", "(double) -rdfentry_ - 10192")
    )
