import pytest
import ROOT
from ROOT._pythonization._th1 import _th1_derived_classes_to_pythonize
from ROOT._pythonization._th2 import _th2_derived_classes_to_pythonize
from ROOT._pythonization._th3 import _th3_derived_classes_to_pythonize
from ROOT._pythonization._uhi.indexing import _temporarily_disable_add_directory

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
