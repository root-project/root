"""The checks of the HEPData test that the HS3 test suite has no check kind for.

The HS3 test suite checks that a document is imported and that a likelihood has
the expected values at fixed parameter points. This module adds, for RooFit:

- the round trip: the imported workspace is exported, and the exported document
  is imported again;
- the fit: the likelihood is minimized, and profiled over its parameters of
  interest. This is slow, and what a benchmark of the minimization wants to run.

The results of the fits are compared with expected_fit.json of a fixture.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from hs3testsuite_roofit_backend import RooFitBackend, suppress_root_output

# The parameters of interest for which the profile likelihood interval is computed, and the
# scan is run. A document can have dozens, and each one takes a fit per point.
MAX_PROFILED_PARAMETERS = 2
# The tolerance of the minimizer (RooMinimizer::setEps), tighter than its default of 1.
MINIMIZER_EPS = 0.01
# The points of a profile likelihood scan, in units of the uncertainty of the parameter.
SCAN_POINTS_IN_SIGMA = (-2.0, -1.0, 1.0, 2.0)
# The status of a minimization that is accepted: converged, or converged with a covariance matrix
# that was made positive definite.
ACCEPTED_STATUS = (0, 1)


def round_trip(ROOT, workspace) -> None:
    """Export the workspace and import the exported document again."""
    exported = str(ROOT.RooJSONFactoryWSTool(workspace).exportJSONtoString())
    again = ROOT.RooWorkspace("hs3hepdata_roundtrip_ws")
    if not ROOT.RooJSONFactoryWSTool(again).importJSONfromString(exported):
        raise AssertionError("the exported document could not be imported")


def model_config(ROOT, workspace, pdf_name: str):
    """The ModelConfig of the pdf, or None."""
    for obj in workspace.allGenericObjects():
        if obj.InheritsFrom("RooStats::ModelConfig") and obj.GetPdf() and obj.GetPdf().GetName() == pdf_name:
            return obj
    return None


def load_parameter_point(workspace, name: str | None) -> None:
    """Set the values and the constness of the parameters to the ones of a parameter point of the document.

    RooFit imports the parameter points as snapshots, and leaves the parameters at
    "default_values". In some documents, that point is the result of a conditional fit with
    most parameters fixed, and another point has the parameters as they are meant to be fitted.
    """
    if name and name != "default_values" and not workspace.loadSnapshot(name):
        raise AssertionError(f"parameter point {name!r} not found")


def create_nll(ROOT, workspace, target: dict[str, str], offset: bool = True):
    pdf = workspace.pdf(target["pdf"])
    if not pdf:
        raise AssertionError(f"PDF {target['pdf']!r} not found")
    data = workspace.data(target["data"])
    if not data:
        raise AssertionError(f"data {target['data']!r} not found")
    args = [ROOT.RooFit.Offset(offset)]
    config = model_config(ROOT, workspace, target["pdf"])
    if config:
        # The global observables are always given, also if there are none: without them, RooFit normalizes
        # the external constraints over their parameters.
        globs = config.GetGlobalObservables()
        args.append(ROOT.RooFit.GlobalObservables(globs if globs else ROOT.RooArgSet()))
        # The terms of the likelihood that are not part of the pdf, e.g. the regularization of an unfolding.
        constraints = config.GetExternalConstraints()
        if constraints and constraints.size() > 0:
            args.append(ROOT.RooFit.ExternalConstraints(constraints))
    # For benchmarks: the evaluation backend of RooFit, e.g. "codegen".
    backend = os.environ.get("HS3HEPDATA_EVAL_BACKEND")
    if backend:
        args.append(ROOT.RooFit.EvalBackend(backend))
    return pdf.createNLL(data, *args)


def minimize(ROOT, nll, strategy: int = 1):
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setPrintLevel(-1)
    minimizer.setStrategy(strategy)
    # At the default tolerance, Migrad stops up to a few percent of the uncertainty away from the minimum, at a
    # point that depends on the platform. That is more than the precision of published results.
    minimizer.setEps(MINIMIZER_EPS)
    status = minimizer.minimize("Minuit2", "Migrad")
    return minimizer, status


def run_fit(ROOT, workspace, fit: dict[str, Any], scan_points: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """Minimize the likelihood of the fit and profile it over its parameters of interest.

    Returns a dictionary with the keys "parameters" (per parameter of interest and
    normalization factor of the fit: "best_fit" and "hesse_error", and for the profiled
    parameters of interest "minus_one_sigma" and "plus_one_sigma"), "profile_scans" (per profiled parameter: "points" and
    "twice_delta_nll") and "seconds" (the time of each step).

    Without `scan_points`, the points of the scans are chosen from the uncertainties
    of the parameters, which is how the expected values are generated.
    """
    snapshot = "hs3hepdata_before_fit"
    workspace.saveSnapshot(snapshot, workspace.allVars())
    seconds = {}
    try:
        load_parameter_point(workspace, fit.get("parameter_point"))
        start = time.perf_counter()
        nll = create_nll(ROOT, workspace, fit["target"])
        seconds["create_nll"] = time.perf_counter() - start

        start = time.perf_counter()
        # The Minuit strategy of the fit, which a fixture can set if its model needs another one.
        strategy = int(fit.get("strategy", 1))
        minimizer, status = minimize(ROOT, nll, strategy)
        if status not in ACCEPTED_STATUS:
            raise AssertionError(f"the minimization failed with status {status}")
        seconds["minimize"] = time.perf_counter() - start
        start = time.perf_counter()
        hesse_status = minimizer.hesse()
        seconds["hesse"] = time.perf_counter() - start
        minimum = float(nll.getVal())
        # Whether the uncertainties from Hesse can be trusted: 3 is a full, accurate covariance matrix.
        result = minimizer.save()
        covariance_quality = int(result.covQual()) if hesse_status == 0 else -1

        pois = []
        for name in fit["parameters_of_interest"]:
            var = workspace.var(name)
            if not var:
                raise AssertionError(f"parameter of interest {name!r} not found")
            pois.append(var)
        # The other free parameters without a constraint, typically normalization factors of backgrounds.
        factors = [workspace.var(name) for name in fit.get("normalization_factors", [])]
        if not all(factors):
            raise AssertionError("a normalization factor of the fit was not found")
        parameters = {
            var.GetName(): {"best_fit": float(var.getVal()), "hesse_error": float(var.getError())}
            for var in pois + factors
        }
        workspace.saveSnapshot("hs3hepdata_best_fit", workspace.allVars())

        floating = [var for var in pois if not var.isConstant()]
        profiled = floating[:MAX_PROFILED_PARAMETERS]
        # All parameters of interest and normalization factors get the interval, which is what papers quote for
        # them; only the first parameters of interest get a scan.
        with_interval = floating + [var for var in factors if not var.isConstant()]
        start = time.perf_counter()
        if with_interval:
            minimizer.minos(ROOT.RooArgSet(*with_interval))
            for var in with_interval:
                # Minos reports a side that it could not find, e.g. at a limit of the parameter, as zero.
                if var.getAsymErrorLo() != 0:
                    parameters[var.GetName()]["minus_one_sigma"] = float(var.getAsymErrorLo())
                if var.getAsymErrorHi() != 0:
                    parameters[var.GetName()]["plus_one_sigma"] = float(var.getAsymErrorHi())
        seconds["minos"] = time.perf_counter() - start

        start = time.perf_counter()
        scans = {}
        for var in profiled:
            name = var.GetName()
            best = parameters[name]["best_fit"]
            if scan_points is not None:
                points = scan_points[name]
            else:
                sigma = parameters[name]["hesse_error"]
                points = [best + k * sigma for k in SCAN_POINTS_IN_SIGMA]
                points = [p for p in points if var.getMin() < p < var.getMax()]
            values = []
            kept = []
            for point in points:
                workspace.loadSnapshot("hs3hepdata_best_fit")
                var.setVal(point)
                var.setConstant(True)
                try:
                    _, scan_status = minimize(ROOT, nll, strategy)
                    converged = scan_status in ACCEPTED_STATUS
                    # A point of a generated scan that does not converge is left out of the fixture.
                    if scan_points is None and not converged:
                        continue
                    if not converged:
                        raise AssertionError(f"the minimization at {name} = {point} failed with status {scan_status}")
                    kept.append(float(point))
                    values.append(2.0 * (float(nll.getVal()) - minimum))
                finally:
                    var.setConstant(False)
            scans[name] = {"points": kept, "twice_delta_nll": values}
        seconds["profile_scans"] = time.perf_counter() - start
        return {
            "parameters": parameters,
            "profile_scans": scans,
            "covariance_quality": covariance_quality,
            "seconds": seconds,
        }
    finally:
        workspace.loadSnapshot(snapshot)


def compare(what: str, actual: float, expected: float, atol: float, rtol: float) -> None:
    allowed = atol + rtol * abs(expected)
    if not abs(actual - expected) <= allowed:
        raise AssertionError(
            f"{what}: got {actual:.10g}, expected {expected:.10g}, delta {abs(actual - expected):.3g} > {allowed:.3g}"
        )


def check_fit(ROOT, workspace, fit: dict[str, Any]) -> dict[str, float]:
    """Run the fit of a fixture and compare the result with its expected values.

    The best-fit values are compared in units of the expected uncertainty. Returns
    the time of each step in seconds.
    """
    tolerance = fit["tolerance"]
    scan_points = {name: scan["points"] for name, scan in fit.get("profile_scans", {}).items()}
    result = run_fit(ROOT, workspace, fit, scan_points)
    # Without an accurate covariance matrix, the uncertainties from Hesse mean nothing, and are not compared.
    trust_hesse = fit.get("covariance_quality", 3) == 3
    for name, expected in fit["expected"].items():
        actual = result["parameters"][name]
        sigma = abs(expected["hesse_error"])
        sides = [abs(expected[key]) for key in ("minus_one_sigma", "plus_one_sigma") if key in expected]
        if not trust_hesse and sides:
            sigma = sum(sides) / len(sides)
        compare(
            f"best fit of {name}", actual["best_fit"], expected["best_fit"], tolerance["best_fit_in_sigma"] * sigma, 0
        )
        for key in ("hesse_error", "minus_one_sigma", "plus_one_sigma"):
            if key == "hesse_error" and not trust_hesse:
                continue
            if key in expected:
                if key not in actual:
                    raise AssertionError(f"{key} of {name} was not computed")
                compare(f"{key} of {name}", actual[key], expected[key], 0.0, tolerance["uncertainty_rtol"])
    for name, expected in fit.get("profile_scans", {}).items():
        actual = result["profile_scans"][name]["twice_delta_nll"]
        for point, got, want in zip(expected["points"], actual, expected["twice_delta_nll"]):
            compare(
                f"profiled 2*delta(NLL) at {name} = {point:.6g}",
                got,
                want,
                tolerance["twice_delta_nll_atol"],
                tolerance["twice_delta_nll_rtol"],
            )
    return result["seconds"]


class HEPDataBackend(RooFitBackend):
    """The RooFit backend of the HS3 test suite, with the additional checks of the HEPData test.

    The runner of the suite imports a document once and then runs its checks. The
    additional checks are run right after the import, before the scans of the suite
    change the parameters. Their results are collected in `extra`, as
    (test_id, check_id, status, message), and the times in `seconds`.
    """

    def __init__(self, fits: dict[str, list[dict[str, Any]]] | None = None, additional_checks: bool = True) -> None:
        super().__init__()
        # Without the additional checks, this is the backend of the suite with the likelihood of the models.
        self.additional_checks = additional_checks
        # The fits to run, by test_id. Without an entry, the fits of a fixture are not run.
        self.fits = fits or {}
        self.extra: list[tuple[str, str, str, str]] = []
        self.seconds: dict[str, dict[str, float]] = {}

    def run_twice_delta_nll_scan(self, workspace, check: dict[str, Any], hs3_path: Path) -> list[float]:
        """The scan of the suite, with the likelihood as the fits have it.

        The backend of the suite builds the likelihood from the pdf and the data alone. The models here have a
        ModelConfig, whose global observables and external constraints (e.g. the regularization of an
        unfolding) belong to the likelihood as well.
        """
        target = check["target"]
        if "pdf" not in target or not model_config(self.ROOT, workspace, target["pdf"]):
            return super().run_twice_delta_nll_scan(workspace, check, hs3_path)
        self._apply_parameter_point(workspace, check["reference_point"])
        with suppress_root_output():
            nll = create_nll(self.ROOT, workspace, target, offset=False)
            reference = float(nll.getVal())
        return [2.0 * (value - reference) for value in self._scan(workspace, check, lambda: float(nll.getVal()))]

    def load_workspace(self, path: Path):
        if not self.additional_checks:
            return super().load_workspace(path)
        test_id = Path(path).parent.name
        times = self.seconds.setdefault(test_id, {})
        start = time.perf_counter()
        try:
            workspace = super().load_workspace(path)
        except Exception as e:
            message = "import failed: " + " ".join(str(e).rpartition("=>")[2].split())[:1000]
            self.extra.append((test_id, "round_trip", "failed", message))
            for fit in self.fits.get(test_id, []):
                self.extra.append((test_id, fit["id"], "failed", message))
            raise
        times["import"] = time.perf_counter() - start

        def run(check_id, function):
            start = time.perf_counter()
            try:
                with suppress_root_output():
                    seconds = function()
            except Exception as e:
                message = " ".join(str(e).rpartition("=>")[2].split())[:1000]
                self.extra.append((test_id, check_id, "failed", message or type(e).__name__))
            else:
                self.extra.append((test_id, check_id, "passed", ""))
                for step, value in (seconds or {}).items():
                    times[f"{check_id}:{step}"] = value
            times[check_id] = time.perf_counter() - start

        run("round_trip", lambda: round_trip(self.ROOT, workspace))
        for fit in self.fits.get(test_id, []):
            run(fit["id"], lambda fit=fit: check_fit(self.ROOT, workspace, fit))
        return workspace
