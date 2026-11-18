"""Generate the fixtures of the HEPData test (roofit/hs3/test/hs3hepdata).

Usage, with ROOT and the runner of the HS3 test suite (HS3TestSuiteCode) on the
Python path:

    python hs3hepdata_generate_fixtures.py --cache <dir> [--fits] [--select <regex>]

The fixtures have the layout of the HS3 test suite, without the HS3 documents:
those stay on HEPData, and are pinned by the hashes in manifest.json. For every
document, the generator writes

- metadata.json: where the document comes from, and the scans of the fixture;
- expected.json: the checks of the HS3 test suite, with the values of
  2*(NLL - NLL(reference)) at fixed parameter points along each parameter of
  interest;
- expected_fit.json, with --fits: the best-fit values and uncertainties of the
  parameters of interest and profile likelihood scans, for the fits of the
  HEPData test, which the HS3 test suite has no check kind for.

The expected values are the ones that RooFit computes when the generator runs.
For now, they are assumed to be correct: they are not compared with the
published results of the analyses yet. Each fixture says so.

Existing fixtures are kept, unless --overwrite is given. Documents that RooFit
can't import get a fixture without scans.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import hs3hepdata_common as common

ASSUMPTION = (
    "The expected values are the ones that RooFit computed when the fixture was generated. For now, they are "
    "assumed to be correct: they have not been compared with the published results of the analysis."
)
# At most this many parameters of interest of an analysis get a scan.
MAX_SCANNED_PARAMETERS = 3
# At most this many free parameters that are not parameters of interest are recorded in a fit.
MAX_FACTORS = 20
# The tolerance of the scans at fixed parameter points. The values are differences of sums over all bins
# and constraint terms, which differ in the last digits between platforms and evaluation backends.
NLL_SCAN_TOLERANCE = {"rtol": 1e-6, "atol": 1e-5}
# The tolerance of the fits: the best-fit values in units of the uncertainty of the parameter, the
# uncertainties relative to their value, and the profiled values of 2*delta(NLL).
FIT_TOLERANCE = {
    "best_fit_in_sigma": 0.02,
    "uncertainty_rtol": 0.05,
    "twice_delta_nll_atol": 0.02,
    "twice_delta_nll_rtol": 0.02,
}


def analysis_targets(document: dict[str, Any], workspace) -> list[dict[str, Any]]:
    """The pdf, the data set and the parameters of interest of the analyses of a document.

    RooFit imports a likelihood over several channels as one RooSimultaneous and one
    combined data set. An analysis of asimov data is left out if there is one of
    observed data.
    """
    model_configs = document.get("misc", {}).get("ROOT_internal", {}).get("ModelConfigs", {})
    data_names = [data.GetName() for data in workspace.allData()]
    targets = []
    for analysis in document.get("analyses", []):
        likelihood = str(analysis.get("likelihood", ""))
        pdf = model_configs.get(analysis.get("name"), {}).get("pdfName")
        if pdf and likelihood.startswith(pdf + "_"):
            data = likelihood[len(pdf) + 1 :]
        elif len(data_names) == 1:
            data = data_names[0]
        else:
            continue
        if not pdf or not workspace.pdf(pdf) or not workspace.data(data):
            continue
        # Parameters of interest that are fixed are not fitted.
        pois = [
            name
            for name in analysis.get("parameters_of_interest", [])
            if workspace.var(name) and not workspace.var(name).isConstant()
        ]
        # Shape factors are free too, but one per bin: they are not what a paper quotes.
        factors = [
            name for name in free_parameters(workspace, pdf, data) if name not in pois and not name.startswith("gamma_")
        ]
        if pois or factors:
            targets.append(
                dict(pdf=pdf, data=data, parameters_of_interest=pois, normalization_factors=factors[:MAX_FACTORS])
            )
    # "hybrid" data sets are Asimov-like as well.
    observed = [t for t in targets if not any(word in t["data"].lower() for word in ("asimov", "hybrid"))]
    return observed or targets


# Parameter points that are preferred over "default_values" as the starting point of a fit, in this order.
# Documents written from workspaces of the SnSh fit framework keep the result of a conditional fit in
# "default_values", with almost all parameters fixed, and the nominal values with floating parameters here.
FIT_PARAMETER_POINTS = ("SnSh_AllVars_Nominal",)


# Settings of the fits that the models of a record need, by INSPIRE id. They are written to the fixtures.
FIT_SETTINGS = {
    # HEPData asks for a higher Minuit strategy for these likelihoods; at strategy 1, one fit stops short of
    # the minimum.
    "2689657": {"strategy": 2},
}


def fit_parameter_point(workspace) -> str:
    for name in FIT_PARAMETER_POINTS:
        if workspace.getSnapshot(name):
            return name
    return "default_values"


def free_parameters(workspace, pdf_name: str, data_name: str) -> list[str]:
    """The floating parameters of the pdf without a constraint term, sorted by name.

    Besides the parameters of interest, these are typically the normalization factors of
    backgrounds, which papers quote as well. A model without parameters of interest, e.g. for
    a background-only fit, is characterized by them.
    """
    import ROOT

    pdf, data = workspace.pdf(pdf_name), workspace.data(data_name)
    observables = ROOT.RooArgSet(data.get())
    parameters = pdf.getParameters(observables)
    floating = [p for p in parameters if isinstance(p, ROOT.RooRealVar) and not p.isConstant()]
    constraints = pdf.getAllConstraints(observables, ROOT.RooArgSet(parameters), False)
    constrained = set()
    for constraint in constraints:
        constrained.update(p.GetName() for p in constraint.getParameters(observables))
    return sorted(p.GetName() for p in floating if p.GetName() not in constrained)


def scan_points(var) -> list[float]:
    """Five points around the value of the variable, inside its range."""
    value, low, high = var.getVal(), var.getMin(), var.getMax()
    step = 0.1 * abs(value) if value != 0 else 0.1
    if math.isfinite(low) and math.isfinite(high):
        step = min(step, (high - low) / 6.0)
    first = -2
    # Shift the points if they would leave the range.
    while math.isfinite(low) and value + first * step <= low and first < 0:
        first += 1
    while math.isfinite(high) and value + (first + 4) * step >= high and first > -4:
        first -= 1
    return [float(f"{value + k * step:.12g}") for k in range(first, first + 5)]


def build_scans(backend, workspace, hs3_path: Path, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The checks of the kind twice_delta_nll_scan, with their expected values."""
    checks = []
    for target in targets:
        # Without parameters of interest, the scans are along the normalization factors.
        scanned = target["parameters_of_interest"] or target["normalization_factors"]
        for name in scanned[:MAX_SCANNED_PARAMETERS]:
            var = workspace.var(name)
            if var.isConstant():
                continue
            points = scan_points(var)
            check = {
                "id": "nll_scan_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{target['data']}_{name}"),
                "kind": "twice_delta_nll_scan",
                "target": {"pdf": target["pdf"], "data": target["data"]},
                # The other parameters stay at the values of the document.
                "reference_point": {name: float(var.getVal())},
                "scan_parameters": [name],
                "scan_points": [[point] for point in points],
                "expected": [],
                "tolerance": dict(NLL_SCAN_TOLERANCE),
            }
            try:
                expected = backend.run_twice_delta_nll_scan(workspace, check, hs3_path)
            except Exception as e:
                print(f"    no scan of {name}: {str(e).strip().splitlines()[-1][:200]}")
                continue
            if all(math.isfinite(value) for value in expected):
                check["expected"] = expected
                checks.append(check)
    return checks


def generate_fixture(backend, fixtures: Path, cache: Path, document: dict[str, Any], root_version: str):
    """Write metadata.json and expected.json of a document. Returns its manifest entry."""
    from hs3suite.manifest import canonical_sha256, extract_features, raw_sha256

    test_id = document["test_id"]
    hs3_path = common.document_path(cache, document)
    payload = common.load_json(hs3_path)
    fixture_dir = fixtures / "fixtures" / test_id

    notes = [ASSUMPTION]
    scans: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    try:
        workspace = backend.load_workspace(hs3_path)
    except Exception as e:
        notes.append("RooFit could not import the document when the fixture was generated, so it has no scans.")
        print(f"    import failed: {str(e).strip().splitlines()[-1][:200]}")
    else:
        targets = analysis_targets(payload, workspace)
        scans = build_scans(backend, workspace, hs3_path, targets)

    source = f"{common.HEPDATA_URL}/record/resource/{document['resource_id']}"
    where = document["filename"] + (f", file {document['member']}" if document.get("member") else "")
    packages = ", ".join(
        f"{p.get('name')} {p.get('version')}" for p in payload.get("metadata", {}).get("packages", []) if p.get("name")
    )
    metadata = {
        "schema_version": 1,
        "test_id": test_id,
        "title": (document.get("title") or f"INSPIRE {document['inspire_id']}")[:300],
        "source": f"HEPData, INSPIRE {document['inspire_id']}: {source} ({where})",
        "description": (
            f"HS3 {payload.get('metadata', {}).get('hs3_version', '?')} document published on HEPData"
            + (f", written by {packages}." if packages else ".")
        ),
        "notes": notes,
        "modified_from_source": False,
        "reference_backend": f"roofit {root_version}",
        "tags": ["hepdata"],
        "conformance": ["published"],
        "nll_scans": [
            {
                "id": check["id"],
                "pdf": check["target"]["pdf"],
                "data": check["target"]["data"],
                "parameters": check["scan_parameters"],
                "points": check["scan_points"],
            }
            for check in scans
        ],
    }
    checks = [{"id": "static_integrity", "kind": "static_integrity"}]
    structure = {
        "pdfs": sorted({target["pdf"] for target in targets}),
        "functions": [],
        "data": sorted({target["data"] for target in targets}),
    }
    checks.append({"id": "structure_import", "kind": "structure_import", "target": structure})
    expected = {"schema_version": 1, "test_id": test_id, "checks": checks + scans}
    common.write_json(fixture_dir / "metadata.json", metadata)
    common.write_json(fixture_dir / "expected.json", expected)

    features = extract_features(payload)
    return {
        "test_id": test_id,
        "path": f"fixtures/{test_id}",
        "hashes": {"sha256": raw_sha256(hs3_path), "canonical_sha256": canonical_sha256(hs3_path)},
        "features": features,
        "tags": metadata["tags"],
        "conformance": metadata["conformance"],
        "checks": [check["id"] for check in expected["checks"]],
    }


def fit_worker(fixtures: Path, cache: Path, test_id: str) -> int:
    """Generate expected_fit.json of one fixture. Runs in a process of its own, which can be stopped."""
    import hs3hepdata_roofit
    import ROOT
    from hs3hepdata_roofit import run_fit
    from hs3testsuite_roofit_backend import RooFitBackend

    document = next(
        d for d in common.load_json(fixtures / common.DOCUMENTS_FILE)["documents"] if d["test_id"] == test_id
    )
    hs3_path = common.document_path(cache, document)
    backend = RooFitBackend()
    workspace = backend.load_workspace(hs3_path)
    # Which parameters are free depends on the parameter point.
    parameter_point = fit_parameter_point(workspace)
    hs3hepdata_roofit.load_parameter_point(workspace, parameter_point)
    fits = []
    for target in analysis_targets(common.load_json(hs3_path), workspace):
        fit = {
            "parameter_point": parameter_point,
            **FIT_SETTINGS.get(document["inspire_id"], {}),
            "id": "fit_" + re.sub(r"[^A-Za-z0-9_.-]", "_", target["data"]),
            "target": {"pdf": target["pdf"], "data": target["data"]},
            "parameters_of_interest": target["parameters_of_interest"],
            "normalization_factors": target["normalization_factors"],
        }
        result = run_fit(ROOT, workspace, fit)
        values = [v for p in result["parameters"].values() for v in p.values()]
        values += [v for scan in result["profile_scans"].values() for v in scan["twice_delta_nll"]]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("the fit gave values that are not finite")
        fit["covariance_quality"] = result["covariance_quality"]
        fit["expected"] = result["parameters"]
        fit["profile_scans"] = result["profile_scans"]
        fit["tolerance"] = dict(FIT_TOLERANCE)
        fit["seconds_when_generated"] = round(sum(result["seconds"].values()), 2)
        fits.append(fit)
    if not fits:
        raise RuntimeError("no analysis with a pdf, a data set and free parameters")
    expected = {
        "schema_version": 1,
        "test_id": test_id,
        "comment": ASSUMPTION,
        "reference_backend": f"roofit {ROOT.gROOT.GetVersion()} ({ROOT.gROOT.GetGitCommit()})",
        "fits": fits,
    }
    common.write_json(fixtures / "fixtures" / test_id / "expected_fit.json", expected)
    return 0


def generate_fits(fixtures: Path, cache: Path, test_ids: list[str], timeout: float, jobs: int) -> None:
    def run(test_id):
        command = [sys.executable, "-B", __file__, "--cache", str(cache), "--fixtures", str(fixtures)]
        command += ["--fit-worker", test_id]
        start = time.perf_counter()
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            lines = (done.stderr or done.stdout).strip().splitlines()
            status = "ok" if done.returncode == 0 else "failed: " + (lines[-1][:200] if lines else "?")
        except subprocess.TimeoutExpired:
            status = f"no fit within {timeout:.0f} s"
        print(f"  fit {test_id}: {status} ({time.perf_counter() - start:.0f} s)", flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run, test_ids))


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="the cache directory of the HEPData test")
    parser.add_argument("--fixtures", type=Path, default=here / "hs3hepdata", help="the fixtures directory")
    parser.add_argument("--offline", action="store_true", help="use only what is in the cache")
    parser.add_argument("--select", help="regular expression for the test_ids to generate")
    parser.add_argument("--overwrite", action="store_true", help="generate fixtures that exist again")
    parser.add_argument("--fits", action="store_true", help="also generate expected_fit.json (slow)")
    parser.add_argument("--fit-timeout", type=float, default=600.0, help="seconds after which a fit is given up")
    parser.add_argument("--jobs", type=int, default=1, help="number of fits that run at the same time")
    parser.add_argument("--fit-worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    fixtures, cache = args.fixtures.resolve(), args.cache.resolve()

    if args.fit_worker:
        return fit_worker(fixtures, cache, args.fit_worker)

    # The documents: the ones that are listed already, and the ones that HEPData lists now.
    documents_file = fixtures / common.DOCUMENTS_FILE
    documents = common.load_json(documents_file)["documents"] if documents_file.is_file() else []
    known = {document["test_id"] for document in documents}
    if not args.offline:
        for resource in common.query_resources():
            if not common.prepare_resource(cache, resource, offline=False):
                continue
            for member, _ in common.resource_documents(cache, resource):
                test_id = common.make_test_id(resource, member)
                if test_id not in known:
                    documents.append(dict(resource, member=member, test_id=test_id))
                    known.add(test_id)
    documents.sort(key=lambda d: d["test_id"])
    common.write_json(documents_file, {"documents": documents})
    available = common.prepare_documents(cache, documents, offline=True)

    selected = [d for d in available if not args.select or re.search(args.select, d["test_id"])]
    manifest_file = fixtures / "manifest.json"
    manifest = (
        common.load_json(manifest_file)
        if manifest_file.is_file()
        else {"schema_version": 1, "generated_by": "roofit/hs3/test/hs3hepdata_generate_fixtures.py", "fixtures": []}
    )
    entries = {entry["test_id"]: entry for entry in manifest["fixtures"]}

    todo = [d for d in selected if args.overwrite or d["test_id"] not in entries]
    if todo:
        import ROOT
        from hs3hepdata_roofit import HEPDataBackend

        # The scans are evaluated like the test evaluates them.
        backend = HEPDataBackend(additional_checks=False)
        for i, document in enumerate(todo):
            print(f"[{i + 1}/{len(todo)}] {document['test_id']}", flush=True)
            entries[document["test_id"]] = generate_fixture(
                backend, fixtures, cache, document, str(ROOT.gROOT.GetVersion())
            )
            manifest["fixtures"] = [entries[key] for key in sorted(entries)]
            common.write_json(manifest_file, manifest)

    if args.fits:
        test_ids = [
            d["test_id"]
            for d in selected
            if d["test_id"] in entries
            and (args.overwrite or not (fixtures / "fixtures" / d["test_id"] / "expected_fit.json").is_file())
        ]
        generate_fits(fixtures, cache, test_ids, args.fit_timeout, args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
