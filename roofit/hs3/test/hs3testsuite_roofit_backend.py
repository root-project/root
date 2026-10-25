from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class RooFitBackend:
    name = "roofit"

    def __init__(self) -> None:
        import ROOT  # type: ignore

        self.ROOT = ROOT
        ROOT.gROOT.SetBatch(True)
        ROOT.gErrorIgnoreLevel = ROOT.kFatal
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)

    def load_workspace(self, path: Path):
        ws = self.ROOT.RooWorkspace("hs3suite_ws")
        tool = self.ROOT.RooJSONFactoryWSTool(ws)
        with suppress_root_output():
            tool.importJSON(str(path))
        return ws

    def structure(self, workspace) -> dict[str, list[str]]:
        return {
            "pdfs": sorted(obj.GetName() for obj in workspace.allPdfs()),
            "functions": sorted(obj.GetName() for obj in workspace.allFunctions()),
            "data": sorted(obj.GetName() for obj in workspace.allData()),
        }

    def run_structure_check(self, workspace, check: dict[str, Any]) -> None:
        actual = self.structure(workspace)
        target = check.get("target", {})
        for key in ("pdfs", "functions", "data"):
            required = set(target.get(key, []))
            missing = required.difference(actual[key])
            if missing:
                raise AssertionError(f"missing {key}: {sorted(missing)}")

    def run_twice_delta_nll_scan(self, workspace, check: dict[str, Any], hs3_path: Path) -> list[float]:
        pdf_data_objs = []
        for pdf_name, data_name in self._resolve_target_pairs(check["target"], hs3_path):
            pdf = workspace.pdf(pdf_name)
            if not pdf:
                raise AssertionError(f"PDF {pdf_name!r} not found")
            data = workspace.data(data_name)
            if not data:
                raise AssertionError(f"data {data_name!r} not found")
            pdf_data_objs.append((pdf, data))

        self._apply_parameter_point(workspace, check["reference_point"])
        with suppress_root_output():
            nlls = [
                pdf.createNLL(data, self.ROOT.RooFit.NumCPU(1), self.ROOT.RooFit.EvalBackend("legacy"))
                for pdf, data in pdf_data_objs
            ]
            if len(nlls) == 1:
                nll = nlls[0]
            else:
                nll = self.ROOT.RooAddition("hs3suite_nll", "hs3suite_nll", self.ROOT.RooArgList(*nlls))
            reference = float(nll.getVal())
        return [2.0 * (value - reference) for value in self._scan(workspace, check, lambda: float(nll.getVal()))]

    def run_pdf_scan(self, workspace, check: dict[str, Any], hs3_path: Path) -> list[float]:
        target = check["target"]
        pdf = workspace.pdf(target["pdf"])
        if not pdf:
            raise AssertionError(f"PDF {target['pdf']!r} not found")
        norm_set = self.ROOT.RooArgSet()
        for name in target["observables"]:
            observable = workspace.var(name)
            if not observable:
                raise AssertionError(f"observable {name!r} not found")
            norm_set.add(observable)
        return self._scan(workspace, check, lambda: float(pdf.getVal(norm_set)))

    def run_function_scan(self, workspace, check: dict[str, Any], hs3_path: Path) -> list[float]:
        name = check["target"]["function"]
        func = workspace.function(name) or workspace.arg(name)
        if not func:
            raise AssertionError(f"function {name!r} not found")
        return self._scan(workspace, check, lambda: float(func.getVal()))

    def _scan(self, workspace, check: dict[str, Any], evaluate) -> list[float]:
        values = []
        scan_parameters = check["scan_parameters"]
        for point in check["scan_points"]:
            self._apply_parameter_point(workspace, check["reference_point"])
            self._apply_parameter_point(workspace, dict(zip(scan_parameters, point, strict=True)))
            with suppress_root_output():
                values.append(evaluate())
        return values

    def _resolve_target_pairs(self, target: dict[str, Any], hs3_path: Path) -> list[tuple[str, str]]:
        if "likelihood" not in target:
            return [(target["pdf"], target["data"])]
        with open(hs3_path, encoding="utf-8") as f:
            likelihoods = json.load(f).get("likelihoods", [])
        name = target["likelihood"]
        entry = next((e for e in likelihoods if e.get("name") == name), None)
        if entry is None:
            raise AssertionError(f"likelihood {name!r} not found in {hs3_path}")
        if len(entry["distributions"]) != len(entry["data"]):
            raise AssertionError(f"likelihood {name!r}: distributions/data length mismatch")
        return list(zip(entry["distributions"], entry["data"]))

    def _apply_parameter_point(self, workspace, values: dict[str, float]) -> None:
        for name, value in values.items():
            var = workspace.var(name)
            if var:
                var.setVal(float(value))
                continue
            arg = workspace.arg(name)
            if not arg:
                raise AssertionError(f"parameter {name!r} not found")
            # Parameters outside of all domains are imported as constants and cannot be
            # changed. Skip them, like the HS3TestSuite reference backend does.
            if not isinstance(arg, self.ROOT.RooConstVar):
                raise AssertionError(f"parameter {name!r} is a {arg.ClassName()}, which cannot be set")


@contextmanager
def suppress_root_output():
    """Suppress noisy C++ diagnostics that bypass RooMsgService."""

    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)
