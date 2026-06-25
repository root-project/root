from __future__ import annotations

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

    def run_twice_delta_nll_scan(self, workspace, check: dict[str, Any]) -> list[float]:
        target = check["target"]
        pdf = workspace.pdf(target["pdf"])
        if pdf is None:
            raise AssertionError(f"PDF {target['pdf']!r} not found")
        data = workspace.data(target["data"])
        if data is None:
            raise AssertionError(f"data {target['data']!r} not found")

        self._apply_parameter_point(workspace, check["reference_point"])
        with suppress_root_output():
            nll = pdf.createNLL(
                data,
                self.ROOT.RooFit.NumCPU(1),
                self.ROOT.RooFit.EvalBackend("legacy"),
            )
            reference = float(nll.getVal())
        values = []
        scan_parameter = check["scan_parameter"]
        for point in check["scan_points"]:
            self._apply_parameter_point(workspace, check["reference_point"])
            var = workspace.var(scan_parameter)
            if var is None:
                raise AssertionError(f"scan parameter {scan_parameter!r} not found")
            var.setVal(float(point))
            with suppress_root_output():
                values.append(2.0 * (float(nll.getVal()) - reference))
        return values

    def _apply_parameter_point(self, workspace, values: dict[str, float]) -> None:
        for name, value in values.items():
            var = workspace.var(name)
            if var is not None:
                var.setVal(float(value))


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
