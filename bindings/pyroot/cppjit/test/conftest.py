"""Suite-wide pytest infrastructure.

Tests within a file share interpreter state (cppdefs, loaded dictionaries,
pythonizations), so distributed runs must keep whole files on one worker.
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-crashing-xfails",
        action="store_true",
        default=False,
        help="run xfail(run=False) crash-class tests; a pass is a strict xpass",
    )


def _applies_here(mark):
    """Whether a mark's conditions hold; pytest evaluates string ones itself."""

    conditions = list(mark.args[:1])
    if "condition" in mark.kwargs:
        conditions.append(mark.kwargs["condition"])
    return all(True if isinstance(c, str) else bool(c) for c in conditions)


def pytest_collection_modifyitems(config, items):
    # [TEMP-CI-PROBE] CPPJIT_LOWLEVEL_PROBE={only|skip}:<name>[,...] keeps
    # only / drops the named tests, for bisecting the Windows x64 exit crash
    # in test_lowlevel. Removed together with the probe tests.
    probe = os.environ.get("CPPJIT_LOWLEVEL_PROBE")
    if probe:
        mode, _, names = probe.partition(":")
        names = set(names.split(","))
        keep = [i for i in items if (i.name in names) == (mode == "only")]
        drop = [i for i in items if i not in keep]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep

    if not config.getoption("--run-crashing-xfails"):
        return
    # Keep only the crash markers that claim this platform, and let them run:
    # the marker stays, so one that stopped crashing reports as a strict
    # xpass. The rest are deselected; they would only add state the real
    # suite never has.
    selected, deselected = [], []
    for item in items:
        crashing = [
            m
            for m in item.own_markers
            if m.name == "xfail" and m.kwargs.get("run") is False and _applies_here(m)
        ]
        if not crashing:
            deselected.append(item)
            continue
        item.own_markers = [
            pytest.mark.xfail(*m.args, **{**m.kwargs, "run": True}).mark
            if m in crashing
            else m
            for m in item.own_markers
        ]
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_configure(config):
    # -n implies --dist load; every mode finer than per-file is remapped
    # ("each" and "no" already keep files whole).
    if config.getoption("numprocesses", None) and config.getoption("dist", "no") in (
        "load",
        "worksteal",
        "loadscope",
        "loadgroup",
    ):
        config.option.dist = "loadfile"
