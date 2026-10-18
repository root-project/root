"""Crash diagnostics for cppjit tests that fail sporadically in ROOT's CI.

Not part of upstream cppjit: ROOT's CMakeLists.txt loads this as a pytest
plugin (``-p root_ci_diagnostics``) for selected test files, together with
``PYTHONMALLOC=debug``. It is meant to turn an unexplained crash into
an actionable report and can be dropped once the crash is understood.

Context: pyunittests-bindings-cppjit-test-datatypes dies in about one of six
runs on the mac-beta ARM64 runner (and nowhere else) with a segmentation
violation or bus error inside Python's cyclic garbage collector while it
traverses a list, during test26_function_pointers. That is the signature of a
Python object that was freed, or whose memory was corrupted, while a container
still referenced it. The damage can originate in any earlier test, because the
collection that trips over it only runs when the allocation count crosses the
GC threshold.

Three measures:

* ``PYTHONMALLOC=debug`` (set by CMake, reported in the header) keeps the
  build's default allocator but adds the debug hooks: freed blocks are filled
  with 0xDD, the guard bytes around every block are checked, and a double free
  or an overrun reports the offending block, so a dangling reference fails
  loudly and recognizably instead of corrupting whatever reused the memory.
* A full ``gc.collect()`` after every test moves the crash into the teardown
  of the test that actually leaves the damage behind.
* Python's faulthandler is re-armed before and after every test. ROOT installs
  its own signal handlers when the interpreter is initialized, which replaces
  the handler pytest's faulthandler plugin set up at session start; ROOT's
  handler prints a C stack without symbols on the macOS runners. Re-arming
  restores the Python traceback (and, with Python 3.14+, a C stack with
  symbols) at the point of the crash.
"""

import faulthandler
import gc
import os
import sys

import pytest

_stderr_fd = None

# The crash always surfaces in test26_function_pointers, in a collection that
# an allocation happens to trigger; collecting after every line of the tests
# around it pins the damage to the statement that first leaves a corrupted
# container behind, instead of wherever the GC threshold is crossed.
# ROOT_CI_DIAGNOSTICS_NO_GC=1 keeps only the faulthandler part, for runs that
# must leave the allocation pattern of the test untouched.
_NO_GC = bool(os.environ.get("ROOT_CI_DIAGNOSTICS_NO_GC"))
_TRACE_FILE = "test_datatypes.py"
_TRACE_TESTS = ("test24_", "test25_", "test26_", "test27_", "test28_", "test29_")
_forced_collections = 0


def _line_tracer(frame, event, arg):
    code = frame.f_code
    if not code.co_filename.endswith(_TRACE_FILE) or not code.co_name.startswith(
        _TRACE_TESTS
    ):
        return None
    if event == "line":
        global _forced_collections
        gc.collect()
        _forced_collections += 1
    return _line_tracer


def _rearm_faulthandler():
    # enable() is a no-op while already enabled, so cycle it to take the
    # signal handlers back from ROOT.
    faulthandler.disable()
    faulthandler.enable(file=_stderr_fd, all_threads=True)


def pytest_configure(config):
    # Capture starts after configuration, so fd 2 is still the real stderr.
    global _stderr_fd
    _stderr_fd = os.dup(2)
    _rearm_faulthandler()
    if not _NO_GC:
        sys.settrace(_line_tracer)


def pytest_report_header(config):
    gc_mode = (
        "forced collections disabled"
        if _NO_GC
        else f"gc.collect() after each test and after every line of {_TRACE_FILE} {_TRACE_TESTS}"
    )
    return (
        f"ROOT CI diagnostics: faulthandler re-armed, {gc_mode}, "
        f"PYTHONMALLOC={os.environ.get('PYTHONMALLOC', '<unset>')}, "
        f"Malloc*={[k for k in os.environ if k.startswith('Malloc')]}"
    )


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        f"ROOT CI diagnostics: {_forced_collections} collections forced by the line tracer"
    )


# Watch the reference counts of pytest's own TestReport objects. The crash on
# mac-beta tripped over a report in the terminal reporter's stats lists: a
# plain Python object cpyrt never sees, so the suspect is a stale Py_DECREF
# through a pointer to a dead object whose memory the report reused. Such a
# decrement shows up as a refcount drop of a report that is still referenced
# from the stats list, and checking after every test names the test during
# which it happened, without depending on a later crash.
_report_refcounts = {}


def _check_reports(config, where, current_nodeid=None):
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    # announce first: if the scan itself crashes on a dead report, the
    # faulthandler output follows this line
    os.write(_stderr_fd, f"ROOT CI diagnostics: {where}\n".encode())
    for key, reports in tr.stats.items():
        for rep in reports:
            ident = id(rep)
            known = _report_refcounts.get(ident)
            try:
                tname = type(rep).__name__
                rc = sys.getrefcount(rep)
                if tname not in ("TestReport", "CollectReport"):
                    raise TypeError(f"{tname} is not a report")
                if known is None:
                    # the runner still holds temporaries to the reports of the
                    # test being torn down; take their baseline at the next check
                    if rep.nodeid != current_nodeid:
                        _report_refcounts[ident] = (rc, rep.nodeid, rep.when)
                    continue
            except Exception as exc:  # a freed report may raise anything
                _report(
                    f"report at {ident:#x} in stats[{key!r}] is unreadable ({exc!r}), {where}"
                )
            if rc < known[0]:
                _report(
                    f"refcount of the {known[2]} report of {known[1]} dropped "
                    f"from {known[0]} to {rc}, {where}"
                )


def _report(msg):
    os.write(_stderr_fd, f"ROOT CI diagnostics: {msg}\n".encode())
    faulthandler.dump_traceback(file=_stderr_fd, all_threads=False)
    os._exit(99)  # make ctest show this output


def pytest_runtest_setup(item):
    _rearm_faulthandler()


def pytest_runtest_teardown(item, nextitem):
    _rearm_faulthandler()
    if not _NO_GC:
        gc.collect()
        _rearm_faulthandler()
    _check_reports(item.config, f"checked after {item.nodeid}", item.nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    _check_reports(session.config, "checked at session finish")


def pytest_unconfigure(config):
    global _stderr_fd
    sys.settrace(None)
    if _stderr_fd is not None:
        faulthandler.disable()
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
        os.close(_stderr_fd)
        _stderr_fd = None
