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

_stderr_fd = None


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


def pytest_report_header(config):
    return (
        "ROOT CI diagnostics: faulthandler re-armed and gc.collect() after "
        f"each test, PYTHONMALLOC={os.environ.get('PYTHONMALLOC', '<unset>')}"
    )


def pytest_runtest_setup(item):
    _rearm_faulthandler()


def pytest_runtest_teardown(item, nextitem):
    _rearm_faulthandler()
    gc.collect()
    _rearm_faulthandler()


def pytest_unconfigure(config):
    global _stderr_fd
    if _stderr_fd is not None:
        faulthandler.disable()
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
        os.close(_stderr_fd)
        _stderr_fd = None
