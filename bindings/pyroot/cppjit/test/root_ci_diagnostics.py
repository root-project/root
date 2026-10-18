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

import ctypes
import faulthandler
import gc
import os
import sys
import sysconfig

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
# The report watch below found the damage done during test21 (or test20);
# collecting after every line there changed the allocation pattern enough to
# make the damage rare (2 in 700 instead of 33 in 700), so the collections
# stay where they were for the run that found it. In the tests around the
# damage the tracer only reads memory: it snapshots the stats lists when the
# test body starts and compares them before every line, which pins the
# statement without a collection.
_GC_TESTS = ("test24_", "test25_", "test26_", "test27_", "test28_", "test29_")
_WATCH_TESTS = ("test19_", "test20_", "test21_", "test22_")
_forced_collections = 0
_config = None
_session_active = False
_last_line = None
_seen_locals = set()


def _line_tracer(frame, event, arg):
    global _forced_collections, _last_line
    code = frame.f_code
    if not code.co_filename.endswith(_TRACE_FILE):
        return None
    name = code.co_name
    if name.startswith(_GC_TESTS):
        if event == "line":
            gc.collect()
            _forced_collections += 1
        return _line_tracer
    if name.startswith(_WATCH_TESTS) and _config is not None:
        if event == "call":
            _snapshot_stats(f"snapshot at the start of {name}")
            _seen_locals.clear()
            _last_line = None
        elif event == "line":
            _compare_stats(
                f"checked after line {_last_line} (before line {frame.f_lineno}) "
                f"of {name}"
            )
            _note_new_locals(frame)
            _last_line = frame.f_lineno
        elif event == "return":
            _compare_stats(f"checked at the return of {name}")
        return _line_tracer
    return None


def _note_new_locals(frame):
    # the addresses of the objects a test creates, to relate them to the
    # addresses of the damage
    for name, obj in list(frame.f_locals.items()):
        if id(obj) in _seen_locals:
            continue
        _seen_locals.add(id(obj))
        cpp = ""
        try:
            import cppjit

            cpp = f", C++ object at {cppjit.addressof(obj):#x}"
        except Exception:
            pass
        os.write(
            _stderr_fd,
            f"ROOT CI diagnostics: local {name} is a {type(obj).__name__} at "
            f"{id(obj):#x}{cpp}\n".encode(),
        )


def _rearm_faulthandler():
    # enable() is a no-op while already enabled, so cycle it to take the
    # signal handlers back from ROOT.
    faulthandler.disable()
    faulthandler.enable(file=_stderr_fd, all_threads=True)


def pytest_configure(config):
    # Capture starts after configuration, so fd 2 is still the real stderr.
    global _stderr_fd, _config, _session_active
    _config = config
    _stderr_fd = os.dup(2)
    _rearm_faulthandler()
    _install_report_finalizer()
    _session_active = True
    if not _NO_GC:
        sys.settrace(_line_tracer)


def pytest_report_header(config):
    gc_mode = (
        "forced collections disabled"
        if _NO_GC
        else f"gc.collect() after each test and after every line of {_TRACE_FILE} {_GC_TESTS}, "
        f"stats lists compared before every line of {_WATCH_TESTS}"
    )
    return (
        f"ROOT CI diagnostics: faulthandler re-armed, report finalizer "
        f"installed, {gc_mode}, "
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

# Everything below reads object memory through ctypes instead of touching the
# objects: a freed report has a zero reference count, and even iterating the
# list or calling sys.getrefcount on it would increment and decrement that
# count and deallocate the block a second time. The raw layout is that of a
# default (GIL) build: ob_refcnt, ob_type, ob_size, ob_item.
_RAW_LAYOUT_KNOWN = not sysconfig.get_config_var("Py_GIL_DISABLED")


def _raw_items(lst):
    ob_item = ctypes.c_void_p.from_address(
        id(lst) + 3 * ctypes.sizeof(ctypes.c_void_p)
    ).value
    n = len(lst)
    if not ob_item or n == 0:
        return []
    return list((ctypes.c_void_p * n).from_address(ob_item))


def _raw_refcnt(p):
    return ctypes.c_ssize_t.from_address(p).value


def _raw_type(p):
    return ctypes.c_void_p.from_address(p + ctypes.sizeof(ctypes.c_void_p)).value


def _check_reports(config, where, current_nodeid=None, quiet=False, record=True):
    if not _RAW_LAYOUT_KNOWN:
        return
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    # announce first: if the scan itself crashes on a dead report, the
    # faulthandler output follows this line
    if not quiet:
        os.write(_stderr_fd, f"ROOT CI diagnostics: {where}\n".encode())
    from _pytest.reports import CollectReport, TestReport

    report_types = {id(TestReport), id(CollectReport)}
    for key, reports in tr.stats.items():
        for index, p in enumerate(_raw_items(reports)):
            known = _report_refcounts.get(p)
            tp = _raw_type(p)
            rc = _raw_refcnt(p)
            if tp not in report_types or rc <= 0:
                was = (
                    f"the {known[2]} report of {known[1]}"
                    if known
                    else "a report not seen before"
                )
                _report(
                    f"stats[{key!r}][{index}] at {p:#x} is no longer a report "
                    f"(type {tp:#x}, refcount {rc}); it was {was}; {where}"
                )
            if known is None:
                # baselines only at the teardown checks, and not for the
                # reports of the test being torn down: the runner still
                # holds temporaries to recent reports
                if record:
                    rep = reports[index]  # alive: type and refcount checked above
                    if rep.nodeid != current_nodeid:
                        _report_refcounts[p] = (rc, rep.nodeid, rep.when)
            elif rc < known[0]:
                _report(
                    f"refcount of the {known[2]} report of {known[1]} dropped "
                    f"from {known[0]} to {rc}; {where}"
                )


def _report(msg):
    os.write(_stderr_fd, f"ROOT CI diagnostics: {msg}\n".encode())
    faulthandler.dump_traceback(file=_stderr_fd, all_threads=False)
    os._exit(99)  # make ctest show this output


# Within a test body pytest does not touch its reports, so the stats lists,
# their item arrays and the reports' headers must stay exactly as they were
# when the body started. The run of 2026-10-18 showed a report whose type
# slot had been overwritten with a pointer to a list while the report was
# still referenced, i.e. a stray write rather than a stray decrement, which
# is why the comparison covers the raw item pointers and type slots.
_stats_snapshot = {}


def _raw_ob_item(lst):
    return ctypes.c_void_p.from_address(
        id(lst) + 3 * ctypes.sizeof(ctypes.c_void_p)
    ).value


def _write(msg):
    os.write(_stderr_fd, f"ROOT CI diagnostics: {msg}\n".encode())


def _snapshot_stats(where):
    if not _RAW_LAYOUT_KNOWN:
        return
    tr = _config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    _stats_snapshot.clear()
    for key, reports in tr.stats.items():
        items = _raw_items(reports)
        _stats_snapshot[key] = (
            id(reports),
            _raw_ob_item(reports),
            items,
            [_raw_type(p) for p in items],
            [_raw_refcnt(p) for p in items],
        )
        _write(
            f"{where}: stats[{key!r}] is the list at {id(reports):#x} with "
            f"{len(items)} items at {_raw_ob_item(reports) or 0:#x}"
            + (f", first report at {items[0]:#x}" if items else "")
        )


def _compare_stats(where):
    if not _RAW_LAYOUT_KNOWN or not _stats_snapshot:
        return
    tr = _config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    for key, (list_id, ob_item, items, types, refcnts) in _stats_snapshot.items():
        reports = tr.stats.get(key)
        if reports is None or id(reports) != list_id:
            _corruption(f"stats[{key!r}] is a different list; {where}", items)
        now = _raw_items(reports)
        if _raw_ob_item(reports) != ob_item or len(now) != len(items):
            _corruption(
                f"stats[{key!r}] changed from {len(items)} items at {ob_item:#x} to "
                f"{len(now)} items at {_raw_ob_item(reports) or 0:#x}; {where}",
                items,
            )
        for index, (p, q) in enumerate(zip(items, now)):
            if q != p:
                _corruption(
                    f"stats[{key!r}][{index}] changed from {p:#x} to {q:#x}; {where}",
                    [p, q],
                )
            tp = _raw_type(p)
            rc = _raw_refcnt(p)
            if tp != types[index]:
                _corruption(
                    f"the type slot of the report at {p:#x} (stats[{key!r}][{index}]) "
                    f"changed from {types[index]:#x} to {tp:#x}; {where}",
                    [p, tp],
                )
            if rc <= 0 or rc < refcnts[index]:
                _corruption(
                    f"the refcount of the report at {p:#x} (stats[{key!r}][{index}]) "
                    f"changed from {refcnts[index]} to {rc}; {where}",
                    [p],
                )


def _corruption(msg, addresses):
    _write(msg)
    # the lookups only compare addresses; the dumps read the memory, which
    # may be unmapped for a garbage pointer, so they come last
    for addr in addresses:
        for line in _locate(addr):
            _write(f"{addr:#x} {line}")
    for addr in addresses:
        _write(f"memory around {addr:#x}: " + _dump_words(addr))
    _report("stopping here")


def _is_real_type(tp):
    # whether the type slot value tp points at a type object: its own type
    # is `type` or a metaclass. Everything the lookup below does with an
    # object goes through its type slot, so a corrupted one must be skipped.
    if not tp:
        return False
    try:
        mt = _raw_type(tp)
        if mt == id(type):
            return True
        if _raw_type(mt) != id(type):
            return False
        return issubclass(ctypes.cast(mt, ctypes.py_object).value, type)
    except Exception:
        return False


def _dump_words(addr, before=8, after=8):
    try:
        words = (ctypes.c_void_p * (before + after)).from_address(
            addr - before * ctypes.sizeof(ctypes.c_void_p)
        )
        return " ".join(
            ("|" if i == before else "") + f"{(w or 0):#018x}"
            for i, w in enumerate(words)
        )
    except Exception as exc:
        return f"unreadable ({exc!r})"


def _type_name(obj):
    try:
        return type(obj).__name__
    except Exception:
        return f"<type slot {_raw_type(id(obj)):#x}>"


def _locate(addr):
    # which live object owns the memory at addr: the object itself, a list's
    # item array, or the inside of an object
    found = []
    tr = _config.pluginmanager.get_plugin("terminalreporter")
    stats_lists = {id(v): k for k, v in tr.stats.items()} if tr else {}
    list_type = id(list)
    for o in gc.get_objects():
        oid = id(o)
        if not _is_real_type(_raw_type(oid)):
            found.append(
                f"(skipped the object at {oid:#x}, whose type slot {_raw_type(oid):#x} "
                f"is not a type)"
            )
            continue
        if oid == addr:
            desc = f"is a live {_type_name(o)} object"
        elif _raw_type(oid) == list_type:
            ob_item = _raw_ob_item(o)
            n = len(o)
            if ob_item and ob_item <= addr < ob_item + n * ctypes.sizeof(
                ctypes.c_void_p
            ):
                desc = (
                    f"is item {(addr - ob_item) // ctypes.sizeof(ctypes.c_void_p)} "
                    f"of the item array of the list of {n} at {oid:#x}"
                )
            else:
                continue
        else:
            try:
                size = sys.getsizeof(o)
            except Exception:
                continue
            # large sizes are external buffers, not the object's own memory
            if size <= 1024 and oid < addr < oid + size:
                desc = f"is {addr - oid} bytes into the {_type_name(o)} of size {size} at {oid:#x}"
            else:
                continue
        if oid in stats_lists:
            desc += f" (stats[{stats_lists[oid]!r}])"
        if _raw_type(oid) == list_type:
            try:
                desc += f" holding {[_type_name(x) for x in o[:4]]}"
            except Exception:
                pass
        found.append(desc)
    return found or ["matches no live GC-tracked object"]


# The checks above only see the damage after the fact. A finalizer on the
# report classes runs inside the very Py_DECREF that drops the last reference,
# so when a report that the stats lists still hold is deallocated, the Python
# traceback names the statement of the test and the C stack (Python 3.14+)
# names the cpyrt code that performed the decrement.
def _install_report_finalizer():
    from _pytest.reports import CollectReport, TestReport

    for cls in (TestReport, CollectReport):
        cls.__del__ = _report_finalizer


def _report_finalizer(self):
    if not _session_active or _config is None:
        return
    tr = _config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return
    for key, reports in tr.stats.items():
        for index, rep in enumerate(reports):
            if rep is self:
                break
        else:
            continue
        os.write(
            _stderr_fd,
            (
                f"ROOT CI diagnostics: the {getattr(self, 'when', '?')} report of "
                f"{getattr(self, 'nodeid', '?')} at {id(self):#x} is being "
                f"deallocated while stats[{key!r}][{index}] still holds it; "
                f"Python and C stacks of the decrement follow\n"
            ).encode(),
        )
        faulthandler.dump_traceback(file=_stderr_fd, all_threads=False)
        if hasattr(faulthandler, "dump_c_stack"):
            faulthandler.dump_c_stack(file=_stderr_fd)
        os._exit(98)  # make ctest show this output


def pytest_runtest_setup(item):
    _rearm_faulthandler()
    # the previous test's protocol is complete here: a stable point to take
    # the baselines of its reports
    _check_reports(item.config, f"checked at setup of {item.nodeid}")


def pytest_runtest_teardown(item, nextitem):
    _rearm_faulthandler()
    if not _NO_GC:
        gc.collect()
        _rearm_faulthandler()
    _check_reports(item.config, f"checked after {item.nodeid}", item.nodeid)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    global _session_active
    _check_reports(session.config, "checked at session finish")
    # from here on pytest drops its reports legitimately
    _session_active = False


def pytest_unconfigure(config):
    global _stderr_fd
    sys.settrace(None)
    if _stderr_fd is not None:
        faulthandler.disable()
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
        os.close(_stderr_fd)
        _stderr_fd = None
