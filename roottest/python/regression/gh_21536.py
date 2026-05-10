# Regression test for https://github.com/root-project/root/issues/21536
#
# An off-by-one in outer_with_template() in clingwrapper.cxx caused dir() on a
# namespace to report ghost entries with the last character of each nested
# sub-scope dropped (e.g. 'Mat' alongside 'Math', 'Util' alongside 'Utils'),
# once those nested names were registered via rootmap entries. Triggering the
# bug requires rootmap entries with names of the form '<ns>::<sub>::<class>' so
# that the rootmap iteration in Cppyy::GetAllCppNames feeds '<sub>::<class>' to
# outer_with_template.

import os
import tempfile

import ROOT

ROOT.gInterpreter.Declare(
    """
namespace gh_21536 {
    namespace alpha { struct A {}; }
    namespace beta  { struct B {}; }
    namespace gamma { struct C {}; }
}
"""
)

rootmap_file = tempfile.NamedTemporaryFile(mode="w", suffix=".rootmap", delete=False)
try:
    # Point to an already-loaded library, as a dummy to make autoloading a
    # no-op without triggering errors (you get errors if the list of libraries
    # is empty). In the end we don't care about the autoloading by the rootmap
    # entries themselves.
    rootmap_file.write("[ libCore.so ]\n")
    for sub in ("alpha", "beta", "gamma"):
        rootmap_file.write("namespace gh_21536::%s\n" % sub)
        rootmap_file.write("class gh_21536::%s::A\n" % sub)
    rootmap_file.close()
    ROOT.gInterpreter.LoadLibraryMap(rootmap_file.name)

    ns = ROOT.gh_21536
    _ = ns.alpha, ns.beta, ns.gamma  # materialize as real attributes

    # Any name returned by dir() that has no matching attribute is a ghost left
    # over by the off-by-one truncation.
    ghosts = [s for s in dir(ns) if not hasattr(ns, s)]
    assert ghosts == [], "spurious truncated names in dir(ROOT.gh_21536): %r" % ghosts
finally:
    os.unlink(rootmap_file.name)
