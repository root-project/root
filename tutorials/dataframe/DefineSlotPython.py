"""
DefineSlotPython.py

Demonstrate a thread-safe Define pattern for Python users using `rdfslot_`.

This shows how to declare a small C++ functor and use it from PyROOT via
`RDataFrame.Define` so each thread gets a thread-local RNG seeded by the slot.

Run with:

  python3 DefineSlotPython.py

Requires a working PyROOT installation (ROOT with Python bindings).
"""
from __future__ import print_function

import ROOT

def main():
    # Declare a small C++ function that uses a thread_local TRandom3 per slot.
    ROOT.gInterpreter.Declare("""
    double myfunctor(unsigned int rdfslot) {
      static thread_local TRandom3 r3(rdfslot);
      return r3.Gaus();
    }
    """)

    # Create an RDataFrame and use Define passing the C++ functor by name.
    # Note: the expression uses the helper column `rdfslot_` which is different
    # for each thread when implicit multi-threading is enabled.
    # Enable implicit multi-threading for example (optional).
    # This must happen before constructing the RDataFrame.
    try:
        ROOT.EnableImplicitMT()
    except Exception:
        # If ROOT was built without implicit MT, continue single-threaded
        pass

    rdf = ROOT.RDataFrame(100000)
    rdf_x = rdf.Define("x", "myfunctor(rdfslot_)")
    h = rdf_x.Histo1D(("h", "Gaussian per-slot test", 100, -5, 5), "x")
    # Trigger event loop by accessing the filled histogram.
    print("Histogram entries:", int(h.GetValue().GetEntries()))

if __name__ == '__main__':
    main()
