#####################################
#
# 'ORGANIZATION AND SIMULTANEOUS FITS' ROOT.RooFit tutorial macro #507
#
#   ROOT.RooFit memory tracing debug tool
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf507_debugtools():
    # Activate ROOT.RooFit memory tracing
    ROOT.RooTrace.active(ROOT.kTRUE)

    # Construct gauss(x,m,s)
    x = ROOT.RooRealVar("x", "x", -10, 10)
    m = ROOT.RooRealVar("m", "m", 0, -10, 10)
    s = ROOT.RooRealVar("s", "s", 1, -10, 10)
    gauss = ROOT.RooGaussian("g", "g", x, m, s)

    # Show dump of all ROOT.RooFit object in memory
    ROOT.RooTrace.dump()

    # Activate verbose mode
    ROOT.RooTrace.verbose(ROOT.kTRUE)

    # Construct poly(x,p0)
    p0 = ROOT.RooRealVar("p0", "p0", 0.01, 0., 1.)
    poly = ROOT.RooPolynomial("p", "p", x, ROOT.RooArgList(p0))

    # Put marker in trace list for future reference
    ROOT.RooTrace.mark()

    # model = f*gauss(x) + (1-f)*poly(x)
    f = ROOT.RooRealVar("f", "f", 0.5, 0., 1.)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(
        gauss, poly), ROOT.RooArgList(f))

    # Show object added to memory since marker
    ROOT.RooTrace.printObjectCounts()

    # Since verbose mode is still on, will see messages
    # pertaining to destructor calls of all ROOT.RooFit objects
    # made in self macro
    #
    # A call to ROOT.RooTrace.dump() at the end of self macro
    # should show that there a no ROOT.RooFit object left in memory


if __name__ == "__main__":
    rf507_debugtools()
