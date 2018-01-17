#####################################
#
# 'ORGANIZATION AND SIMULTANEOUS FITS' ROOT.RooFit tutorial macro #505
#
# Reading and writing ASCII configuration files
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf505_asciicfg():
    # C r e a t e  p d f
    # ------------------

    # Construct gauss(x,m,s)
    x = ROOT.RooRealVar("x", "x", -10, 10)
    m = ROOT.RooRealVar("m", "m", 0, -10, 10)
    s = ROOT.RooRealVar("s", "s", 1, -10, 10)
    gauss = ROOT.RooGaussian("g", "g", x, m, s)

    # Construct poly(x,p0)
    p0 = ROOT.RooRealVar("p0", "p0", 0.01, 0., 1.)
    poly = ROOT.RooPolynomial("p", "p", x, ROOT.RooArgList(p0))

    # model = f*gauss(x) + (1-f)*poly(x)
    f = ROOT.RooRealVar("f", "f", 0.5, 0., 1.)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(
        gauss, poly), ROOT.RooArgList(f))

    # F i t   m o d e l   t o   t o y   d a t a
    # -----------------------------------------

    d = model.generate(ROOT.RooArgSet(x), 1000)
    model.fitTo(d)

    # W r i t e   p a r a m e t e r s   t o   a s c i i   f i l e
    # -----------------------------------------------------------

    # Obtain set of parameters
    params = model.getParameters(ROOT.RooArgSet(x))

    # Write parameters to file
    params.writeToFile("rf505_asciicfg_example.txt")

    # R e a d    p a r a m e t e r s   f r o m    a s c i i   f i l e
    # ----------------------------------------------------------------

    # Read parameters from file
    params.readFromFile("rf505_asciicfg_example.txt")
    params.Print("v")

    # Read parameters from section 'Section2' of file
    params.readFromFile("rf505_asciicfg.txt", "", "Section2")
    params.Print("v")

    # Read parameters from section 'Section3' of file. Mark all
    # variables that were processed with the "READ" attribute
    params.readFromFile("rf505_asciicfg.txt", "READ", "Section3")

    # Print the list of parameters that were not read from Section3
    print "The following parameters of the were _not_ read from Section3: ", params.selectByAttrib("READ", ROOT.kFALSE)

    # Read parameters from section 'Section4' of file, contains
    # 'include file' statement of rf505_asciicfg_example.txt
    # so that we effective read the same
    params.readFromFile("rf505_asciicfg.txt", "", "Section4")
    params.Print("v")


if __name__ == "__main__":
    rf505_asciicfg()
