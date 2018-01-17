# /
#
# 'LIKELIHOOD AND MINIMIZATION' ROOT.RooFit tutorial macro #604
#
# Fitting with constraints
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf604_constraints():

    # C r e a t e   m o d e l  a n d   d a t a s e t
    # ----------------------------------------------

    # Construct a Gaussian p.d.f
    x = ROOT.RooRealVar("x", "x", -10, 10)

    m = ROOT.RooRealVar("m", "m", 0, -10, 10)
    s = ROOT.RooRealVar("s", "s", 2, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gauss(x,m,s)", x, m, s)

    # Construct a flat p.d.f (polynomial of 0th order)
    poly = ROOT.RooPolynomial("poly", "poly(x)", x)

    # model = f*gauss + (1-f)*poly
    f = ROOT.RooRealVar("f", "f", 0.5, 0., 1.)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(gauss, poly), ROOT.RooArgList(f))

    # Generate small dataset for use in fitting below
    d = model.generate(ROOT.RooArgSet(x), 50)

    # C r e a t e   c o n s t r a i n t   p d f
    # -----------------------------------------

    # Construct Gaussian constraint p.d.f on parameter f at 0.8 with
    # resolution of 0.1
    fconstraint = ROOT.RooGaussian(
        "fconstraint", "fconstraint", f, ROOT.RooFit.RooConst(0.8), ROOT.RooFit.RooConst(0.1))

    # M E ROOT.T H O D   1   -   A d d   i n t e r n a l   c o n s t r a i n t   t o   m o d e l
    # -------------------------------------------------------------------------------------

    # Multiply constraint term with regular p.d.f using ROOT.RooProdPdf
    # Specify in fitTo() that internal constraints on parameter f should be
    # used

    # Multiply constraint with p.d.f
    modelc = ROOT.RooProdPdf(
        "modelc", "model with constraint", ROOT.RooArgList(model, fconstraint))

    # Fit model (without use of constraint term)
    r1 = model.fitTo(d, ROOT.RooFit.Save())

    # Fit modelc with constraint term on parameter f
    r2 = modelc.fitTo(d, ROOT.RooFit.Constrain(ROOT.RooArgSet(f)), ROOT.RooFit.Save())

    # M E ROOT.T H O D   2   -     S p e c i f y   e x t e r n a l   c o n s t r a i n t   w h e n   f i t t i n g
    # -------------------------------------------------------------------------------------------------------

    # Construct another Gaussian constraint p.d.f on parameter f at 0.8 with
    # resolution of 0.1
    fconstext = ROOT.RooGaussian("fconstext", "fconstext", f, ROOT.RooFit.RooConst(
        0.2), ROOT.RooFit.RooConst(0.1))

    # Fit with external constraint
    r3 = model.fitTo(d, ROOT.RooFit.ExternalConstraints(
        ROOT.RooArgSet(fconstext)), ROOT.RooFit.Save())

    # Print the fit results
    print "fit result without constraint (data generated at f=0.5)"
    r1.Print("v")
    print "fit result with internal constraint (data generated at f=0.5, is f=0.8+/-0.2)"
    r2.Print("v")
    print "fit result with (another) external constraint (data generated at f=0.5, is f=0.2+/-0.1)"
    r3.Print("v")


if __name__ == "__main__":
    rf604_constraints()
