#####################################
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #313
#
# Working with parameterized ranges to define non-rectangular regions
# for fitting and integration
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf313_paramranges():

    # C r e a t e   3 D   p d f
    # -------------------------

    # Define observable (x,y,z)
    x = ROOT.RooRealVar("x", "x", 0, 10)
    y = ROOT.RooRealVar("y", "y", 0, 10)
    z = ROOT.RooRealVar("z", "z", 0, 10)

    # Define 3 dimensional pdf
    z0 = ROOT.RooRealVar("z0", "z0", -0.1, 1)
    px = ROOT.RooPolynomial("px", "px", x, ROOT.RooArgList(ROOT.RooFit.RooConst(0)))
    py = ROOT.RooPolynomial("py", "py", y, ROOT.RooArgList(ROOT.RooFit.RooConst(0)))
    pz = ROOT.RooPolynomial("pz", "pz", z, ROOT.RooArgList(z0))
    pxyz = ROOT.RooProdPdf("pxyz", "pxyz", ROOT.RooArgList(px, py, pz))

    # D e f i n e d   n o n - r e c t a n g u l a r   r e g i o n   R   i n   ( x , y , z )
    # -------------------------------------------------------------------------------------

    #
    # R = Z[0 - 0.1*Y^2] * Y[0.1*X - 0.9*X] * X[0 - 10]
    #

    # Construct range parameterized in "R" in y [ 0.1*x, 0.9*x ]
    ylo = ROOT.RooFormulaVar("ylo", "0.1*x", ROOT.RooArgList(x))
    yhi = ROOT.RooFormulaVar("yhi", "0.9*x", ROOT.RooArgList(x))
    y.setRange("R", ylo, yhi)

    # Construct parameterized ranged "R" in z [ 0, 0.1*y^2 ]
    zlo = ROOT.RooFormulaVar("zlo", "0.0*y", ROOT.RooArgList(y))
    zhi = ROOT.RooFormulaVar("zhi", "0.1*y*y", ROOT.RooArgList(y))
    z.setRange("R", zlo, zhi)

    # C a l c u l a t e   i n t e g r a l   o f   n o r m a l i z e d   p d f   i n   R
    # ----------------------------------------------------------------------------------

    # Create integral over normalized pdf model over x,y, in "R" region
    intPdf = pxyz.createIntegral(ROOT.RooArgSet(
        x, y, z), ROOT.RooArgSet(x, y, z), "R")

    # Plot value of integral as function of pdf parameter z0
    frame = z0.frame(ROOT.RooFit.Title(
        "Integral of pxyz over x,y, in region R"))
    intPdf.plotOn(frame)

    c = ROOT.TCanvas("rf313_paramranges", "rf313_paramranges", 600, 600)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.6)
    frame.Draw()

    c.SaveAs("rf313_paramranges.png")

    return


if __name__ == "__main__":
    rf313_paramranges()
