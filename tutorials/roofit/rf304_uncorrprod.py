# /
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #304
#
# Simple uncorrelated multi-dimensional p.d.f.s
#
# pdf = gauss(x,mx,sx) * gauss(y,my,sy)
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf304_uncorrprod():

    # C r e a t e   c o m p o n e n t   p d f s   i n   x   a n d   y
    # ----------------------------------------------------------------

    # Create two p.d.f.s gaussx(x,meanx,sigmax) gaussy(y,meany,sigmay) and its
    # variables
    x = ROOT.RooRealVar("x", "x", -5, 5)
    y = ROOT.RooRealVar("y", "y", -5, 5)

    meanx = ROOT.RooRealVar("mean1", "mean of gaussian x", 2)
    meany = ROOT.RooRealVar("mean2", "mean of gaussian y", -2)
    sigmax = ROOT.RooRealVar("sigmax", "width of gaussian x", 1)
    sigmay = ROOT.RooRealVar("sigmay", "width of gaussian y", 5)

    gaussx = ROOT.RooGaussian("gaussx", "gaussian PDF", x, meanx, sigmax)
    gaussy = ROOT.RooGaussian("gaussy", "gaussian PDF", y, meany, sigmay)

    # C o n s t r u c t   u n c o r r e l a t e d   p r o d u c t   p d f
    # -------------------------------------------------------------------

    # Multiply gaussx and gaussy into a two-dimensional p.d.f. gaussxy
    gaussxy = ROOT.RooProdPdf(
        "gaussxy", "gaussx*gaussy", ROOT.RooArgList(gaussx, gaussy))

    # S a m p l e   p d f , l o t   p r o j e c t i o n   o n   x   a n d   y
    # ---------------------------------------------------------------------------

    # Generate 10000 events in x and y from gaussxy
    data = gaussxy.generate(ROOT.RooArgSet(x, y), 10000)

    # Plot x distribution of data and projection of gaussxy x = Int(dy)
    # gaussxy(x,y)
    xframe = x.frame(ROOT.RooFit.Title("X projection of gauss(x)*gauss(y)"))
    data.plotOn(xframe)
    gaussxy.plotOn(xframe)

    # Plot x distribution of data and projection of gaussxy y = Int(dx)
    # gaussxy(x,y)
    yframe = y.frame(ROOT.RooFit.Title("Y projection of gauss(x)*gauss(y)"))
    data.plotOn(yframe)
    gaussxy.plotOn(yframe)

    # Make canvas and draw ROOT.RooPlots
    c = ROOT.TCanvas("rf304_uncorrprod", "rf304_uncorrprod", 800, 400)
    c.Divide(2)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    xframe.GetYaxis().SetTitleOffset(1.4)
    xframe.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    yframe.GetYaxis().SetTitleOffset(1.4)
    yframe.Draw()

    c.SaveAs("rf304_uncorrprod.png")


if __name__ == "__main__":
    rf304_uncorrprod()
