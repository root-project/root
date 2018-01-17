#####################################
#
# 'BASIC FUNCTIONALITY' ROOT.RooFit tutorial macro #109
#
# Calculating chi^2 from histograms and curves in ROOT.RooPlots,
# making histogram of residual and pull distributions
#
#
#
# 07/2008 - Wouter Verkerke
#
# /

import ROOT


def rf109_chi2residpull():

    # S e t u p   m o d e l
    # ---------------------

    # Create observables
    x = ROOT.RooRealVar("x", "x", -10, 10)

    # Create Gaussian
    sigma = ROOT.RooRealVar("sigma", "sigma", 3, 0.1, 10)
    mean = ROOT.RooRealVar("mean", "mean", 0, -10, 10)
    gauss = ROOT.RooGaussian(
        "gauss", "gauss", x, mean, sigma)

    # Generate a sample of 1000 events with sigma=3
    data = gauss.generate(ROOT.RooArgSet(x), 10000)

    # Change sigma to 3.15
    sigma = 3.15

    # P l o t   d a t a   a n d   s l i g h t l y   d i s t o r t e d   m o d e l
    # ---------------------------------------------------------------------------

    # Overlay projection of gauss with sigma=3.15 on data with sigma=3.0
    frame1 = x.frame(ROOT.RooFit.Title(
        "Data with distorted Gaussian pdf"), ROOT.RooFit.Bins(40))
    data.plotOn(frame1, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
    gauss.plotOn(frame1)

    # C a l c u l a t e   c h i ^ 2
    # ------------------------------

    # Show the chi^2 of the curve w.r.t. the histogram
    # If multiple curves or datasets live in the frame you can specify
    # the name of the relevant curve and/or dataset in chiSquare()
    print "chi^2 = ", frame1.chiSquare()

    # S h o w   r e s i d u a l   a n d   p u l l   d i s t s
    # -------------------------------------------------------

    # Construct a histogram with the residuals of the data w.r.t. the curve
    hresid = frame1.residHist()

    # Construct a histogram with the pulls of the data w.r.t the curve
    hpull = frame1.pullHist()

    # Create a frame to draw the residual distribution and add the
    # distribution to the frame
    frame2 = x.frame(ROOT.RooFit.Title("Residual Distribution"))
    frame2.addPlotable(hresid, "P")

    # Create a frame to draw the pull distribution and add the distribution to
    # the frame
    frame3 = x.frame(ROOT.RooFit.Title("Pull Distribution"))
    frame3.addPlotable(hpull, "P")

    c = ROOT.TCanvas("rf109_chi2residpull", "rf109_chi2residpull", 900, 300)
    c.Divide(3)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.6)
    frame1.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.6)
    frame2.Draw()
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.15)
    frame3.GetYaxis().SetTitleOffset(1.6)
    frame3.Draw()

    c.SaveAs("rf109_chi2residpull.png")

if __name__ == "__main__":
    rf109_chi2residpull()
