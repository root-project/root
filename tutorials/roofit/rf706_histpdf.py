#####################################
#
# 'SPECIAL PDFS' ROOT.RooFit tutorial macro #706
#
# Histogram based p.d.f.s and functions
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf706_histpdf():
    # C r e a t e   p d f   f o r   s a m p l i n g
    # ---------------------------------------------

    x = ROOT.RooRealVar("x", "x", 0, 20)
    p = ROOT.RooPolynomial("p", "p", x, ROOT.RooArgList(ROOT.RooFit.RooConst(
        0.01), ROOT.RooFit.RooConst(-0.01), ROOT.RooFit.RooConst(0.0004)))

    # C r e a t e   l o w   s t a t s   h i s t o g r a m
    # ---------------------------------------------------

    # Sample 500 events from p
    x.setBins(20)
    data1 = p.generate(ROOT.RooArgSet(x), 500)

    # Create a binned dataset with 20 bins and 500 events
    hist1 = data1.binnedClone()

    # Represent data in dh as pdf in x
    histpdf1 = ROOT.RooHistPdf("histpdf1", "histpdf1", ROOT.RooArgSet(x), hist1, 0)

    # Plot unbinned data and histogram pdf overlaid
    frame1 = x.frame(ROOT.RooFit.Title(
        "Low statistics histogram pdf"), ROOT.RooFit.Bins(100))
    data1.plotOn(frame1)
    histpdf1.plotOn(frame1)

    # C r e a t e   h i g h   s t a t s   h i s t o g r a m
    # -----------------------------------------------------

    # Sample 100000 events from p
    x.setBins(10)
    data2 = p.generate(ROOT.RooArgSet(x), 100000)

    # Create a binned dataset with 10 bins and 100K events
    hist2 = data2.binnedClone()

    # Represent data in dh as pdf in x, 2nd order interpolation
    histpdf2 = ROOT.RooHistPdf("histpdf2", "histpdf2", ROOT.RooArgSet(x), hist2, 2)

    # Plot unbinned data and histogram pdf overlaid
    frame2 = x.frame(ROOT.RooFit.Title(
        "High stats histogram pdf with interpolation"), ROOT.RooFit.Bins(100))
    data2.plotOn(frame2)
    histpdf2.plotOn(frame2)

    c = ROOT.TCanvas("rf706_histpdf", "rf706_histpdf", 800, 400)
    c.Divide(2)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.4)
    frame1.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.8)
    frame2.Draw()

    c.SaveAs("rf706_histpdf.png")


if __name__ == "__main__":
    rf706_histpdf()
