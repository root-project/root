#####################################
#
# 'BASIC FUNCTIONALITY' ROOT.RooFit tutorial macro #106
#
#  Adding boxes with parameters, to ROOT.RooPlots.
#  Decorating ROOT.RooPlots with arrows, etc...
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf106_plotdecoration():

    # S e t u p   m o d e l
    # ---------------------

    # Create observables
    x = ROOT.RooRealVar("x", "x", -10, 10)

    # Create Gaussian
    sigma = ROOT.RooRealVar("sigma", "sigma", 1, 0.1, 10)
    mean = ROOT.RooRealVar("mean", "mean", -3, -10, 10)
    gauss = ROOT.RooGaussian("gauss", "gauss", x, mean, sigma)

    # Generate a sample of 1000 events with sigma=3
    data = gauss.generate(ROOT.RooArgSet(x), 1000)

    # Fit pdf to data
    gauss.fitTo(data)

    # P l o t   p . d . f   a n d   d a t a
    # -------------------------------------

    # Overlay projection of gauss on data
    frame = x.frame(ROOT.RooFit.Name("xframe"), ROOT.RooFit.Title(
        "RooPlot with decorations"), ROOT.RooFit.Bins(40))
    data.plotOn(frame)
    gauss.plotOn(frame)

    # A d d   b o x   w i t h   p d f   p a r a m e t e r s
    # -----------------------------------------------------

    # Left edge of box starts at 55% of Xaxis)
    gauss.paramOn(frame, ROOT.RooFit.Layout(0.55))

    # A d d   b o x   w i t h   d a t a   s t a t i s t i c s
    # -------------------------------------------------------

    # X size of box is from 55% to 99% of Xaxis range, of box is at 80% of
    # Yaxis range)
    data.statOn(frame, ROOT.RooFit.Layout(0.55, 0.99, 0.8))

    # A d d   t e x t   a n d   a r r o w
    # -----------------------------------

    # Add text to frame
    txt = ROOT.TText(2, 100, "Signal")
    txt.SetTextSize(0.04)
    txt.SetTextColor(ROOT.kRed)
    frame.addObject(txt)

    # Add arrow to frame
    arrow = ROOT.TArrow(2, 100, -1, 50, 0.01, "|>")
    arrow.SetLineColor(ROOT.kRed)
    arrow.SetFillColor(ROOT.kRed)
    arrow.SetLineWidth(3)
    frame.addObject(arrow)

    # P e r s i s t   f r a m e   w i t h   a l l   d e c o r a t i o n s   i n   R O O ROOT.T   f i l e
    # ---------------------------------------------------------------------------------------------

    f = ROOT.TFile("rf106_plotdecoration.root", "RECREATE")
    frame.Write()
    f.Close()

    # ROOT.To read back and plot frame with all decorations in clean root session do
    # root> ROOT.TFile f("rf106_plotdecoration.root")
    # root>  xframe.Draw()

    c = ROOT.TCanvas("rf106_plotdecoration", "rf106_plotdecoration", 600, 600)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.6)
    frame.Draw()

    c.SaveAs("rf106_plotdecoration.png")

if __name__ == "__main__":
    rf106_plotdecoration()
