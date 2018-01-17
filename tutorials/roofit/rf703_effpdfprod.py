#####################################
#
# 'SPECIAL PDFS' ROOT.RooFit tutorial macro #703
#
# Using a product of an (acceptance) efficiency and a p.d.f as p.d.f.
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf703_effpdfprod():
    # D e f i n e   o b s e r v a b l e s   a n d   d e c a y   p d f
    # ---------------------------------------------------------------

    # Declare observables
    t = ROOT.RooRealVar("t", "t", 0, 5)

    # Make pdf
    tau = ROOT.RooRealVar("tau", "tau", -1.54, -4, -0.1)
    model = ROOT.RooExponential("model", "model", t, tau)

    # D e f i n e   e f f i c i e n c y   f u n c t i o n
    # ---------------------------------------------------

    # Use error function to simulate turn-on slope
    eff = ROOT.RooFormulaVar("eff", "0.5*(TMath::Erf((t-1)/0.5)+1)", ROOT.RooArgList(t))

    # D e f i n e   d e c a y   p d f   w i t h   e f f i c i e n c y
    # ---------------------------------------------------------------

    # Multiply pdf(t) with efficiency in t
    modelEff = ROOT.RooEffProd("modelEff", "model with efficiency", model, eff)

    # P l o t   e f f i c i e n c y , d f
    # ----------------------------------------

    frame1 = t.frame(ROOT.RooFit.Title("Efficiency"))
    eff.plotOn(frame1, ROOT.RooFit.LineColor(ROOT.kRed))

    frame2 = t.frame(ROOT.RooFit.Title("Pdf with and without efficiency"))

    model.plotOn(frame2, ROOT.RooFit.LineStyle(ROOT.kDashed))
    modelEff.plotOn(frame2)

    # G e n e r a t e   t o y   d a t a , i t   m o d e l E f f   t o   d a t a
    # ------------------------------------------------------------------------------

    # Generate events. If the input pdf has an internal generator, internal generator
    # is used and an accept/reject sampling on the efficiency is applied.
    data = modelEff.generate(ROOT.RooArgSet(t), 10000)

    # Fit pdf. ROOT.The normalization integral is calculated numerically.
    modelEff.fitTo(data)

    # Plot generated data and overlay fitted pdf
    frame3 = t.frame(ROOT.RooFit.Title("Fitted pdf with efficiency"))
    data.plotOn(frame3)
    modelEff.plotOn(frame3)

    c = ROOT.TCanvas("rf703_effpdfprod", "rf703_effpdfprod", 1200, 400)
    c.Divide(3)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.4)
    frame1.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.6)
    frame2.Draw()
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.15)
    frame3.GetYaxis().SetTitleOffset(1.6)
    frame3.Draw()

    c.SaveAs("rf703_effpdfprod.png")


if __name__ == "__main__":
    rf703_effpdfprod()
