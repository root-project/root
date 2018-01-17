#####################################
#
# 'SPECIAL PDFS' ROOT.RooFit tutorial macro #701
#
# Unbinned maximum likelihood fit of an efficiency eff(x) function to
# a dataset D(x,cut), cut is a category encoding a selection, which
# the efficiency as function of x should be described by eff(x)
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf701_efficiencyfit():
    # C o n s t r u c t   e f f i c i e n c y   f u n c t i o n   e ( x )
    # -------------------------------------------------------------------

    # Declare variables x,mean, with associated name, title, value and allowed
    # range
    x = ROOT.RooRealVar("x", "x", -10, 10)

    # Efficiency function eff(x;a,b)
    a = ROOT.RooRealVar("a", "a", 0.4, 0, 1)
    b = ROOT.RooRealVar("b", "b", 5)
    c = ROOT.RooRealVar("c", "c", -1, -10, 10)
    effFunc = ROOT.RooFormulaVar(
        "effFunc", "(1-a)+a*cos((x-c)/b)", ROOT.RooArgList(a, b, c, x))

    # C o n s t r u c t   c o n d i t i o n a l    e f f i c i e n c y   p d f   E ( c u t | x )
    # ------------------------------------------------------------------------------------------

    # Acceptance state cut (1 or 0)
    cut = ROOT.RooCategory("cut", "cutr")
    cut.defineType("accept", 1)
    cut.defineType("reject", 0)

    # Construct efficiency p.d.f eff(cut|x)
    effPdf = ROOT.RooEfficiency("effPdf", "effPdf", effFunc, cut, "accept")

    # G e n e r a t e   d a t a   ( x , u t )   f r o m   a   t o y   m o d e l
    # -----------------------------------------------------------------------------

    # Construct global shape p.d.f shape(x) and product model(x,cut) = eff(cut|x)*shape(x)
    # (These are _only_ needed to generate some toy MC here to be used later)
    shapePdf = ROOT.RooPolynomial("shapePdf", "shapePdf", x, ROOT.RooArgList(ROOT.RooFit.RooConst(-0.095)))
    model = ROOT.RooProdPdf("model", "model", ROOT.RooArgSet(shapePdf), ROOT.RooFit.Conditional(ROOT.RooArgSet(effPdf), ROOT.RooArgSet(cut)))

    # Generate some toy data from model
    data = model.generate(ROOT.RooArgSet(x, cut), 10000)

    # F i t   c o n d i t i o n a l   e f f i c i e n c y   p d f   t o   d a t a
    # --------------------------------------------------------------------------

    # Fit conditional efficiency p.d.f to data
    effPdf.fitTo(data, ROOT.RooFit.ConditionalObservables(ROOT.RooArgSet(x)))

    # P l o t   f i t t e d , a t a   e f f i c i e n c y
    # --------------------------------------------------------

    # Plot distribution of all events and accepted fraction of events on frame
    frame1 = x.frame(ROOT.RooFit.Bins(
        20), ROOT.RooFit.Title("Data (all, accepted)"))
    data.plotOn(frame1)
    data.plotOn(frame1, ROOT.RooFit.Cut("cut==cut::accept"), ROOT.RooFit.MarkerColor(
        ROOT.kRed), ROOT.RooFit.LineColor(ROOT.kRed))

    # Plot accept/reject efficiency on data overlay fitted efficiency curve
    frame2 = x.frame(ROOT.RooFit.Bins(
        20), ROOT.RooFit.Title("Fitted efficiency"))
    data.plotOn(frame2, ROOT.RooFit.Efficiency(cut))  # needs ROOT version >= 5.21
    effFunc.plotOn(frame2, ROOT.RooFit.LineColor(ROOT.kRed))

    # Draw all frames on a canvas
    ca = ROOT.TCanvas("rf701_efficiency", "rf701_efficiency", 800, 400)
    ca.Divide(2)
    ca.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.6)
    frame1.Draw()
    ca.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.4)
    frame2.Draw()

    ca.SaveAs("rf701_efficiencyfit.png")


if __name__ == "__main__":
    rf701_efficiencyfit()
