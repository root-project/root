#####################################
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #314
#
# Working with parameterized ranges in a fit. ROOT.This an example of a
# fit with an acceptance that changes per-event
#
#  pdf = exp(-t/tau) with t[tmin,5]
#
#  where t and tmin are both observables in the dataset
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf314_paramfitrange():

    # D e f i n e   o b s e r v a b l e s   a n d   d e c a y   p d f
    # ---------------------------------------------------------------

    # Declare observables
    t = ROOT.RooRealVar("t", "t", 0, 5)
    tmin = ROOT.RooRealVar("tmin", "tmin", 0, 0, 5)

    # Make parameterized range in t : [tmin,5]
    t.setRange(tmin, ROOT.RooFit.RooConst(t.getMax()))

    # Make pdf
    tau = ROOT.RooRealVar("tau", "tau", -1.54, -10, -0.1)
    model = ROOT.RooExponential("model", "model", t, tau)

    # C r e a t e   i n p u t   d a t a
    # ------------------------------------

    # Generate complete dataset without acceptance cuts (for reference)
    dall = model.generate(ROOT.RooArgSet(t), 10000)

    # Generate a (fake) prototype dataset for acceptance limit values
    tmp = ROOT.RooGaussian("gmin", "gmin", tmin, ROOT.RooFit.RooConst(
        0), ROOT.RooFit.RooConst(0.5)).generate(ROOT.RooArgSet(tmin), 5000)

    # Generate dataset with t values that observe (t>tmin)
    dacc = model.generate(ROOT.RooArgSet(t), ROOT.RooFit.ProtoData(tmp))

    # F i t   p d f   t o   d a t a   i n   a c c e p t a n c e   r e g i o n
    # -----------------------------------------------------------------------

    r = model.fitTo(dacc, ROOT.RooFit.Save())

    # P l o t   f i t t e d   p d f   o n   f u l l   a n d   a c c e p t e d   d a t a
    # ---------------------------------------------------------------------------------

    # Make plot frame, datasets and overlay model
    frame = t.frame(ROOT.RooFit.Title("Fit to data with per-event acceptance"))
    dall.plotOn(frame, ROOT.RooFit.MarkerColor(ROOT.kRed),
                ROOT.RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame)
    dacc.plotOn(frame)

    # Print fit results to demonstrate absence of bias
    r.Print("v")

    c = ROOT.TCanvas("rf314_paramranges", "rf314_paramranges", 600, 600)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.6)
    frame.Draw()

    c.SaveAs("rf314_paramranges.png")

    return


if __name__ == "__main__":
    rf314_paramfitrange()
