#####################################
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #307
#
# Complete example with use of full p.d.f. with per-event errors
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf307_fullpereventerrors():
    # B - p h y s i c s   p d f   w i t h   p e r - e v e n t  G a u s s i a n   r e s o l u t i o n
    # ----------------------------------------------------------------------------------------------

    # Observables
    dt = ROOT.RooRealVar("dt", "dt", -10, 10)
    dterr = ROOT.RooRealVar("dterr", "per-event error on dt", 0.01, 10)

    # Build a gaussian resolution model scaled by the per-error =
    # gauss(dt,bias,sigma*dterr)
    bias = ROOT.RooRealVar("bias", "bias", 0, -10, 10)
    sigma = ROOT.RooRealVar(
        "sigma", "per-event error scale factor", 1, 0.1, 10)
    gm = ROOT.RooGaussModel(
        "gm1", "gauss model scaled bt per-event error", dt, bias, sigma, dterr)

    # Construct decay(dt) (x) gauss1(dt|dterr)
    tau = ROOT.RooRealVar("tau", "tau", 1.548)
    decay_gm = ROOT.RooDecay("decay_gm", "decay", dt,
                             tau, gm, ROOT.RooDecay.DoubleSided)

    # C o n s t r u c t   e m p i r i c a l   p d f   f o r   p e r - e v e n t   e r r o r
    # -----------------------------------------------------------------

    # Use landau p.d.f to get empirical distribution with long tail
    pdfDtErr = ROOT.RooLandau("pdfDtErr", "pdfDtErr", dterr, ROOT.RooFit.RooConst(
        1), ROOT.RooFit.RooConst(0.25))
    expDataDterr = pdfDtErr.generate(ROOT.RooArgSet(dterr), 10000)

    # Construct a histogram pdf to describe the shape of the dtErr distribution
    expHistDterr = expDataDterr.binnedClone()
    pdfErr = ROOT.RooHistPdf(
        "pdfErr", "pdfErr", ROOT.RooArgSet(dterr), expHistDterr)

    # C o n s t r u c t   c o n d i t i o n a l   p r o d u c t   d e c a y _ d m ( d t | d t e r r ) * p d f ( d t e r r )
    # ----------------------------------------------------------------------------------------------------------------------

    # Construct production of conditional decay_dm(dt|dterr) with empirical
    # pdfErr(dterr)
    model = ROOT.RooProdPdf("model", "model", ROOT.RooArgSet(
        pdfErr), ROOT.RooFit.Conditional(ROOT.RooArgSet(decay_gm), ROOT.RooArgSet(dt)))

    # (Alternatively you could also use the landau shape pdfDtErr)
    # ROOT.RooProdPdf model("model", "model",pdfDtErr,
    # ROOT.RooFit.Conditional(decay_gm,dt))

    # S a m p l e, i t   a n d   p l o t   p r o d u c t   m o d e l
    # ------------------------------------------------------------------

    # Specify external dataset with dterr values to use model_dm as
    # conditional p.d.f.
    data = model.generate(ROOT.RooArgSet(dt, dterr), 10000)

    # F i t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
    # ---------------------------------------------------------------------

    # Specify dterr as conditional observable
    model.fitTo(data)

    # P l o t   c o n d i t i o n a l   d e c a y _ d m ( d t | d t e r r )
    # ---------------------------------------------------------------------

    # Make two-dimensional plot of conditional p.d.f in (dt,dterr)
    hh_model = model.createHistogram("hh_model", dt, ROOT.RooFit.Binning(
        50), ROOT.RooFit.YVar(dterr, ROOT.RooFit.Binning(50)))
    hh_model.SetLineColor(ROOT.kBlue)

    # Make projection of data an dt
    frame = dt.frame(ROOT.RooFit.Title("Projection of model(dt|dterr) on dt"))
    data.plotOn(frame)
    model.plotOn(frame)

    # Draw all frames on canvas
    c = ROOT.TCanvas("rf307_fullpereventerrors",
                     "rf307_fullpereventerrors", 800, 400)
    c.Divide(2)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.20)
    hh_model.GetZaxis().SetTitleOffset(2.5)
    hh_model.Draw("surf")
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.6)
    frame.Draw()

    c.SaveAs("rf307_fullpereventerrors.png")

if __name__ == "__main__":
    rf307_fullpereventerrors()
